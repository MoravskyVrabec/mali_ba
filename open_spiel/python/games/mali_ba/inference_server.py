"""Batched GPU inference server for Mali-Ba self-play actors.

Why this exists
---------------
Actors run TensorFlow single-threaded (OMP_NUM_THREADS=1, set deliberately so 25
actor processes don't oversubscribe the CPU). Measured on an RTX 4090 host, one
batch-1 forward pass through both nets costs:

    threads:      1      2      4      8     16     32
    ms/call:  37.11  24.42  15.71  11.46   8.23   9.78

At 1 thread that ~37ms is essentially the entire cost of self-play: actor
self-timing puts 99.8% of move-loop wall time inside bot.mcts_search, and within
that the forward pass dominates. Batching does NOT help on CPU (batch-32
single-threaded is 36.6ms/sample). The same nets on the GPU:

    batch:        1      8     32     64    128    256
    ms/sample: 1.360  0.270  0.181  0.202  0.254  0.247

So one process owning the GPU and batching requests from every actor replaces a
~37ms CPU call with a ~0.18ms GPU slice. The practical ceiling is GPU throughput,
~5500 evals/s at batch 32, versus ~676 evals/s for 25 CPU actors: roughly 8x.
Actors become mostly blocked rather than compute-bound, so more of them fit.

Design
------
Each client owns exactly one request slot, which is sufficient because MCTS is
synchronous: an actor has at most one outstanding evaluation. That removes any
need for a queue or ring buffer.

  * Observations, policies and values live in mp.RawArray shared memory, so a
    21,375-float observation is never pickled or copied through a pipe.
  * A client writes its observation, then raises its request flag. On x86 the
    store ordering makes this safe without a lock; the server clears the flag
    before signalling completion, so no stale flag can be observed.
  * The server spins over the flag array (one core, deliberately -- it is the
    thing we want to keep fed), gathers every pending slot into one batch, runs a
    single traced forward pass, writes the results back and sets each waiter's
    event. Clients block on an Event rather than spinning, so 25 waiting actors
    cost no CPU.
  * Natural batching falls out of the synchronous clients: while the server is
    busy, every other actor queues up, so the next batch is large.
"""

import os
import time
import multiprocessing as mp
import numpy as np

# Sentinel pushed to the weights queue to make the server exit.
STOP = "__stop__"


class InferenceArena:
    """Shared-memory request/response slots, one per client."""

    def __init__(self, n_slots, obs_size, n_actions, n_players):
        self.n_slots = n_slots
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.n_players = n_players
        # RawArray: no lock, we synchronise with the flag + event pair below.
        self.obs = mp.RawArray("f", n_slots * obs_size)
        self.policy = mp.RawArray("f", n_slots * n_actions)
        self.value = mp.RawArray("f", n_slots * n_players)
        self.req_flag = mp.RawArray("b", n_slots)      # 1 = request pending
        # Per-slot wake-up: a plain semaphore, NOT mp.Event. Event.set() goes
        # through Condition.notify(), which BLOCKS until the woken process has
        # actually been scheduled -- a full context-switch round trip per actor,
        # serialised inside the server. Measured ~28 us per actor on the desktop
        # and ~100 us on the laptop, i.e. 31-44% of every batch cycle. Releasing a
        # semaphore is one non-blocking syscall; the actor wakes on its own time.
        self.wake = [mp.Semaphore(0) for _ in range(n_slots)]
        # Request/response sequence numbers. A semaphore COUNTS, so a release left
        # over from a request that timed out would otherwise satisfy the next
        # request with stale results. Clients only accept a wake-up whose
        # response sequence matches their current request.
        self.req_seq = mp.RawArray("q", n_slots)
        self.resp_seq = mp.RawArray("q", n_slots)
        self.slot_lock = mp.Lock()
        self.next_slot = mp.Value("i", 0)
        # Server-side stats, readable by the parent for logging.
        self.stat_batches = mp.Value("q", 0)
        self.stat_requests = mp.Value("q", 0)
        self.stat_infer_ns = mp.Value("q", 0)
        self.stat_gather_ns = mp.Value("q", 0)
        self.stat_scatter_ns = mp.Value("q", 0)
        self.stat_poll_ns = mp.Value("q", 0)
        self.ready = mp.Event()                        # server has loaded weights
        # Set as soon as the server confirms it has a GPU, before any weights
        # arrive. A remote worker cannot wait for `ready` (weights only reach it
        # via its own actors' jobs), so this is what the parent checks to decide
        # whether using the server is safe at all.
        self.gpu_ok = mp.Event()

    def claim_slot(self):
        """Reserve a slot index for one client. Raises if the arena is full."""
        with self.slot_lock:
            i = self.next_slot.value
            if i >= self.n_slots:
                raise RuntimeError(
                    f"InferenceArena exhausted: {self.n_slots} slots already claimed")
            self.next_slot.value = i + 1
            return i

    def stats(self):
        b = self.stat_batches.value
        r = self.stat_requests.value
        ns = self.stat_infer_ns.value
        return {
            "batches": b,
            "requests": r,
            "avg_batch": (r / b) if b else 0.0,
            "avg_infer_ms": (ns / b / 1e6) if b else 0.0,
            "ms_per_request": (ns / r / 1e6) if r else 0.0,
            "gather_ms": (self.stat_gather_ns.value / b / 1e6) if b else 0.0,
            "scatter_ms": (self.stat_scatter_ns.value / b / 1e6) if b else 0.0,
            "poll_ms": (self.stat_poll_ns.value / b / 1e6) if b else 0.0,
        }


class InferenceClient:
    """Actor-side handle. One evaluation in flight at a time, by construction."""

    def __init__(self, arena, shape, slot=None):
        self.arena = arena
        self.shape = tuple(shape)
        self.slot = arena.claim_slot() if slot is None else slot
        o, a, v = arena.obs_size, arena.n_actions, arena.n_players
        # numpy views onto this client's slice of the shared buffers
        self._obs_view = np.frombuffer(
            arena.obs, dtype=np.float32, count=o, offset=self.slot * o * 4)
        self._pol_view = np.frombuffer(
            arena.policy, dtype=np.float32, count=a, offset=self.slot * a * 4)
        self._val_view = np.frombuffer(
            arena.value, dtype=np.float32, count=v, offset=self.slot * v * 4)
        # Legacy arena: a run started before the semaphore protocol hands its
        # arena (pickled, per-slot mp.Event in `done`) to actors that re-import
        # this file from disk -- e.g. an actor respawned after the file was
        # updated mid-run. Speak the old protocol to it rather than failing and
        # silently dropping to CPU inference.
        self._legacy = not hasattr(arena, "wake")
        if self._legacy:
            self._event = arena.done[self.slot]
            return
        self._sem = arena.wake[self.slot]
        # A recycled slot may carry a wake-up left by its previous owner's last,
        # timed-out request. Drain it, and continue the slot's sequence so an
        # in-flight response to the old owner can never match a new request.
        while self._sem.acquire(False):
            pass
        self._seq = max(arena.req_seq[self.slot], arena.resp_seq[self.slot])

    def infer(self, obs_flat, timeout=120.0):
        """Submit one observation, block until served. Returns (policy, value) copies."""
        if self._legacy:
            return self._infer_legacy(obs_flat, timeout)
        slot = self.slot
        self._seq += 1
        seq = self._seq
        self._obs_view[:] = obs_flat                   # write payload first
        self.arena.req_seq[slot] = seq
        self.arena.req_flag[slot] = 1                  # then publish
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._sem.acquire(True, remaining):
                self.arena.req_flag[slot] = 0
                raise TimeoutError(
                    f"inference slot {slot} timed out after {timeout}s "
                    f"(is the inference server alive?)")
            if self.arena.resp_seq[slot] == seq:
                return self._pol_view.copy(), self._val_view.copy()
            # otherwise: a stale wake-up from an earlier, abandoned request -- keep waiting

    def _infer_legacy(self, obs_flat, timeout):
        """The pre-semaphore protocol, for an arena built by an older server."""
        self._obs_view[:] = obs_flat
        self._event.clear()
        self.arena.req_flag[self.slot] = 1
        if not self._event.wait(timeout):
            self.arena.req_flag[self.slot] = 0
            raise TimeoutError(
                f"inference slot {self.slot} timed out after {timeout}s "
                f"(is the inference server alive?)")
        return self._pol_view.copy(), self._val_view.copy()


def server_loop(arena, shape, game_params, weights_queue, stop_event,
                max_batch=32, spin_before_sleep=200000, idle_sleep=0.001,
                log_every=0, gpu_memory_limit_mb=4096, jit_compile=True,
                bucket_step=8):
    """Run in its own process. Owns the GPU and serves batched inference.

    weights_queue carries (policy_weights, value_weights) tuples, or STOP.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import pyspiel
    from pyspiel.mali_ba import log, LogLevel
    from mali_ba.training_utils import (create_mali_ba_policy_network,
                                        create_mali_ba_value_network)

    # Cap this process's VRAM. The trainer holds its own TF context on the same GPU
    # and both grow greedily: measured 23,920 of 24,564 MiB in use with growth-only,
    # leaving no headroom for a bigger batch or network. Inference needs very little
    # -- weights are a few MB and the largest batch here is 32 -- so a hard cap keeps
    # the trainer's allocation safe.
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for g in gpus:
        try:
            if gpu_memory_limit_mb:
                tf.config.set_logical_device_configuration(
                    g, [tf.config.LogicalDeviceConfiguration(
                        memory_limit=int(gpu_memory_limit_mb))])
            else:
                tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass
    if gpus:
        log(LogLevel.INFO,
            f"InferenceServer: COMPUTE DEVICE = GPU x{len(gpus)}, max_batch={max_batch}, "
            f"vram_cap={gpu_memory_limit_mb or 'growth'}MB")
        # analyze_log reads this to print the taskset command that pins the server
        # to the large-L3 cores (2026-10-05: ~5,100 -> ~8,000 evals/s on a 7950X3D).
        import socket
        log(LogLevel.INFO,
            f"InferenceServer: pid {os.getpid()} on host {socket.gethostname()}")
        arena.gpu_ok.set()
    else:
        # REFUSE to serve. A CPU-only batched server is not merely no better than
        # per-actor inference, it is dramatically worse: every client serialises
        # through one single-threaded inference stream. Measured, batch-32
        # single-threaded takes ~1170ms, so N clients share ~27 evals/s, against
        # ~27 evals/s PER actor when each does its own. With 28 actors that is a
        # ~28x regression. Exiting here leaves gpu_ok unset, which the parent takes
        # as "do not hand the arena to actors", so they fall back to local
        # inference -- slow, but the correct slow.
        log(LogLevel.ERROR,
            f"InferenceServer: COMPUTE DEVICE = CPU ONLY -- no GPU visible to "
            f"TensorFlow {tf.__version__} (built_with_cuda="
            f"{tf.test.is_built_with_cuda()}). Refusing to serve: a batched CPU "
            f"server would serialise every actor through one inference stream and "
            f"be far slower than letting each actor infer locally. If this machine "
            f"has an NVIDIA GPU the CUDA runtime wheels are probably missing: "
            f"pip install 'tensorflow[and-cuda]=={tf.__version__}'")
        return

    game = pyspiel.load_game("mali_ba", game_params)
    pm = create_mali_ba_policy_network(shape, game.num_distinct_actions())
    vm = create_mali_ba_value_network(shape, game.num_players())

    # Batch sizes are BUCKETED and padded up to the bucket, so only a handful of
    # graphs exist and all are traced before serving starts. Tracing costs ~1s, so
    # a graph per distinct batch size (1..max_batch) made the first call at each
    # size ruinously slow -- measured worse than not batching at all. Padding a
    # batch of 11 up to 16 wastes a few GPU-milliseconds; retracing wasted ~1000.
    #
    # Above 16 the buckets step linearly (bucket_step) rather than doubling. With
    # doubling, a batch of 48 under max_batch=56 was padded straight to 56, so 17%
    # of every full GPU call processed throwaway rows; saturated servers sit at
    # exactly those sizes. Finer buckets cost a few more graphs to warm at startup.
    def _buckets(n):
        out, b = [], 1
        while b < min(n, 16):
            out.append(b)
            b *= 2
        b = 16
        while b < n:
            out.append(b)
            b += max(1, bucket_step)
        out.append(n)
        return sorted(set(x for x in out if x <= n))

    BUCKETS = _buckets(max_batch)
    traced = {}
    _use_jit = [bool(jit_compile)]      # list so the fallback below can flip it

    def bucket_for(bs):
        for b in BUCKETS:
            if bs <= b:
                return b
        return BUCKETS[-1]

    def get_traced(bs):
        fn = traced.get(bs)
        if fn is None:
            spec = tf.TensorSpec(shape=(bs, *shape), dtype=tf.float32)

            # XLA fuses the many small conv/bn/relu kernels into a few larger
            # ones. At these batch sizes the GPU call is dominated by per-kernel
            # launch overhead rather than arithmetic, which is what fusion removes.
            @tf.function(input_signature=[spec], jit_compile=_use_jit[0])
            def _f(batch):
                return pm(batch, training=False), vm(batch, training=False)

            traced[bs] = fn = _f
        return fn

    n_slots = arena.n_slots
    obs_size, n_actions, n_players = arena.obs_size, arena.n_actions, arena.n_players
    obs_all = np.frombuffer(arena.obs, dtype=np.float32).reshape(n_slots, obs_size)
    pol_all = np.frombuffer(arena.policy, dtype=np.float32).reshape(n_slots, n_actions)
    val_all = np.frombuffer(arena.value, dtype=np.float32).reshape(n_slots, n_players)
    flags = arena.req_flag

    have_weights = False
    while not have_weights and not stop_event.is_set():
        try:
            item = weights_queue.get(timeout=1.0)
        except Exception:
            continue
        if item == STOP:
            return
        pw, vw = item
        pm.set_weights(pw)
        vm.set_weights(vw)
        have_weights = True
    if not have_weights:
        return

    # Warm every bucket so no actor ever eats a tracing stall mid-game. With XLA
    # this is also where each size is compiled. If XLA is unavailable on this
    # machine (e.g. ptxas missing from the CUDA wheels), fall back to plain graphs
    # rather than refusing to serve.
    _w0 = time.time()
    try:
        for bs in BUCKETS:
            get_traced(bs)(tf.zeros((bs, *shape), dtype=tf.float32))
    except Exception as e:
        if not _use_jit[0]:
            raise
        log(LogLevel.WARN,
            f"InferenceServer: XLA compilation failed ({str(e)[:160]}); "
            f"falling back to non-XLA graphs.")
        _use_jit[0] = False
        traced.clear()
        for bs in BUCKETS:
            get_traced(bs)(tf.zeros((bs, *shape), dtype=tf.float32))
    log(LogLevel.INFO,
        f"InferenceServer: warmed {len(BUCKETS)} batch sizes in "
        f"{time.time() - _w0:.1f}s (XLA {'on' if _use_jit[0] else 'off'}).")
    arena.ready.set()
    log(LogLevel.INFO, "InferenceServer: ready, weights loaded and graphs traced.")

    batch_buf = np.empty((max_batch, *shape), dtype=np.float32)
    flat_buf = batch_buf.reshape(max_batch, -1)   # view: same memory, one row per request
    wake = arena.wake
    req_seq = arena.req_seq
    resp_seq = arena.resp_seq
    log(LogLevel.INFO, f"InferenceServer: batch buckets {BUCKETS}")
    last_log = time.time()

    # Round-robin serving cursor. See the fair-selection block below: without it,
    # any slot numbered above max_batch can starve indefinitely.
    serve_cursor = 0
    if n_slots > max_batch:
        log(LogLevel.WARN,
            f"InferenceServer: {n_slots} slots but max_batch={max_batch}; serving "
            f"round-robin so no slot starves. Raise inference_max_batch to "
            f">= {n_slots} to serve every pending request in one batch.")

    while not stop_event.is_set():
        # Drain any newly published weights (keep only the freshest).
        newest = None
        while True:
            try:
                item = weights_queue.get_nowait()
            except Exception:
                break
            if item == STOP:
                stop_event.set()
                break
            newest = item
        if newest is not None:
            pm.set_weights(newest[0])
            vm.set_weights(newest[1])
        if stop_event.is_set():
            break

        # BUSY-SPIN while clients are mid-turn. An earlier version slept
        # idle_sleep=0.2ms here, but Linux rounds short sleeps up to 1-2ms, which
        # showed up as 4.31ms of dead time per batch -- more than the 3.38ms the
        # GPU itself needed, and it capped throughput below the CPU baseline.
        # Spinning costs one core, which is the correct trade: this loop is the
        # thing every actor is blocked on. Only after a long dry spell (run over,
        # or actors restarting) do we fall back to sleeping.
        _p0 = time.perf_counter_ns()
        pending = None
        spins = 0
        while not stop_event.is_set():
            pending = [i for i in range(n_slots) if flags[i]]
            if pending:
                break
            spins += 1
            if spins >= spin_before_sleep:
                time.sleep(idle_sleep)
                spins = 0
                break
        poll_ns = time.perf_counter_ns() - _p0
        if not pending:
            with arena.stat_poll_ns.get_lock():
                arena.stat_poll_ns.value += poll_ns
            continue
        if len(pending) > max_batch:
            # FAIR SELECTION (fixed 2026-09-29).
            # `pending` comes out of the spin scan in ascending slot order, so the
            # previous `pending[:max_batch]` always served the LOWEST slots. With
            # 64 actors through a 32-wide window the batch was full every cycle
            # (observed avg batch 32.0), so slots above max_batch were only served
            # in the rare lull -- slots 57-67 hit the client's 120s timeout and
            # their actor processes died, 24 of them in 15 minutes. Crashes landed
            # at move 0, where every actor requests at once and contention peaks.
            #
            # Rotating the start point bounds the wait: every slot is served
            # within ceil(n_slots / max_batch) cycles (3 cycles ~= 7ms here)
            # regardless of how far n_slots exceeds max_batch.
            start = 0
            for j, sl in enumerate(pending):
                if sl >= serve_cursor:
                    start = j
                    break
            pending = (pending[start:] + pending[:start])[:max_batch]
        # Advance past the last slot served, so the next cycle begins after it.
        serve_cursor = (pending[-1] + 1) % n_slots

        bs = len(pending)
        _g0 = time.perf_counter_ns()
        # Per-row copies. A single np.take over the batch was tried and measured
        # SLOWER (0.35 -> 0.68 ms at batch 48): fancy indexing out of the shared
        # RawArray costs more than these contiguous row copies.
        # Sequence numbers are read with the payload: the client writes the
        # observation, then req_seq, then the flag, so all are current here.
        seqs = [req_seq[slot] for slot in pending]
        for k, slot in enumerate(pending):
            flat_buf[k] = obs_all[slot]
        idx = np.asarray(pending, dtype=np.intp)
        padded = bucket_for(bs)
        if padded > bs:
            batch_buf[bs:padded] = batch_buf[0]      # pad with a repeat, results ignored

        gather_ns = time.perf_counter_ns() - _g0
        t0 = time.perf_counter_ns()
        pol, val = get_traced(padded)(tf.constant(batch_buf[:padded]))
        pol = pol.numpy()
        val = val.numpy()
        infer_ns = time.perf_counter_ns() - t0

        _s0 = time.perf_counter_ns()
        pol_all[idx] = pol[:bs]      # whole-batch copies into shared memory
        val_all[idx] = val[:bs]
        for slot, sq in zip(pending, seqs):
            resp_seq[slot] = sq      # stamp which request these results answer
            # Clear the flag only if the client has not already posted a newer
            # request on this slot (possible if it timed out and moved on).
            if req_seq[slot] == sq:
                flags[slot] = 0
            wake[slot].release()     # non-blocking; the actor wakes on its own
        scatter_ns = time.perf_counter_ns() - _s0

        with arena.stat_batches.get_lock():
            arena.stat_batches.value += 1
        with arena.stat_requests.get_lock():
            arena.stat_requests.value += bs
        with arena.stat_infer_ns.get_lock():
            arena.stat_infer_ns.value += infer_ns
        with arena.stat_gather_ns.get_lock():
            arena.stat_gather_ns.value += gather_ns
        with arena.stat_scatter_ns.get_lock():
            arena.stat_scatter_ns.value += scatter_ns
        with arena.stat_poll_ns.get_lock():
            arena.stat_poll_ns.value += poll_ns

        if log_every and time.time() - last_log >= log_every:
            s = arena.stats()
            log(LogLevel.INFO,
                f"InferenceServer: {s['requests']:,} evals in {s['batches']:,} batches "
                f"(avg batch {s['avg_batch']:.1f}, {s['avg_infer_ms']:.2f} ms/batch, "
                f"{s['ms_per_request']:.3f} ms/eval) "
                f"[per batch: gather {s['gather_ms']:.2f}, scatter {s['scatter_ms']:.2f}, "
                f"waiting for requests {s['poll_ms']:.2f} ms]")
            last_log = time.time()

    log(LogLevel.INFO, "InferenceServer: stopping.")
