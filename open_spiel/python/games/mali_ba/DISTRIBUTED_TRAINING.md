# Distributed Training with Remote Actors

This document describes how to extend a single-machine OpenSpiel AlphaZero-style
training script to use additional machines (or extra processes on the same machine)
as actor workers. The approach was developed for Mali-Ba and can be adapted for
any game that uses the same trainer/actor/learner multiprocess pattern.

---

# Mali-Ba GCP Operations Playbook

Everything below is specific to the actual live Mali-Ba setup (not the generic
guide further down). Written 2026-07-16 after the first full end-to-end GCP
session -- covers the current architecture, every command you'll need
day-to-day, a full restart checklist, and every non-obvious bug found and
fixed that evening, so a future session (or future me) doesn't have to
re-discover any of it.

## 1. Current architecture

- **Trainer** runs on your **desktop** (4090), not a GCP VM. GCP GPU (T4)
  capacity was exhausted in every zone tried on 2026-07-16
  (us-central1-a/b/c/f, us-west1-b, us-east1-c) -- not a quota problem, a
  genuine stockout. The desktop-as-trainer + GCP CPU workers design sidesteps
  GPU quota/capacity entirely. The GCP GPU main-vm path (`create_main_vm.sh`,
  `main_vm_setup.sh`) is still there and working if you want to revive it
  later -- see `HANDOFF-GCP-SETUP.md` for the setup walkthrough.
- **Workers** are GCP spot CPU VMs (`mali-ba-worker-*`) reaching the trainer
  over **Tailscale** (not a public IP / port-forward) -- desktop, workers, and
  a future GPU main-vm are all just nodes on the same tailnet and can reach
  each other directly regardless of which one is "the trainer" at any given
  time.
- **GCP project:** `mali-ba-training`. Account: `robert.charles.parker@gmail.com`.
- Config for all `gcp/*.sh` scripts lives in `gcp/env.sh` (gitignored --
  contains secrets). Copy `env.sh.example` if it's ever missing.

## 2. Current live resources (as of 2026-07-16 evening)

| Resource | Value |
|---|---|
| Desktop Tailscale IP | `100.92.31.37` (hostname `robp-system-ubuntu`) |
| `mali-ba-worker-1` | `us-central1-b`, actor-id-block `100000`, Tailscale `100.82.48.60` |
| `mali-ba-worker-20260716-2000` | `us-central1-b`, actor-id-block `200000`, Tailscale `100.96.40.125` |
| Golden image family | `mali-ba-worker-base` (see §6 -- was mid-rebuild as of writing; confirm the rebuild finished before trusting it) |
| Instance template | `mali-ba-worker-template` |
| Project metadata `trainer-tailscale-ip` | `100.92.31.37` (update via `set_trainer_host.sh` if the trainer moves) |
| Project metadata `tailscale-authkey` | set (reusable key, in `env.sh` as `TS_AUTHKEY`) -- lets new workers auto-authenticate with no browser click |
| Training authkey (queue server) | `malibatraining2024` (train_mali_ba.py's built-in default -- not the same as `AUTHKEY` in `env.sh`, which is only for the unused main-vm path) |
| `replay_buffer_size` | `mali_ba.ini` line ~176, currently `25000` |
| Last known full training command | see §3 below |
| Recommended `--num_actors` per worker | 15 (n2-standard-16, leaves 1 core for OS/tailscaled) |

## 3. Full restart checklist (desktop reboot / new session)

Use this any time you're starting a fresh session -- e.g. after rebooting the
desktop to Windows/WSL and back, or the next day.

**A. Stop what's currently running (if anything):**
```bash
# On the desktop, wherever train_mali_ba.py is running:
# Ctrl-C it, or:
pkill -f train_mali_ba.py

# Every worker's remote_actors.py needs to reconnect fresh once the trainer
# restarts (it's a new queue server process/socket, not the same one) -- it's
# NOT enough to just leave the old ones running. Stop them now, or just fold
# this into step E's restart_actors.sh call, which kills+restarts in one go.
cd gcp && ./stop_actors.sh
```

**B. Reboot / do whatever you need to on the desktop.** Tailscale
(`tailscaled`) is a systemd service and starts automatically on boot, and
since this machine is already an authenticated node, it should reconnect on
its own with no browser prompt. After rebooting back into Linux, verify:
```bash
source /home/robp/google-cloud-sdk/path.bash.inc   # gcloud isn't on PATH in a fresh shell until you source this, or open a new terminal (added to ~/.bashrc)
tailscale status
# Should show 100.92.31.37 robp-system-ubuntu ... with no "logged out" state.
# If it's somehow NOT connected, this reconnects using the existing identity
# (no new browser auth needed unless you revoked the key):
sudo tailscale up
```

**C. Check the GCP workers are still up** (spot VMs can get preempted while
you were away -- see §5 for the preemption email alert that should have told
you if this happened):
```bash
gcloud compute instances list --project=mali-ba-training
# If a worker shows TERMINATED (STOP, not spot-DELETE, so the disk survived):
gcloud compute instances start mali-ba-worker-1 --project=mali-ba-training --zone=us-central1-b
gcloud compute instances start mali-ba-worker-20260716-2000 --project=mali-ba-training --zone=us-central1-b
```

**D. Restart the trainer on the desktop.** Last known full command (adjust
`replay_buffer_size` in `mali_ba.ini` first if you're changing it, e.g. WSL's
50000 vs the 25000 currently in the file -- it's not a CLI flag). **Also
check `--remote_actors` matches your actual total worker capacity** --
queues are sized off this value at startup (see §"Queue Sizing Reference"
below); too low and remote actors silently block/stall rather than error.
`--remote_actors <N> = (number of workers) x (actors per worker)` -- e.g. 4
workers x 15 actors/worker = 60, shown below (was 35, sized for only 2
workers):
```bash
cd /media/robp/UD/Projects/open_spiel
python /media/robp/UD/Projects/mali_ba/open_spiel/python/games/mali_ba/train_mali_ba.py \
  --config_file mali_ba.ini --num_actors 24 --num_episodes 3000 \
  --bootstrap_episodes 0 --max_simulations 500 --batch_size 128 \
  --learning_rate 0.0002 --distributed --remote_actors 60 --queue_port 50000 \
  --save_model_path mali_ba_agent_v101.weights.h5 \
  --load_model_path mali_ba_agent_v39.weights.h5 \
  --save_buffer_path /media/robp/UD/Projects/open_spiel/mali_ba_buffer.pkl.gz \
  --save_every 30 --skip_timeout_games --replay_game 5
```
(Bump `--save_model_path`/`--load_model_path` version numbers as appropriate
for a fresh run vs resuming.)

**E. Restart `remote_actors.py` on every worker.** One command restarts
actors on ALL currently running `mali-ba-worker-*` instances (see §9 for
what this is actually doing under the hood, and how to watch it live):
```bash
cd gcp && ./restart_actors.sh
```
Or target just one: `./restart_actors.sh 1` (suffix or full name both work).
This kills+recreates the `actors` tmux session on each worker, which runs
`~/run_actors.sh` -- a loop that keeps `remote_actors.py` running (with the
correct PYTHONPATH/actor count/`--actor_id_start` already baked in) even
across a crash or an SSH disconnect. You do NOT need to SSH in yourself for
routine restarts anymore; do that only if you want to watch it live (§9) or
something's actually broken.

If you ever do open a fresh SSH session and want a human-readable summary,
`sudo /etc/update-motd.d/99-mali-ba-worker` on a worker prints the current
banner (cores, recommended actor count, ready-to-paste command) -- or it
just shows automatically on login if the worker was created from the
instance template (see §6).

## 4. Common operations reference

```bash
# --- Status ---
gcloud compute instances list --project=mali-ba-training
gcloud compute instances describe <name> --zone=us-central1-b --project=mali-ba-training

# --- SSH into a worker ---
gcloud compute ssh <name> --zone=us-central1-b --project=mali-ba-training

# --- Watch a worker's log in real time without a full interactive session ---
cd gcp && ./tail_worker.sh 1                    # -> tails ~/remote_actors.log on mali-ba-worker-1
cd gcp && ./tail_worker.sh 20260716-2000         # suffix or full name both work
cd gcp && ./tail_worker.sh 1 /var/log/mali-ba-startup.log   # optional 2nd arg overrides the log path
# Equivalent by hand:
#   gcloud compute ssh mali-ba-worker-1 --zone=us-central1-b --project=mali-ba-training \
#     --command="tail -F ~/remote_actors.log"
# ~/run_actors.sh (see §9) always writes here now, so this has something to
# tail as long as a worker's actors tmux session has ever been started.

# --- Restart / stop remote_actors.py on workers (see §9 for what's actually
#     happening -- tmux session + auto-restart loop under the hood) ---
cd gcp && ./restart_actors.sh              # ALL running workers
cd gcp && ./restart_actors.sh 1            # just mali-ba-worker-1
cd gcp && ./stop_actors.sh                 # ALL running workers, no restart
cd gcp && ./stop_actors.sh 20260716-2000   # just one worker

# --- Stop/start a worker (STOP preserves the disk; spot VMs also
#     auto-STOP on preemption since instance-termination-action=STOP) ---
gcloud compute instances stop <name> --project=mali-ba-training --zone=us-central1-b
gcloud compute instances start <name> --project=mali-ba-training --zone=us-central1-b

# --- Create a new worker (scripted path -- auto-unique actor-id-block,
#     auto-generated timestamp name, automatic zone/machine-type retry on
#     capacity exhaustion -- see §8) ---
cd gcp && ./create_worker_vm.sh
# It prints the exact push_code.sh/ssh/worker_vm_setup.sh commands to run
# next, using whichever zone it actually landed on (may differ from
# env.sh's GCP_ZONE if that was exhausted and it fell back).

# --- Create a new worker (Console path -- zero scripting) ---
# Console: Compute Engine -> Instance Templates -> mali-ba-worker-template
#   -> Create VM -> pick a zone -> Create. Then open its "Open in browser
#   window" SSH button -- login banner has cores/actors/host/command ready.
#   No push_code.sh or worker_vm_setup.sh needed; setup runs automatically.

# --- Move the trainer to a different machine later ---
cd gcp && ./set_trainer_host.sh <new-tailscale-ip>

# --- Rotate the Tailscale auth key (they expire, 90 days default) ---
# 1. Generate a new REUSABLE key in the Tailscale admin console
# 2. Update TS_AUTHKEY in gcp/env.sh
# 3. cd gcp && ./set_tailscale_authkey.sh
```

## 5. Preemption monitoring

A GCP log-based alert policy (`Mali-Ba Worker Preempted`, project
`mali-ba-training`) emails `robert.charles.parker@gmail.com` (rate-limited to
1/5min) any time a `mali-ba-worker-*` spot VM is preempted. If you're getting
these often, that's the signal to switch a worker to on-demand (drop
`--provisioning-model=SPOT` in `create_worker_vm.sh`/`create_worker_template.sh`)
or try a different zone/machine type.

## 6. Golden image + instance template maintenance

The golden image (`mali-ba-worker-base`) has conda/TF/pyspiel fully built, so
a new worker is ready in ~20-30s instead of a 20-40 min from-scratch build.

**Rebuild the image** (`cd gcp && ./create_worker_image.sh`) when:
- You've changed mali_ba C++ source files (added/removed files -- routine
  edits to existing files don't need this, see below)
- Dependency versions changed (`requirements-cpu.txt`, `install.sh`'s pinned
  abseil-cpp version)
- You want a fresh security-patched base

You do **NOT** need to rebuild the image for routine mali_ba code edits --
`push_code.sh` + an incremental `make pyspiel` on an already-running worker
picks up the change in seconds (abseil-cpp/json/pybind11 stay built).

**After rebuilding the image**, re-run `./create_worker_template.sh` if
you've also edited `worker_startup_script.sh` (templates are immutable, so
this deletes and recreates it). You do NOT need to re-run it just because a
new image landed in the family -- `create_worker_vm.sh`/the template both
reference the family, which always resolves to the latest image.

**Known trap (hit and fixed 2026-07-16):** merely installing the `tailscale`
package starts `tailscaled`, which generates a persistent machine key
immediately -- independent of ever running `tailscale up`. If that gets
baked into the golden image, every VM cloned from it shares one Tailscale
identity and fights over it (new workers silently kick old ones off the
tailnet -- "duplicate node key", only one machine reachable at a time).
`create_worker_image.sh` now clears `/var/lib/tailscale/tailscaled.state`
right before imaging to prevent this. If you ever see a worker mysteriously
lose Tailscale connectivity right after a new worker was created, this is
almost certainly why -- the golden image is stale (built before this fix).
Fix: `systemctl stop tailscaled && rm -f /var/lib/tailscale/tailscaled.state
&& systemctl start tailscaled && tailscale up --authkey=<key>` on the
affected worker, and rebuild the image.

## 7. Everything else fixed this session (troubleshooting reference)

- **`apply_open_spiel_edits.sh`'s CMake edit was completely wrong.** Real
  mali_ba integration is NOT `add_subdirectory(mali_ba)` with an
  `open_spiel_game()` macro (that file, `open_spiel/games/mali_ba/CMakeLists.txt`,
  is dead/unused) -- it's a flat source list inserted into `GAME_SOURCES` in
  `open_spiel/games/CMakeLists.txt`, plus a separately-appended `mali_ba_test`
  executable block. Also `PYBIND_CMAKE` pointed at a nonexistent
  `python/pybind11/CMakeLists.txt` -- the real file is `python/CMakeLists.txt`.
- **`cmake ..` from `$OPENSPIEL/build` was wrong** (no `CMakeLists.txt` at
  repo root) -- needs `cmake "$OPENSPIEL/open_spiel"` (source is one level
  down).
- **Setup scripts never ran OpenSpiel's own `install.sh`** (fetches vendored
  abseil-cpp/json/pybind11 deps). Added it, with all optional deps
  (hanabi/ACPC/xinxin/roshambo/libnop/libtorch/ortools) forced OFF to match
  `CMakeLists.txt`'s own defaults and avoid multi-GB pointless downloads.
- **Anaconda now requires `conda tos accept`** for `pkgs/main`/`pkgs/r`
  before `conda create` works non-interactively -- didn't exist when the
  scripts were originally written.
- **`PYTHONPATH` was missing the OpenSpiel repo root** -- see §
  "Troubleshooting" below (the big one, caused silent tracebackless
  actor-crash loops).
- **`mali_ba/__init__.py` unconditionally imports its `ui` submodule**
  (pygame board visualizer), even for headless actor processes. Added
  `pygame==2.6.1` to `requirements-cpu.txt`/`requirements-gpu.txt`.
- **T4 GPU and even CPU (`c2-standard-16`, then `n2-standard-16`) spot
  capacity was exhausted repeatedly** across many zones on 2026-07-16 --
  genuine stockouts, not quota. `create_worker_vm.sh`/`create_worker_image.sh`
  now retry across zones and machine types automatically -- see §8.
- **`.gitignore`'s `gcp/env.sh` pattern was anchored to repo root** but the
  real file is 4 directories deeper -- fixed to `**/gcp/env.sh`. Worth
  double-checking `git status` never shows `env.sh` as untracked before any
  `git add -A`.
- **Default CPU quota (`CPUS_ALL_REGIONS`) is 32** on a new project --
  exactly two `n2-standard-16` workers. Increased to 96 via a Console quota
  request (self-service override API caps out at the current limit, can't be
  raised past 32 without an actual reviewed request).

## 8. Capacity exhaustion: what to try, and automatic retry

GCP doesn't expose available capacity ahead of time -- a
`ZONE_RESOURCE_POOL_EXHAUSTED` error is only discoverable by actually trying
to create the VM. There's no way to know in advance which zone/machine type
will succeed.

**Scripted path (`create_worker_vm.sh`, `create_worker_image.sh`):** both now
retry automatically via `gcp/lib_create_with_retry.sh` -- on capacity
exhaustion they try the requested machine type across a list of fallback
zones, then (if still exhausted everywhere) fall back to `e2-standard-16`
(GCP's most elastic general-purpose family) across the same zones. A
different kind of error (bad quota, permissions, a typo'd flag) is surfaced
immediately rather than masked by pointless retries. Both scripts print which
zone/machine type it actually landed on (`CREATED_ZONE`/`CREATED_MACHINE_TYPE`)
-- use that value for any follow-up command (`push_code.sh`,
`gcloud compute ssh`, etc.), not the one you originally requested. Override
the fallback list/type via `env.sh`:
```bash
export WORKER_VM_FALLBACK_ZONES="us-central1-a us-central1-b us-central1-c us-central1-f us-west1-b us-east1-c"
export WORKER_VM_FALLBACK_MACHINE_TYPE="e2-standard-16"
```

**Console path** (creating from `mali-ba-worker-template`): no automatic
retry is possible here -- nothing scripted runs inside your browser clicks.
You'll see an error dialog saying the zone lacks capacity for that machine
type, with zone and machine type both editable fields in the creation form
(chosen at VM-creation time, not baked into the template). Just change the
zone dropdown and click Create again.

**Empirical order to try, from the night this was written** (capacity
fluctuates, so treat this as a starting point, not gospel):
1. `n2-standard-16` in `us-central1-a/b/c/f` -- all exhausted (spot AND
   on-demand)
2. `n2-standard-16` in `us-west1-b`, `us-east1-c` -- also exhausted (spot,
   for the original T4 main-vm attempt; not re-tested for plain CPU)
3. `n2-standard-8` in `us-central1-b` -- also exhausted
4. `e2-standard-8` in `us-central1-b` -- **worked**

`e2` is GCP's most elastic general-purpose family (doesn't require the
dedicated NUMA-node characteristics `n2`/`c2` do), so it's the best first
fallback to reach for if `n2` is tight, rather than continuing to zone-hop
with the same machine family.

## 9. Running actors overnight/unattended: tmux + auto-restart

**The problem this solves:** if you just SSH into a worker and run
`remote_actors.py` directly in that terminal, two things can kill it without
you knowing: closing the terminal / your SSH connection dropping (the
process dies with the session), or the process itself crashing for any
reason (network blip talking to the trainer, an unhandled exception, etc.).
For a run you're actively watching that's a non-issue -- you'd notice and
restart it. Overnight, unattended, it just silently stops contributing and
you find out in the morning.

**The fix is two independent layers**, both already wired up automatically
by `worker_vm_setup.sh` / `worker_startup_script.sh` on every worker:

1. **`~/run_actors.sh`** (auto-generated on every worker, correct
   PYTHONPATH/actor-count/`--actor_id_start`/trainer-host already baked in)
   is a `while true` loop around `remote_actors.py` -- if the Python process
   ever exits for any reason, the loop waits 10s and starts it again. This
   handles *crashes*.
2. **tmux** is a terminal multiplexer -- it runs a shell session that keeps
   existing on the VM independent of whether you're connected to it. You
   start a tmux session, run something inside it, *detach* (the session
   keeps running in the background), and can *reattach* later from a new SSH
   connection and find it exactly as you left it. This handles *SSH/terminal
   disconnects*.

Together: `run_actors.sh` running inside a tmux session survives both a
crash AND a dropped connection -- which is what an overnight run needs.

### tmux quick reference (if you've never used it)

```bash
# Start a new named session (do this once, when first starting actors):
tmux new -s actors

# ... now you're "inside" tmux. Run whatever you want here, e.g.:
~/run_actors.sh
# You'll see it running live, exactly like a normal terminal.

# Detach WITHOUT stopping it -- press Ctrl-b, release, then press d.
# (Ctrl-b is tmux's "prefix" key -- almost every tmux command starts with it.)
# You're back at your normal shell prompt. run_actors.sh keeps running.

# Later (even after closing your laptop, reconnecting, etc.), reattach:
gcloud compute ssh <worker-name> --zone=<zone> --project=mali-ba-training
tmux attach -t actors
# You're back inside, seeing the same live output, as if you never left.

# See what sessions exist (useful if you forgot the name, or want to check
# something is actually running):
tmux list-sessions

# Actually stop it (not just detach): from inside the session, Ctrl-C stops
# the current remote_actors.py, but the while-loop will just restart it 10s
# later -- Ctrl-C alone will NOT stop it for good. To really kill it, either
# kill the whole session from OUTSIDE tmux:
tmux kill-session -t actors
# ...or use gcp/stop_actors.sh from the desktop instead of doing this by hand.
```

### In practice, you mostly don't need to touch tmux directly

`gcp/restart_actors.sh` and `gcp/stop_actors.sh` (see §4) do the
`tmux kill-session` / `tmux new-session -d` dance for you, from the desktop,
without needing to SSH in yourself. Reach for manual tmux commands only when
you actually want to watch a worker live (`tmux attach -t actors`) or
something's broken and you're debugging it directly.

---

## Overview

The architecture has three roles:

- **Trainer** — one process, owns the neural network, trains on replay buffer data,
  periodically broadcasts updated weights.
- **Learner** (main process) — dispatches jobs, collects results from actors,
  feeds the replay buffer, coordinates everything.
- **Actors** — many processes, each plays one game using MCTS + the current NN
  weights, returns the trajectory and result to the learner.

In the single-machine version, actors are `mp.Process` children of the main
process and communicate via local `mp.Queue` objects.

In the distributed version, those same queues are **exposed over TCP** using
Python's `multiprocessing.managers.BaseManager`. Remote machines connect as
clients, get proxies to the same queues, and run actor processes that push/pull
through the network transparently.

```
Desktop (training server)
┌──────────────────────────────────────────────┐
│  main()  ←→  job_queue / result_queue        │
│              ↕  (BaseManager TCP server)      │
│  local actor processes (mp.Process)           │
└──────────────────────────────────────────────┘
        ↑ TCP (port 50000)
Laptop / second machine
┌──────────────────────────────────────────────┐
│  remote_actors.py                             │
│    connects to BaseManager                    │
│    spawns actor processes (mp.Process)        │
│    each actor pulls jobs, pushes results      │
└──────────────────────────────────────────────┘
```

---

## Files to Copy

Copy these two files from `mali_ba/` into your new game's folder:

- `queue_server.py` — the BaseManager server/client. **No changes needed.**
- `remote_actors.py` — spawns remote actor processes. Needs **one small change**
  (see Step 4 below).

---

## Step-by-Step Implementation

### Step 1 — Size the queues for total actors

In your `main()`, replace any hardcoded queue sizes with calculations based on
the total expected actor count (local + remote).

```python
# Add a --remote_actors argument to your arg parser
parser.add_argument('--remote_actors', type=int, default=0,
    help='Number of remote actors connecting from other machines')

# In main():
total_actors = args.num_actors + args.remote_actors
job_queue   = mp.Queue(maxsize=total_actors * 3)
result_queue = mp.Queue(maxsize=total_actors)
```

Also update any dispatch throttle that limits how many jobs are queued at once:

```python
target_job_queue_size = total_actors * 2
```

If you forget this, remote actors will starve — the queue will always be full
of local actor jobs and remotes will block waiting for work.

---

### Step 2 — Add queue server args

```python
parser.add_argument('--distributed', action='store_true',
    help='Start a queue server for remote actors')
parser.add_argument('--queue_port', type=int, default=50000)
parser.add_argument('--authkey', type=str, default='mytraining2024',
    help='Shared secret — must match on all machines')
```

---

### Step 3 — Start the queue server thread

Add this block in `main()` after the queues are created, before the main loop:

```python
if args.distributed:
    import threading
    from queue_server import start_server

    shared_config = {
        'game_name':           args.game_name,
        'initial_game_params': initial_game_params,
        'uct_c':               args.uct_c,
        'max_simulations':     args.max_simulations,
        'games_per_actor':     args.games_per_actor,
        # Add any other fields your actor_process() needs from args
    }
    server_thread = threading.Thread(
        target=start_server,
        args=(job_queue, result_queue, shared_config),
        kwargs={'port': args.queue_port, 'authkey': args.authkey.encode()},
        daemon=True
    )
    server_thread.start()
```

The server runs as a daemon thread — it dies automatically when training ends.
`queue_server.py` requires no modification.

---

### Step 4 — Adapt remote_actors.py for your game

Open `remote_actors.py` and find `_actor_worker()` and the config reconstruction
block. These are the only game-specific parts:

**`_actor_worker()`** — change the import to your game's actor function:

```python
def _actor_worker(actor_id, initial_game_params, actor_args, job_queue, result_queue, games_per_actor):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from your_train_script import actor_process   # ← change this import
    actor_process(actor_id, initial_game_params, actor_args, job_queue, result_queue, games_per_actor)
```

**Config reconstruction** — rebuild the `actor_args` namespace to match whatever
fields your `actor_process()` reads from `args`:

```python
actor_args = types.SimpleNamespace(
    game_name=config['game_name'],
    uct_c=config['uct_c'],
    max_simulations=config['max_simulations'],
    games_per_actor=config['games_per_actor'],
    # Add any other fields your actor_process() reads from args
)
```

Make sure the keys in `shared_config` (Step 3) match what you unpack here.

---

### Step 5 — Pass FINISHED log messages through the result queue

Actors run as subprocesses — their `print()`/`log()` output goes to their own
stdout, not the training server's log. To get FINISHED lines into the central
log, pass the message as part of the result tuple.

**In `actor_process()`**, build the message string and include it in the put:

```python
finished_msg = (
    f"Actor {actor_id}, Game {episode_num}: FINISHED in {move_count} moves. "
    f"Winner: {winner_str}. Reason: '{reason}'. Returns: {returns}"
)
result_queue.put((trajectory, returns, finished_msg))
```

Add a default value before the `if state.is_terminal():` block so the variable
is always defined:

```python
finished_msg = f"Actor {actor_id}, Game {episode_num}: FINISHED (non-terminal exit)."
if state.is_terminal():
    ...
    finished_msg = f"Actor {actor_id} ... (full message)"
result_queue.put((trajectory, returns, finished_msg))
```

**In the learner loop**, unpack the third element and log it:

```python
trajectory, returns, finished_msg = result_queue.get(timeout=1.0)
log(LogLevel.INFO, finished_msg)   # appears in server log for ALL actors
```

---

### Step 6 — Run on the training server

```bash
python train_your_game.py \
  --num_actors 24 \
  --remote_actors 25 \
  --distributed \
  --queue_port 50000 \
  --authkey mytraining2024 \
  2>&1 | tee ~/train_run.log
```

Find your server's IP:

```bash
ip addr | grep 'inet 192'
```

---

### Step 7 — Run remote_actors.py on the other machine

Make sure the other machine has:
1. The same codebase (use `rsync -avL` — the `-L` flag follows symlinks)
2. The same Python environment (same TF/numpy versions)
3. The compiled `pyspiel.so` — either copy from server or rebuild:
   ```bash
   cd ~/Projects/open_spiel/build
   cmake .. -DCMAKE_CXX_FLAGS="-O2" -DCMAKE_C_FLAGS="-O2"
   make -j$(nproc) pyspiel
   ```

Then start remote actors:

```bash
cd ~/Projects/open_spiel/open_spiel/python/games/your_game
python remote_actors.py \
  --server_host 192.168.0.102 \
  --num_actors 20 \
  --actor_id_start 100000
```

Use `--actor_id_start 100000` for the laptop and `200000` for any additional
remote process on the desktop, so actor IDs in the log are unique per machine.

---

### Step 8 — Wait for bootstrap before starting remote actors

If your training script has a bootstrap phase (heuristic games before MCTS),
remote actors should not connect until bootstrap is complete — they only run
MCTS and would dilute the bootstrap signal.

Watch for the transition log line:

```bash
tail -f ~/train_run.log | grep -m1 'Bootstrap phase complete'
```

This command blocks until the line appears, then exits — at which point you
start the remote processes.

---

## Queue Sizing Reference

| Parameter | Formula | Notes |
|-----------|---------|-------|
| `job_queue maxsize` | `total_actors × 3` | Keeps all actors fed with work |
| `result_queue maxsize` | `total_actors` | One slot per actor |
| `target_job_queue_size` | `total_actors × 2` | Dispatch throttle in learner loop |
| `total_actors` | `num_actors + remote_actors` | Must include ALL remote actors |

If `remote_actors` is set too low, the queue will be undersized and remote
actors will block waiting to push results, appearing to stall.

---

## Troubleshooting

**Remote actors connect but produce no results in the server log**
- Check `--remote_actors` was set correctly on the server. If too low, the
  result_queue fills up and actors block silently.
- Check actor IDs — remote actor FINISHED lines only appear in the server log
  if you implemented Step 5.

**`ModuleNotFoundError: No module named 'open_spiel'` on the remote machine**
- `open_spiel.python.algorithms.mcts` (used by `actor_process`) is plain
  Python, not part of the compiled `pyspiel.so` -- it needs the OpenSpiel repo
  *root* on `PYTHONPATH` too, not just `build/python`. Easy to miss this on a
  fresh machine because a stray `pip install open-spiel` elsewhere can
  silently paper over the gap (see the landmine note in
  `HANDOFF-GCP-SETUP.md`) -- worked "by accident" on one machine, then failed
  immediately with silent, instant actor-respawn loops (no traceback at all)
  on a clean one:
  ```bash
  export PYTHONPATH=/home/robp/Projects/open_spiel/build/python:\
  /home/robp/Projects/open_spiel:\
  /home/robp/Projects/open_spiel/open_spiel/python/games:$PYTHONPATH
  ```

**`ModuleNotFoundError: No module named 'pygame'` on the remote machine**
- `mali_ba/__init__.py` unconditionally imports its `ui` submodule (board
  visualizer), which needs `pygame`, even though headless actor processes
  never touch it. `pip install pygame` (already added to
  `requirements-cpu.txt`/`requirements-gpu.txt`).

**`undefined symbol: __asan_option_detect_stack_use_after_return`**
- The `.so` was compiled with AddressSanitizer. Rebuild with:
  ```bash
  cmake .. -DCMAKE_CXX_FLAGS="-O2" -DCMAKE_C_FLAGS="-O2"
  make -j$(nproc) pyspiel
  ```

**Remote actors stall after finishing first batch**
- `remote_actors.py` respawns actors automatically when they finish their quota.
  If it stops, check the remote terminal for Python errors.
