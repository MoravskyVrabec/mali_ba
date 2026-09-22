# Mali-Ba GCP Distributed Training — Setup Handoff

**Prepared:** 2026-07-16, for execution after switching this machine to native Linux.

This picks up where `HANDOFF-20260630.html` left off: moving from a two-machine
(desktop + laptop) training setup to Google Cloud, so much larger parallel
self-play runs are possible. It assumes **no GCP account/project exists yet**
and walks through everything from scratch.

Everything referenced here (`gcp/*.sh`, `requirements-gpu.txt`,
`requirements-cpu.txt`) already exists in this repo as of this handoff. No
GitHub push is required for this first test — code is transferred directly to
each VM with `gcp/push_code.sh`.

---

## 0. Why native Linux, not WSL

Per `HANDOFF-20260630.html` §12 and confirmed again this session: training
processes run noticeably faster on native Linux than WSL, and WSL git has
caused commit complications before. Do all of the following from the native
Ubuntu install, not WSL.

---

## 1. Code changes already made this session (informational — already done)

- `train_mali_ba.py`: added `--bind_host` (default `0.0.0.0`), now threaded
  into the `start_server(...)` call. Previously the queue server always tried
  to bind the hardcoded LAN IP `192.168.0.102` (from `queue_server.py`'s
  `start_server` default) because `--distributed` never passed `host=` at all
  — this would have silently failed to accept connections on any GCP VM.
- `.gitignore`: added `*.weights.h5`, `*.pkl.gz`, `*.pkl.gz.old`,
  `*.mali_ba_replay`, `build/`, `gcp/env.sh`.
- New: `requirements-gpu.txt` / `requirements-cpu.txt` (captured from the
  working WSL `mali_ba` conda env: Python 3.11, TensorFlow 2.21.0, Keras
  3.14.0, NumPy 2.4.4, absl-py 2.2.2).
- New: `gcp/` directory with `apply_open_spiel_edits.sh`, `main_vm_setup.sh`,
  `worker_vm_setup.sh`, `create_main_vm.sh`, `create_worker_vm.sh`,
  `push_code.sh`, `env.sh.example`.

**Landmine to remember:** the current WSL env has `open-spiel==1.5` installed
via pip — a real PyPI package that shadows the custom-built `pyspiel.so`. Never
`pip install open_spiel`/`open-spiel` on the new VMs. Every setup script here
sanity-checks `pyspiel.__file__` to catch this.

---

## 2. GCP account, project, and CLI (one-time)

```bash
# Install gcloud CLI on native Ubuntu
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init          # log in, pick/create project
```

Create a dedicated project (or use one `gcloud init` created for you):

```bash
gcloud projects create mali-ba-training --name="Mali-Ba Training"
gcloud config set project mali-ba-training
```

Link a billing account to it in the [console](https://console.cloud.google.com/billing)
— this can't be done from the CLI alone the first time.

Enable the Compute Engine API:

```bash
gcloud services enable compute.googleapis.com
```

### Request GPU quota now — this is the long pole

New projects default to **0 GPU quota**. Request a `NVIDIA_T4_GPUS` quota
increase (e.g. to 1 or 2) in `us-central1` from the
[IAM & Admin → Quotas](https://console.cloud.google.com/iam-admin/quotas) page.
Approval can take anywhere from minutes to ~48 hours — request this **first**,
then do everything else in this doc while you wait.

### Verify default networking

The `default` VPC (auto-mode) normally ships with a `default-allow-internal`
firewall rule permitting all traffic between instances on RFC1918 ranges —
this is what lets the worker VM reach the main-vm's queue server on port 50000
with zero extra config. Confirm it exists:

```bash
gcloud compute firewall-rules list --filter="name~default-allow-internal"
```

If it's missing (e.g. a custom VPC, or the rule was deleted), create one:

```bash
gcloud compute firewall-rules create allow-internal-mali-ba \
  --network=default --allow=tcp,udp,icmp --source-ranges=10.128.0.0/9
```

---

## 3. Configure the deployment scripts

```bash
cd ~/mali_ba/open_spiel/python/games/mali_ba/gcp   # or wherever the repo lands
cp env.sh.example env.sh
$EDITOR env.sh   # set GCP_PROJECT_ID, and a real AUTHKEY (shared secret)
```

`env.sh` is gitignored — it holds your project ID and the shared queue-server
secret, not meant to be committed.

---

## 4. Provision the main-vm (GPU, on-demand)

```bash
./create_main_vm.sh
# wait for RUNNING:
gcloud compute instances list --filter="name=mali-ba-main"
```

Push the code and run setup:

```bash
./push_code.sh mali-ba-main <path-to-your-local-mali_ba-repo>
gcloud compute ssh mali-ba-main --zone=us-central1-a
# --- now inside the VM ---
cd ~/mali_ba/open_spiel/python/games/mali_ba
./gcp/main_vm_setup.sh
```

`main_vm_setup.sh` installs the NVIDIA driver, Miniconda, the `mali_ba` conda
env, clones vanilla OpenSpiel, applies the 3 `OPEN_SPIEL_EDITS.txt` edits,
symlinks in the mali_ba code, and builds `pyspiel`. It ends with a sanity
check confirming `pyspiel` resolves to the custom build, not a pip package.

---

## 5. Provision the first worker-vm (spot CPU)

```bash
cd ~/mali_ba/open_spiel/python/games/mali_ba/gcp   # back on your local machine
./create_worker_vm.sh 1
./push_code.sh mali-ba-worker-1 <path-to-your-local-mali_ba-repo>
gcloud compute ssh mali-ba-worker-1 --zone=us-central1-a
# --- now inside the VM ---
cd ~/mali_ba/open_spiel/python/games/mali_ba
./gcp/worker_vm_setup.sh
```

---

## 6. Start training

Find the main-vm's internal IP:

```bash
gcloud compute instances describe mali-ba-main --zone=us-central1-a \
  --format='get(networkInterfaces[0].networkIP)'
```

On **main-vm** (inside `tmux` so it survives SSH disconnect):

```bash
tmux new -s train
conda activate mali_ba
export PYTHONPATH=~/open_spiel/build/python:~/open_spiel:~/open_spiel/open_spiel/python/games
cd ~/mali_ba/open_spiel/python/games/mali_ba
python train_mali_ba.py \
  --distributed --bind_host 0.0.0.0 --queue_port 50000 --authkey <same as env.sh> \
  --remote_actors 16 \
  --config_file "$(pwd)/mali_ba.ini" \
  --num_actors 4 \
  2>&1 | tee ~/train_run_gcp01.log
```

(`--num_actors 4` runs a few local actors on the main-vm itself alongside the
GPU trainer; adjust down if you'd rather dedicate the GPU VM's CPU fully to
training overhead. `--remote_actors 16` should roughly match the worker's
vCPU-derived actor count below, so queues are sized correctly — see
`DISTRIBUTED_TRAINING.md`'s queue-sizing reference.)

On **worker-vm-1**:

```bash
tmux new -s actors
conda activate mali_ba
export PYTHONPATH=~/open_spiel/build/python:~/open_spiel:~/open_spiel/open_spiel/python/games
cd ~/mali_ba/open_spiel/python/games/mali_ba
python remote_actors.py \
  --server_host <main-vm internal IP> --server_port 50000 --authkey <same as env.sh> \
  --num_actors 16 --actor_id_start 100000 --cpu_only
```

---

## 7. Verification checklist

- [ ] Main-vm log shows `Distributed queue server started on 0.0.0.0:50000.`
- [ ] Worker log shows `[Remote] Connected. Config received:` with the correct
      game/sim settings.
- [ ] `FINISHED` lines tagged with actor IDs ≥ 100000 appear in the main-vm log
      (proves remote results are flowing back, not just local actors working).
- [ ] Replay buffer file grows; a checkpoint save happens on schedule
      (`--save_every`).
- [ ] **Spot preemption test:** temporarily restart training with
      `--job_timeout_hours 0.05` (~3 min) for this test only, then manually
      stop the worker VM mid-run:
      `gcloud compute instances stop mali-ba-worker-1 --zone=us-central1-a`.
      Confirm the main-vm log emits a `jobs_timed_out` warning and keeps
      dispatching rather than stalling. Restart the worker afterward and set
      `--job_timeout_hours` back to a real value (default 3.0) for real runs.

---

## 8. After this works — scaling up and deferred items

- Increase worker count/size once the single-worker test is clean (just run
  `create_worker_vm.sh 2`, `3`, ... and `push_code.sh` + `worker_vm_setup.sh`
  each; `remote_actors.py --actor_id_start 200000` etc. per worker to keep IDs
  unique in the log).
- Revisit the GitHub push that was deferred from `HANDOFF-20260630.html` —
  now that `.gitignore` excludes large binaries, committing is more
  reasonable. Not required for GCP training to work, since `push_code.sh`
  bypasses git entirely for now.
- Consider `rare_goods_bonus` C++ rebuild status (per `HANDOFF-20260630.html`
  §10) — rebuild on the main-vm the same way as any other C++ change:
  `cd ~/open_spiel/build && make -j$(nproc) pyspiel`, then re-run
  `worker_vm_setup.sh`'s build step (or just `make pyspiel`) on each worker too,
  since they all need the same compiled `.so`.

---

## 9. Hetzner Cloud — an alternative to GCP spot workers

Added 2026-07-19/25, after GCP's GPU main-vm plan was abandoned (quota denied)
and the desktop + laptop became the trainer, with GCP spot CPU workers as
remote actors. A same-workload throughput comparison found Hetzner
**~2x faster** than GCP for this specific actor workload, at comparable or
lower cost, with **zero preemption risk** — worth using instead of or
alongside GCP spot workers.

### Why Hetzner, concretely

A controlled comparison (time to reach move 230 in-game, same MCTS tier
config on both sides) between a GCP `n2-custom-16-24576` spot worker and a
Hetzner CPX62:

| Machine | Avg time to move 230 |
|---|---|
| GCP `n2-custom-16-24576` (spot) | 122.4 min |
| Hetzner CPX62 | 66.7 min (**45% faster**) |

Best-guess explanation (not fully confirmed): GCP's N2 "vCPU" is one SMT
hyperthread (so 16 vCPU ≈ 8 real cores), while Hetzner's CPX line may have a
more favorable oversubscription model, plus possible CPU-generation
differences (Hetzner's cloud line generally runs AMD EPYC; GCP N2 runs Intel
Xeon). For this CPU-bound MCTS workload, the practical effect is what
matters regardless of exact cause.

Pricing (see `costgoat.com/pricing/hetzner` for current numbers — Hetzner
raised prices June 2026, so check before relying on this): **CPX62** (16
vCPU shared / 32GB RAM) is roughly **$0.23/hr**, in the same ballpark as GCP
N2 spot pricing but with **no preemption** — every dollar of Hetzner compute
survives to completion, unlike spot, where an unlucky preemption on a
2-hour game destroys 100% of the compute already spent on every in-flight
game on that worker (games have no mid-game checkpoint).

### Prerequisites (one-time)

1. **Hetzner Cloud account** at [hetzner.com/cloud](https://www.hetzner.com/cloud),
   with a Project created (e.g. "mali-ba-training").
2. **API token**: Security → API Tokens → Generate API Token, Read & Write
   permission. Hetzner only shows it once at creation — save it immediately.
3. **SSH key**, generated locally and registered with the Hetzner project
   (Security → SSH Keys, paste the public key):
   ```bash
   ssh-keygen -t ed25519 -C "hetzner-mali-ba" -f ~/.ssh/hetzner_mali_ba
   # empty passphrase is fine for scripted/automated access
   ```
4. **`hcloud` CLI**, installed as a user-local binary (no `sudo` needed —
   useful if passwordless sudo isn't set up):
   ```bash
   mkdir -p ~/.local/bin
   LATEST=$(curl -s https://api.github.com/repos/hetznercloud/cli/releases/latest | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
   curl -sL "https://github.com/hetznercloud/cli/releases/download/${LATEST}/hcloud-linux-amd64.tar.gz" -o /tmp/hcloud.tar.gz
   tar -xzf /tmp/hcloud.tar.gz -C ~/.local/bin hcloud
   chmod +x ~/.local/bin/hcloud
   export PATH="$HOME/.local/bin:$PATH"   # add to shell profile
   ```

### Local config file (mirrors `gcp/env.sh`)

Create `hetzner/env.sh` next to the `gcp/` directory (gitignored — see
`.gitignore`'s `**/hetzner/env.sh` pattern):

```bash
export HCLOUD_TOKEN="<your API token>"
export HETZNER_SSH_KEY_PATH="$HOME/.ssh/hetzner_mali_ba"
export HETZNER_SSH_KEY_NAME="hetzner-mali-ba"
```

### Creating a worker

No `create_worker_vm.sh`/`push_code.sh`/golden-image equivalent exists yet
for Hetzner — everything below is done with plain `hcloud`/`scp`/`ssh`, one
command at a time. **CPX62 is only available in `fsn1`, `nbg1`, and `hel1`**
(not the US or Singapore locations) — try `fsn1` first, fall back to the
others if capacity is tight (see the capacity-exhaustion note below).

```bash
source hetzner/env.sh
export PATH="$HOME/.local/bin:$PATH"

# 1. Create the server
hcloud server create \
  --name mali-ba-hetzner-1 \
  --type cpx62 \
  --image ubuntu-22.04 \
  --location fsn1 \
  --ssh-key "$HETZNER_SSH_KEY_NAME"
# prints the new server's IPv4 -- note it down

# 2. Install + authenticate Tailscale (get TS_AUTHKEY from gcp/env.sh --
#    same reusable key works for both GCP and Hetzner workers)
ssh -o StrictHostKeyChecking=accept-new -i "$HETZNER_SSH_KEY_PATH" root@<IP> \
  "curl -fsSL https://tailscale.com/install.sh | sh && tailscale up --authkey='<TS_AUTHKEY>'"

# 3. Push code (plain tar/scp -- no git push needed, same exclusions as push_code.sh)
TARBALL=$(mktemp /tmp/mali_ba_repo.XXXXXX.tar.gz)
tar -czf "$TARBALL" -C /path/to/Projects --exclude='*.weights.h5' --exclude='*.pkl.gz' \
  --exclude='*.pkl.gz.old' --exclude='*.mali_ba_replay' --exclude='build' --exclude='.git' \
  --exclude='__pycache__' mali_ba
scp -i "$HETZNER_SSH_KEY_PATH" "$TARBALL" root@<IP>:~/mali_ba.tar.gz
ssh -i "$HETZNER_SSH_KEY_PATH" root@<IP> \
  "rm -rf ~/mali_ba && mkdir -p ~/mali_ba && tar -xzf ~/mali_ba.tar.gz -C ~/mali_ba --strip-components=1 && rm ~/mali_ba.tar.gz"

# 4. Run setup (builds pyspiel from scratch -- no golden image for Hetzner yet,
#    so this takes the full 20-40 min every time)
ssh -i "$HETZNER_SSH_KEY_PATH" root@<IP> \
  "cd ~/mali_ba/open_spiel/python/games/mali_ba && ./gcp/worker_vm_setup.sh '' '' <actor_id_start> <trainer_tailscale_ip>"
```

`worker_vm_setup.sh` (in the `gcp/` dir, reused as-is — nothing Hetzner-
specific needed there) works unmodified on Hetzner **as long as both the
3rd arg (`actor_id_start`) and 4th arg (`trainer_host`) are passed
explicitly**. The script's `TRAINER_HOST`/`actor_id_start`-from-GCP-metadata
lookups now fail gracefully with a clear error (rather than hanging/aborting
under `set -e`) if metadata isn't reachable and no override was given — this
was a real portability bug found and fixed this session; it used to fail
silently mid-script on non-GCP hosts, right before writing `run_actors.sh`,
with no useful error message.

### Starting actors

Same pattern as GCP workers — `run_actors.sh` gets written by
`worker_vm_setup.sh`, then start it in `tmux`:

```bash
ssh -i "$HETZNER_SSH_KEY_PATH" root@<IP> "command -v tmux || apt-get install -y tmux"
ssh -i "$HETZNER_SSH_KEY_PATH" root@<IP> "tmux new-session -d -s actors '~/run_actors.sh'"
```

### CPU thread-limiting fix (important — large measured effect)

`remote_actors.py`/`train_mali_ba.py`'s actor processes don't set any
thread-count environment variables, so each actor's own TensorFlow/BLAS
stack spawns its own internal thread pool — measured at **~50 OS threads
per actor process**, i.e. ~750 threads competing for 16 cores on a CPX62.
Adding these three env vars before launch cut that to **2 threads/actor**
and produced a **measured 28.6%** wall-clock speedup (Hetzner) / **23.3%**
(desktop local actors) to reach the same in-game checkpoint:

```bash
export OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
```

This is now fixed **at the code level**, not just as a manual env-var
workaround — `train_mali_ba.py`'s `actor_process()` sets these via
`os.environ.setdefault(...)` right before its own `import tensorflow`,
before the trainer's own (separate, un-throttled) TF import happens. Since
`remote_actors.py` imports and reuses this exact function, **every actor —
local, laptop, GCP, or Hetzner — picks this up automatically with no manual
env vars needed**, and the trainer's own gradient-step computation is
unaffected (stays multi-threaded, since it's a separate `mp.Process` with
its own later TF import). If setting up a worker manually before this code
fix was pulled, the env-var version above still works identically (the code
version uses `setdefault`, so an explicit env var takes precedence).

### Billing — stopping a server does NOT save money

**Different from GCP.** Powering off a Hetzner Cloud server does not stop
billing — you're charged the full rate for as long as the server object
exists, on or off, because the CPU/RAM/storage stay reserved for you either
way. The only way to actually stop being charged is to **delete** the
server:

```bash
hcloud server delete mali-ba-hetzner-1
```

(Source: [Why Hetzner charges for stopped servers](https://cloudtally.eu/blog/why-hetzner-charges-for-stopped-servers).)
If you want to pause overnight without losing the setup, taking a snapshot
first (`hcloud server create-image`) lets you recreate quickly later,
skipping the 20-40 min build — not yet used this session, but worth doing
if pause/resume becomes a regular pattern.

### Known issue: Hetzner-side capacity exhaustion (as of 2026-07-25, unresolved)

Server creation started failing with `hcloud: permission denied (forbidden,
<hex-id>)` — misleading wording, since `hcloud server list` (read) and
`hcloud server delete` (write) both still worked fine, and the error
persisted identically across CPX62/CCX23, across all three CPX62-capable
locations (fsn1/nbg1/hel1), and across multiple retries over ~48 hours.
Root cause confirmed via Hetzner's own console FAQ: *"there is currently a
limited availability of Cloud Servers. This affects both the creation of
Cloud Servers and scaling through the rescale feature."* — a Hetzner-side
capacity constraint, not an account, token, or config problem. No
workaround found; periodic retry or contacting Hetzner support are the only
options. Check `hcloud server create ...` (or the web console) periodically
— if it starts succeeding again, resume from "Creating a worker" above.
