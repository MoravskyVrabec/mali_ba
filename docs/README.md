# Mali-Ba documentation

The project's to-do list is [`../TODO.md`](../TODO.md) at the repo root.

## rules/
- [Mali-ba rules (Mancala-style game)-0.8.md](rules/Mali-ba%20rules%20(Mancala-style%20game)-0.8.md):
  the game's rules. The code follows them, except that the rare-goods end
  condition is "a rare good from five regions" (see `TODO.md`).

## guides/ (current how-to documents)
- [DIAGNOSTIC_TOOLS.md](guides/DIAGNOSTIC_TOOLS.md): what `analyze_log.py`,
  `ab_eval.py`, `policy_probe.py`, `search_sharpness.py` and
  `analyze_placements.py` measure, how they work and how to read them.
- [DISTRIBUTED_TRAINING.md](guides/DISTRIBUTED_TRAINING.md): running training
  across machines (queue server, remote actors).
- [GCP_SETUP.md](guides/GCP_SETUP.md): the Google Cloud worker setup (not in use
  since 2026-09-25).
- [DEVELOPMENT.md](guides/DEVELOPMENT.md): the UI development mode (working on the
  game UI without the C++ backend).

## design/ (proposals for review)
- [ENGINE_UI_SEPARATION.md](design/ENGINE_UI_SEPARATION.md): separating the game
  engine from the presentation layer, for a local GUI and a web front end against
  trained bots (2026-10-08).

## handoffs/ (dated session handoffs, oldest first)
Snapshots of where the project stood at the end of a working session: runs,
results, open problems and next steps. The newest is the best starting point;
older ones are history.
- HANDOFF-20260523.md, -20260624, -20260626, -20260630, -20260725 (.html)
- [HANDOFF-20261002.html](handoffs/HANDOFF-20261002.html): the A-D series, the
  memorization finding and the plan that led to D005-D007.

## notes/ (past design notes and write-ups)
Records of specific changes, investigations and plans, kept for context:
`dev_approach_2026-04-02.md`, `HANDOFF_bfs_fix.md`, `REVERT_TENSOR_SHAPE_CHANGE.md`,
`improve_rewards-20260719.md`, `ImplementCentralRulesSwitching.md`,
`RemoveBoardConfigRefactor.md`, `SwitchToINIFile.md`, `HowToTrain.md` (March 2026,
likely out of date), and `TODO-2026-09-30.md` (the previous to-do list).

## archive/
Old working files from March 2026 (code and output dumps, small scripts), kept
for reference only.

## Elsewhere, next to the code
- `open_spiel/python/games/mali_ba/README.md`: running training.
- `open_spiel/python/games/mali_ba/probe_data/README.md`: the probe's position
  sets, and which ini to score them with.
- `open_spiel/games/mali_ba/commands.txt`: build and run commands.
