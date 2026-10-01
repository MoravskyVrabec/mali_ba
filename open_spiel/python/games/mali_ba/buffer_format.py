"""Replay-buffer file format: pool names and conversion of older saves.

Kept free of TensorFlow and pyspiel imports so tools such as resize_buffer.py can
use it without loading either. The trainer (train_mali_ba.py) and resize_buffer.py
both read saved buffers through normalize_saved_buffer(), so they cannot disagree
about what an older file contains.

Pools (current format, 2026-10-01):
  bootstrap_buffer       heuristic bootstrap games
  mcts_timeout_buffer    ordinary timeouts (no near-win)
  mcts_nearwin_buffer    near-win timeouts
  mcts_raregoods_buffer  rare-goods wins
  mcts_timbuktu_buffer   Timbuktu wins

Older saves had one "natural" pool (mcts_natural_buffer) holding Timbuktu wins and
ordinary timeouts together -- the name dates from when it held only "natural" wins.
Each entry is (observation, policy_target, (player, value, value_vector)); a win
position's value_vector has a target near +1 for the winner (about +0.8 after
per-move time penalties), a timeout's tops out near +0.1. So old natural entries
are split on max(value_vector) > WIN_TARGET_THRESHOLD.
"""

POOL_KEYS = (
    'bootstrap_buffer',
    'mcts_timeout_buffer',
    'mcts_nearwin_buffer',
    'mcts_raregoods_buffer',
    'mcts_timbuktu_buffer',
)

# A win's target for the winner is ~0.8 or more; a timeout leader's is ~0.1.
WIN_TARGET_THRESHOLD = 0.5


def _is_win_entry(entry):
    """True if this experience comes from a game that someone won."""
    try:
        return max(entry[2][2]) > WIN_TARGET_THRESHOLD
    except (IndexError, TypeError, ValueError):
        return False


def normalize_saved_buffer(saved):
    """Return ({pool_key: list_of_entries} for every POOL_KEYS entry, notes).

    Accepts the current format and older ones:
      - natural-pool saves (mcts_natural_buffer): split into Timbuktu wins and
        ordinary timeouts by the win target;
      - single-pool saves (mcts_buffer): treated like a natural pool;
      - saves without a rare-goods pool: the natural pool is first split 50/50 to
        seed it, as the trainer has always done for such files.
    `notes` describes any conversion, for logging.
    """
    notes = []
    out = {k: list(saved.get(k) or []) for k in POOL_KEYS}

    if 'mcts_timeout_buffer' not in saved:
        old = list(saved.get('mcts_natural_buffer') or saved.get('mcts_buffer') or [])
        if 'mcts_raregoods_buffer' not in saved and old:
            half = len(old) // 2
            out['mcts_raregoods_buffer'] = old[half:]
            old = old[:half]
            notes.append(f"no rare-goods pool in file: split {half * 2} natural entries "
                         f"50/50 to seed it")
        wins = [e for e in old if _is_win_entry(e)]
        timeouts = [e for e in old if not _is_win_entry(e)]
        out['mcts_timbuktu_buffer'] = wins + out['mcts_timbuktu_buffer']
        out['mcts_timeout_buffer'] = timeouts
        if old:
            notes.append(f"old 'natural' pool ({len(old):,}) split into Timbuktu wins "
                         f"({len(wins):,}) and ordinary timeouts ({len(timeouts):,})")
    return out, notes
