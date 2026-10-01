"""
resize_buffer.py

Trims or expands a replay buffer to the specified pool sizes.
When a target is larger than the current content, all existing entries are kept
and the pool simply has room for more (it cannot create experiences from nothing).

Reads any buffer format the trainer can read (see buffer_format.py) and always
writes the current one. A pre-2026-10-01 file has a single "natural" pool holding
Timbuktu wins and ordinary timeouts together; it is split into the timbuktu and
timeout pools on the way through, exactly as the trainer does on load.

Usage:
  python resize_buffer.py <input.pkl.gz> [output.pkl.gz]
                          [--timeout N] [--nearwin N] [--raregoods N]
                          [--timbuktu N] [--bootstrap N]

  --timeout    N  Target size for mcts_timeout_buffer    (ordinary timeouts)
  --nearwin    N  Target size for mcts_nearwin_buffer    (near-win timeouts)
  --raregoods  N  Target size for mcts_raregoods_buffer  (rare-goods wins)
  --timbuktu   N  Target size for mcts_timbuktu_buffer   (Timbuktu wins)
  --bootstrap  N  Target size for bootstrap_buffer
  --natural    N  Old name for --timeout, still accepted

  Default for every pool: keep all. Pass 0 to empty a pool entirely.
  Trimming keeps the most recent entries.

If output path is omitted, writes <stem>_resized.pkl.gz in the same directory.
"""

import argparse
import gzip
import os
import pathlib
import pickle
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buffer_format import POOL_KEYS, normalize_saved_buffer  # noqa: E402

# (pool key, command-line option, label)
POOLS = (
    ('mcts_timeout_buffer',   'timeout',   'timeout'),
    ('mcts_nearwin_buffer',   'nearwin',   'nearwin'),
    ('mcts_raregoods_buffer', 'raregoods', 'raregoods'),
    ('mcts_timbuktu_buffer',  'timbuktu',  'timbuktu'),
    ('bootstrap_buffer',      'bootstrap', 'bootstrap'),
)
assert {p[0] for p in POOLS} == set(POOL_KEYS)


def resize_pool(src, target):
    """Return (new_deque, action_label) for a pool."""
    n = len(src)
    if target < 0:
        new = deque(src)
        label = f"unchanged ({n:,})"
    elif target == 0:
        new = deque()
        label = f"cleared  ({n:,} → 0)"
    elif target >= n:
        new = deque(src)
        label = f"expand   ({n:,} → capacity {target:,})"
    else:
        new = deque(src[-target:])
        label = f"trim     ({n:,} → {target:,})"
    return new, label


def main():
    parser = argparse.ArgumentParser(description="Trim or expand replay buffer pools.")
    parser.add_argument("input",  type=pathlib.Path, help="Input .pkl.gz buffer file")
    parser.add_argument("output", type=pathlib.Path, nargs="?",
                        help="Output path (default: <stem>_resized.pkl.gz)")
    for _, opt, label in POOLS:
        parser.add_argument(f"--{opt}", type=int, default=-1,
                            help=f"Target size for the {label} pool (default: keep all; 0 = empty)")
    parser.add_argument("--natural", type=int, default=None,
                        help="Old name for --timeout (the timeout pool was called 'natural')")
    args = parser.parse_args()
    if args.natural is not None:
        if args.timeout != -1 and args.timeout != args.natural:
            parser.error("--natural and --timeout given with different values")
        args.timeout = args.natural

    src_path = args.input
    if args.output:
        dst_path = args.output
    else:
        stem = src_path.name.split(".")[0]
        dst_path = src_path.with_name(f"{stem}_resized.pkl.gz")

    print(f"Loading {src_path} ...")
    with gzip.open(src_path, "rb") as f:
        saved = pickle.load(f)
    pools, notes = normalize_saved_buffer(saved)
    del saved
    for n in notes:
        print(f"Converted: {n}")

    print("Source pool sizes:")
    for key, _, label in POOLS:
        print(f"  {label:9} : {len(pools[key]):,}")
    print()

    new_buf, actions = {}, []
    for key, opt, label in POOLS:
        new_buf[key], act = resize_pool(pools[key], getattr(args, opt))
        actions.append((label, act))

    print("Actions:")
    for label, act in actions:
        print(f"  {label:9} : {act}")
    print()
    print("Output pool sizes:")
    for key, _, label in POOLS:
        print(f"  {label:9} : {len(new_buf[key]):,}")
    print()

    print(f"Saving to {dst_path} ...")
    # compresslevel=1: level 9 (gzip's default) takes ~4 minutes on a 200k-entry
    # buffer for a file only ~2x smaller. See buffer_compresslevel in mali_ba.ini.
    with gzip.open(dst_path, "wb", compresslevel=1) as f:
        pickle.dump(new_buf, f)

    print("Done.")


if __name__ == "__main__":
    main()
