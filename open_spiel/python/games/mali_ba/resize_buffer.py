"""
resize_buffer.py

Trims or expands a replay buffer to the specified pool sizes.
When a target is larger than the current content, all existing entries are kept
and the pool simply has room for more (it cannot create experiences from nothing).

Usage:
  python resize_buffer.py <input.pkl.gz> [output.pkl.gz]
                          [--natural N] [--nearwin N] [--bootstrap N] [--raregoods N]

  --natural    N  Target size for mcts_natural_buffer   (default: keep all)
  --nearwin    N  Target size for mcts_nearwin_buffer    (default: keep all)
  --bootstrap  N  Target size for bootstrap_buffer       (default: keep all)
  --raregoods  N  Target size for mcts_raregoods_buffer  (default: keep all)

  Pass 0 for any pool to empty it entirely.

If output path is omitted, writes <stem>_resized.pkl.gz in the same directory.
"""

import argparse
import gzip
import pathlib
import pickle
import sys
from collections import deque


def resize_pool(src, target):
    """Return (new_deque, action_label) for a pool."""
    n = len(src)
    if target < 0:
        # keep all
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
    parser = argparse.ArgumentParser(
        description="Trim or expand replay buffer pools.")
    parser.add_argument("input",  type=pathlib.Path, help="Input .pkl.gz buffer file")
    parser.add_argument("output", type=pathlib.Path, nargs="?",
                        help="Output path (default: <stem>_resized.pkl.gz)")
    parser.add_argument("--natural",   type=int, default=-1,
                        help="Target size for mcts_natural_buffer   (default: keep all; 0 = empty)")
    parser.add_argument("--nearwin",   type=int, default=-1,
                        help="Target size for mcts_nearwin_buffer    (default: keep all; 0 = empty)")
    parser.add_argument("--bootstrap", type=int, default=-1,
                        help="Target size for bootstrap_buffer       (default: keep all; 0 = empty)")
    parser.add_argument("--raregoods", type=int, default=-1,
                        help="Target size for mcts_raregoods_buffer  (default: keep all; 0 = empty)")
    args = parser.parse_args()

    src_path = args.input
    if args.output:
        dst_path = args.output
    else:
        stem = src_path.name.split(".")[0]
        dst_path = src_path.with_name(f"{stem}_resized.pkl.gz")

    print(f"Loading {src_path} ...")
    with gzip.open(src_path, "rb") as f:
        buf = pickle.load(f)

    natural_src   = list(buf.get("mcts_natural_buffer",   []))
    nearwin_src   = list(buf.get("mcts_nearwin_buffer",   []))
    bootstrap_src = list(buf.get("bootstrap_buffer",      []))
    raregoods_src = list(buf.get("mcts_raregoods_buffer", []))

    print(f"Source pool sizes:")
    print(f"  natural   : {len(natural_src):,}")
    print(f"  nearwin   : {len(nearwin_src):,}")
    print(f"  bootstrap : {len(bootstrap_src):,}")
    print(f"  raregoods : {len(raregoods_src):,}")
    print()

    new_natural,   nat_label  = resize_pool(natural_src,   args.natural)
    new_nearwin,   nw_label   = resize_pool(nearwin_src,   args.nearwin)
    new_bootstrap, boot_label = resize_pool(bootstrap_src, args.bootstrap)
    new_raregoods, rg_label   = resize_pool(raregoods_src, args.raregoods)

    print(f"Actions:")
    print(f"  natural   : {nat_label}")
    print(f"  nearwin   : {nw_label}")
    print(f"  bootstrap : {boot_label}")
    print(f"  raregoods : {rg_label}")
    print()

    print(f"Output pool sizes:")
    print(f"  natural   : {len(new_natural):,}")
    print(f"  nearwin   : {len(new_nearwin):,}")
    print(f"  bootstrap : {len(new_bootstrap):,}")
    print(f"  raregoods : {len(new_raregoods):,}")
    print()

    new_buf = {
        "bootstrap_buffer":      new_bootstrap,
        "mcts_natural_buffer":   new_natural,
        "mcts_nearwin_buffer":   new_nearwin,
        "mcts_raregoods_buffer": new_raregoods,
    }

    print(f"Saving to {dst_path} ...")
    with gzip.open(dst_path, "wb") as f:
        pickle.dump(new_buf, f)

    print("Done.")


if __name__ == "__main__":
    main()
