"""
strip_bootstrap.py  —  remove bootstrap experiences from a mali_ba replay buffer.

Usage:
    python strip_bootstrap.py <buffer.pkl.gz> [--out <output.pkl.gz>]

If --out is not given, writes to <original_stem>_no_bootstrap.pkl.gz in the
same directory as the input file.

Reads any buffer format the trainer can read (see buffer_format.py) and always
writes the current one, keeping the file's keep-every marker so the trainer does
not thin it again. A pre-2026-10-01 file's "natural" pool is split into the
timbuktu and timeout pools on the way through, exactly as the trainer does on load.

Required packages: standard library only.
"""

import argparse
import gzip
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buffer_format import KEEP_EVERY_KEY, POOL_KEYS, normalize_saved_buffer, saved_keep_every  # noqa: E402

LABELS = {
    'bootstrap_buffer':      'bootstrap',
    'mcts_timeout_buffer':   'timeout',
    'mcts_nearwin_buffer':   'nearwin',
    'mcts_raregoods_buffer': 'raregoods',
    'mcts_timbuktu_buffer':  'timbuktu',
}
assert set(LABELS) == set(POOL_KEYS)


def load_buffer(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def save_buffer(buf, path):
    # compresslevel=1: level 9 (gzip's default) takes ~4 minutes on a 200k-entry
    # buffer for a file only ~2x smaller. See buffer_compresslevel in mali_ba.ini.
    with gzip.open(path, 'wb', compresslevel=1) as f:
        pickle.dump(buf, f, protocol=pickle.HIGHEST_PROTOCOL)


def print_sizes(pools):
    for key in POOL_KEYS:
        print(f'  {LABELS[key]:10}: {len(pools[key]):>8,}')
    print(f'  {"total":10}: {sum(len(pools[k]) for k in POOL_KEYS):>8,}')


def main():
    parser = argparse.ArgumentParser(
        description='Strip bootstrap experiences from a mali_ba replay buffer.')
    parser.add_argument('buffer', help='Path to the .pkl.gz buffer file')
    parser.add_argument('--out', default=None,
                        help='Output path (default: <stem>_no_bootstrap.pkl.gz)')
    args = parser.parse_args()

    if not os.path.isfile(args.buffer):
        print(f'ERROR: file not found: {args.buffer}')
        sys.exit(1)

    if args.out:
        out_path = args.out
    else:
        stem = os.path.basename(args.buffer).replace('.pkl.gz', '')
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.buffer)),
                                f'{stem}_no_bootstrap.pkl.gz')

    print(f'Loading {args.buffer} ...')
    saved = load_buffer(args.buffer)
    pools, notes = normalize_saved_buffer(saved)
    keep_every = saved_keep_every(saved)
    del saved
    for n in notes:
        print(f'Converted: {n}')
    print(f'Positions kept per game: 1 in {keep_every}')
    print_sizes(pools)

    removed = len(pools['bootstrap_buffer'])
    pools['bootstrap_buffer'] = []
    pools[KEEP_EVERY_KEY] = keep_every

    print(f'\nBootstrap experiences removed: {removed:,}. Remaining:')
    print_sizes(pools)
    print(f'Saving to {out_path} ...')
    save_buffer(pools, out_path)
    print('Done.')


if __name__ == '__main__':
    main()
