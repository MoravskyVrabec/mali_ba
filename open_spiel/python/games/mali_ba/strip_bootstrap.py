"""
strip_bootstrap.py  —  remove bootstrap experiences from a mali_ba replay buffer.

Usage:
    python strip_bootstrap.py <buffer.pkl.gz> [--out <output.pkl.gz>]

If --out is not given, writes to <original_stem>_no_bootstrap.pkl.gz in the
same directory as the input file.

Required packages: numpy  (standard library only otherwise)
"""

import argparse
import gzip
import os
import pickle
import sys


def load_buffer(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def save_buffer(buf, path):
    with gzip.open(path, 'wb') as f:
        pickle.dump(buf, f, protocol=pickle.HIGHEST_PROTOCOL)


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
    buf = load_buffer(args.buffer)

    before_bootstrap = len(buf['bootstrap_buffer'])
    before_natural   = len(buf['mcts_natural_buffer'])
    before_nearwin   = len(buf['mcts_nearwin_buffer'])
    before_total     = before_bootstrap + before_natural + before_nearwin

    print(f'  bootstrap   : {before_bootstrap:>8,}')
    print(f'  mcts natural: {before_natural:>8,}')
    print(f'  mcts near-win:{before_nearwin:>8,}')
    print(f'  total       : {before_total:>8,}')

    buf['bootstrap_buffer'].clear()

    after_total = len(buf['mcts_natural_buffer']) + len(buf['mcts_nearwin_buffer'])
    print(f'\nBootstrap experiences removed. Remaining: {after_total:,}')
    print(f'Saving to {out_path} ...')
    save_buffer(buf, out_path)
    print('Done.')


if __name__ == '__main__':
    main()
