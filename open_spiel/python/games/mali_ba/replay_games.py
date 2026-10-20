"""Replay the self-play games in a training log through the engine, and measure them.

A game is fully determined by what its log lines record: the chance outcome
(`SETUP layout=Setup_N`; the meeple layout and starting goods depend only on N), the
token placements on the same SETUP line (in serpentine order per seat), and every
main-play decision:
    Actor 52, Game 1234: SETUP layout=Setup_540 placement=random tokens P0=(x,y,z);.. P1=..
    Actor 52, Game 1234, Move 17: Player 1 plays 'MancalaDir_4'
    Actor 52, Game 1234: FINISHED in 342 moves. ... Reason: '...'
So each finished game can be rebuilt exactly and any board quantity measured at any
point. Every logged move is checked against the legal moves, and the player to move
against the log. On the first mismatch the script stops with a message naming the log,
game and move (usually the ini or the engine build does not match the run's rules);
--keep_going drops such games instead and counts them in the summary.

Only games the desktop played are replayable: remote (laptop) games log their result
but not their moves ("no moves/setup logged" in the summary).

Measured now (one record per game, see replay_game):
  - meeples on the board at the start, every 50 moves and at the end (meeples in a
    player's hand during a mancala move count as still on the board);
  - empty hexes at the same points;
  - meeples removed, by the move type that removed them;
  - legal options at the start of each turn.
To measure something else, add it to replay_game (the state is a full engine state
at every step) and to summarize.

Usage (from open_spiel/python/games/mali_ba, with the usual PYTHONPATH; single
process, ~17 games/s, safe to run at low priority beside training):
    nice -n 19 python replay_games.py /media/robp/UD/Projects/open_spiel/train_runD041.log \
        [--json records.json] [--max_games 500] [--config_file mali_ba.ini] [--keep_going]
"""

import argparse
import json
import os
import re
import statistics as st
import sys
import time
from collections import defaultdict

import pyspiel

RE_KEY = re.compile(r"Actor (\d+), Game (\d+)")
RE_PLAY = re.compile(r"Actor (\d+), Game (\d+), Move (\d+): Player (\d+) plays '([^']*)'")
RE_SETUP = re.compile(r"Actor (\d+), Game (\d+): SETUP layout=(Setup_\d+) "
                      r"(?:placement=\w+ )?tokens (.*)$")
RE_FINISHED = re.compile(r"Actor (\d+), Game (\d+): FINISHED in (\d+) moves.*?Reason: '([^']*)'")
TIMEOUT_REASON = "Max game length"
CURVE_EVERY = 50


class ReplayMismatch(Exception):
    """A logged game the engine cannot reproduce."""

    def __init__(self, kind, reason, detail=""):
        super().__init__(reason)
        self.kind, self.reason, self.detail = kind, reason, detail


def board_counts(state, num_hexes):
    """(meeples on the board or in hand, empty hexes)."""
    j = json.loads(pyspiel.mali_ba.downcast_state(state).serialize())
    hm = j["hexMeeples"]
    in_hand = len(j.get("midTurnState", {}).get("meeplesInHand", []))
    # serialize() leaves empty hexes out of hexMeeples
    return (sum(len(v) for v in hm.values()) + in_hand,
            num_hexes - sum(1 for v in hm.values() if v))


def apply_string(state, wanted, where):
    """Applies the legal action whose string is `wanted`; raises ReplayMismatch if none."""
    player = state.current_player()
    legal = [state.action_to_string(player, a) for a in state.legal_actions()]
    if wanted not in legal:
        shown = ", ".join(legal[:12]) + (f", ... ({len(legal)} in all)" if len(legal) > 12 else "")
        raise ReplayMismatch(f"{where.split()[0]} not legal", f"{where}: '{wanted}' is not legal",
                             f"player {player} to move; legal: {shown or 'none'}")
    state.apply_action(state.legal_actions()[legal.index(wanted)])


def replay_game(game, num_hexes, setup, tokens, moves):
    """Rebuilds one game and returns its record; raises ReplayMismatch if it can't."""
    place_phase = pyspiel.mali_ba.Phase.PLACE_TOKEN
    play_phase = getattr(pyspiel.mali_ba.Phase, "PLAY", None)
    state = game.new_initial_state()
    apply_string(state, setup, "setup")
    start_meeples, _ = board_counts(state, num_hexes)
    queues = {p: list(t) for p, t in tokens.items()}
    while (not state.is_terminal()
           and pyspiel.mali_ba.downcast_state(state).current_phase() == place_phase):
        p = state.current_player()
        if not queues.get(p):
            raise ReplayMismatch("token missing", "token placement: the SETUP line has no token left for "
                                 f"player {p}", f"tokens logged: {tokens}")
        apply_string(state, "PlaceToken_" + queues[p].pop(0), "token placement")

    removed_by = defaultdict(int)
    curve, turn_options = {}, []
    prev, _ = board_counts(state, num_hexes)
    for i, (player, move) in enumerate(moves):
        if state.is_terminal():
            raise ReplayMismatch("game over early", f"move {i}: the game is already over, but the log has "
                                 f"{len(moves) - i} more moves",
                                 f"end reason: {pyspiel.mali_ba.downcast_state(state).get_game_end_reason()}")
        if state.current_player() != player:
            raise ReplayMismatch("wrong player", f"move {i}: the log has player {player} to move, the engine "
                                 f"player {state.current_player()}", f"logged move: '{move}'")
        if (play_phase is not None
                and pyspiel.mali_ba.downcast_state(state).current_phase() == play_phase):
            turn_options.append((i, len(state.legal_actions())))
        apply_string(state, move, f"move {i}")
        now, empty = board_counts(state, num_hexes)
        if now != prev:
            removed_by[move.split("_")[0]] += prev - now
        prev = now
        if (i + 1) % CURVE_EVERY == 0:
            curve[i + 1] = (now, empty)

    end_meeples, end_empty = board_counts(state, num_hexes)
    return {"start_meeples": start_meeples, "end_meeples": end_meeples,
            "end_empty": end_empty, "moves": len(moves), "terminal": state.is_terminal(),
            "removed_by": dict(removed_by), "curve": curve,
            "turn_options": turn_options}


def replay_log(path, game, num_hexes, max_games, keep_going=False, progress_every=1000):
    """Streams a training log; replays each game as its FINISHED line arrives.

    Raises ReplayMismatch (with the log and game added) on the first game the engine
    cannot reproduce, unless keep_going, which drops and counts it instead."""
    setups, moves = {}, defaultdict(list)
    records, failures = [], defaultdict(int)
    t0 = time.time()
    with open(path, errors="replace") as f:
        for line in f:
            if " plays '" in line:
                m = RE_PLAY.search(line)
                if m:
                    moves[(m[1], m[2])].append((int(m[4]), m[5]))
                continue
            if "Starting token placement" in line:
                # A restarted actor can reuse an (actor, game) pair: start clean.
                m = RE_KEY.search(line)
                if m:
                    moves.pop((m[1], m[2]), None)
                    setups.pop((m[1], m[2]), None)
                continue
            if "SETUP layout=" in line:
                m = RE_SETUP.search(line)
                if m:
                    tokens = {}
                    for part in m[4].split():
                        seat, _, coords = part.partition("=")
                        tokens[int(seat[1:])] = [c for c in coords.split(";") if c]
                    setups[(m[1], m[2])] = (m[3], tokens)
                continue
            if "FINISHED in" in line:
                m = RE_FINISHED.search(line)
                if not m:
                    continue
                key = (m[1], m[2])
                game_moves, setup = moves.pop(key, []), setups.pop(key, None)
                if not game_moves or setup is None:
                    failures["no moves/setup logged (remote game)"] += 1
                    continue
                try:
                    rec = replay_game(game, num_hexes, setup[0], setup[1], game_moves)
                except (ReplayMismatch, pyspiel.SpielError) as e:
                    if not isinstance(e, ReplayMismatch):
                        e = ReplayMismatch("engine error", f"engine error: {e}")
                    if keep_going:
                        failures["not reproducible: " + e.kind] += 1
                        continue
                    e.where = (f"{os.path.basename(path)}, Actor {key[0]}, Game {key[1]} "
                               f"({setup[0]}, finished in {m[3]} moves, '{m[4]}')")
                    e.replayed = len(records)
                    raise e
                rec["reason"] = m[4]
                rec["log"] = os.path.basename(path)
                records.append(rec)
                if len(records) % progress_every == 0:
                    print(f"  {len(records)} replayed, {time.time() - t0:.0f}s", flush=True)
                if len(records) >= max_games:
                    break
    return records, dict(failures)


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(name, rows):
    if not rows:
        return
    start = rows[0]["start_meeples"]
    end = [r["end_meeples"] for r in rows]
    removed = [r["start_meeples"] - r["end_meeples"] for r in rows]
    empty = [r["end_empty"] for r in rows]
    length = [r["moves"] for r in rows]
    print(f"\n== {name}: {len(rows)} games, {start} meeples at the start")
    print(f"  game length (decisions)  median {st.median(length):.0f}  "
          f"p10 {_pct(length, .1)}  p90 {_pct(length, .9)}")
    print(f"  turns per player         median {st.median(len(r['turn_options']) for r in rows) / 3:.0f}"
          "  (approximate)")
    print(f"  meeples left at end      median {st.median(end):.0f}  min {min(end)}  "
          f"p10 {_pct(end, .1)}  p90 {_pct(end, .9)}")
    print(f"  meeples removed          median {st.median(removed):.0f}  max {max(removed)}  "
          f"({100 * st.median(removed) / start:.0f}% of the start)")
    print(f"  empty hexes at end       median {st.median(empty):.0f}  max {max(empty)}")
    cause = defaultdict(int)
    for r in rows:
        for k, v in r["removed_by"].items():
            cause[k] += v
    total = sum(v for v in cause.values() if v > 0) or 1
    print("  removed by move type     " + "  ".join(
        f"{k} {v / len(rows):.1f}/game ({100 * v / total:.0f}%)"
        for k, v in sorted(cause.items(), key=lambda kv: -kv[1]) if v > 0))
    curve = defaultdict(list)
    for r in rows:
        for k, v in r["curve"].items():
            curve[int(k)].append(v)
    print("  board by move (median meeples / empty hexes, games still running)")
    print("    " + "  ".join(f"{k}: {st.median(m for m, _ in v):.0f}/{st.median(e for _, e in v):.0f}"
                             f" (n={len(v)})" for k, v in sorted(curve.items()) if k % 100 == 0))
    fifths = defaultdict(list)
    for r in rows:
        for i, n in r["turn_options"]:
            fifths[min(4, int(5 * i / max(1, r["moves"])))].append(n)
    print("  legal options at the start of a turn, by fifth of the game")
    print("    " + "  ".join(f"{20 * b}-{20 * (b + 1)}%: {st.mean(v):.1f}"
                             for b, v in sorted(fifths.items())))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+", help="training logs (train_runDxxx.log)")
    ap.add_argument("--config_file", default=os.path.join(here, "mali_ba.ini"),
                    help="ini with the rules the run used (default: the current one)")
    ap.add_argument("--max_games", type=int, default=10**9, help="per log")
    ap.add_argument("--json", help="write every game's record to this file")
    ap.add_argument("--keep_going", action="store_true",
                    help="drop games the engine cannot reproduce instead of stopping")
    args = ap.parse_args()

    game = pyspiel.load_game("mali_ba", {"config_file": args.config_file})
    num_hexes = len(pyspiel.mali_ba.downcast_game(game).get_valid_hexes())
    records, failures = [], defaultdict(int)
    for path in args.logs:
        print(f"Replaying {path}", flush=True)
        try:
            recs, fails = replay_log(path, game, num_hexes, args.max_games, args.keep_going)
        except ReplayMismatch as e:
            print(f"\nSTOPPED: a logged game could not be replayed.\n"
                  f"  game:    {e.where}\n"
                  f"  problem: {e.reason}\n"
                  + (f"  detail:  {e.detail}\n" if e.detail else "")
                  + f"  ({e.replayed} games of this log replayed before it)\n"
                  "Likely causes: the ini or the engine build differs from the run's rules (pass\n"
                  "the run's ini with --config_file, or check the build date of pyspiel.so), or the\n"
                  "log line was garbled by interleaved output. --keep_going drops such games and\n"
                  "counts them instead.", file=sys.stderr)
            sys.exit(1)
        records += recs
        for k, v in fails.items():
            failures[k] += v
    print(f"\n{len(records)} games replayed; not replayed: {dict(failures)}")
    summarize("all games", records)
    summarize("natural end", [r for r in records if TIMEOUT_REASON not in r["reason"]])
    summarize("timeout", [r for r in records if TIMEOUT_REASON in r["reason"]])
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"records": records, "failures": dict(failures)}, f)
        print(f"\nRecords written to {args.json}")


if __name__ == "__main__":
    main()
