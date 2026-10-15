"""
seat_balance.py

Compare starting-goods setups for seat balance: the same checkpoint plays all three
seats (ab_eval three-way), once per setup, and the share of decided games each seat
wins is reported with a 95% interval. Fair seats would each win about a third.

Each setup is a copy of the game ini with only [Rules] starting_common_goods_by_seat
and starting_rare_goods_by_seat changed, written to seat_tests/; the live mali_ba.ini
is never touched. Every setup uses the same --seed, so all of them are played on the
same boards, which makes the comparisons tighter.

Setups are given as COMMON/RARE count lists, one entry per seat from seat 0, or
"base" for the ini's own goods, optionally followed by other [Rules] settings after
semicolons:
    base            the current rules (0,1,2 common, no rare as of 2026-10-15)
    0,2,1/0,0,1     seat 1 two common goods; seat 2 one common and one rare
    "base;end_game_finish_round=true;main_play_start_seat=2"
                    the ini's goods, finish the round, seat 2 moves first
(Quote a setup that contains semicolons, or the shell splits it.) In file names
end_game_finish_round becomes fr and main_play_start_seat ss, e.g. base_fr1_ss2.

Needs the engine build with per-seat starting goods (commit 40449f0) and, for the
end-of-game and start-seat rules, the build after it (2026-10-15); an older build
ignores them and silently plays the base game. Each run's ab_eval log shows
"starting goods by seat ... common X | rare Y" and "end_game_finish_round: ... |
main_play_start_seat: ..." as applied, and the summary warns on a mismatch.

Usage (from this directory, with the trainer's environment):
    python seat_balance.py base 0,2,1/0,0,1 0,1,1/0,0,1
    python seat_balance.py --agent mali_ba_agent_vD028.weights.h5 --games 504 base 0,2,1/0,0,1
    python seat_balance.py --summary_only base 0,2,1/0,0,1      # re-read finished logs

A finished setup (its log has a result) is skipped on a rerun unless --rerun is given.
Each setup takes about as long as one head-to-head (~25-30 min at 64 workers).
"""
import argparse
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "seat_tests")
RE_SEATS = re.compile(r"wins by seat \(turn-order effect\):\s+(.*)")
RE_PLAYED = re.compile(r"games played\s*:\s*(\d+)\s+\(decided (\d+), timed out (\d+)\)")
RE_APPLIED = re.compile(r"starting goods by seat .*?common ([\d,]*) \| rare ([\d,]*)")
RE_RULES = re.compile(r"end_game_finish_round: (true|false) \| main_play_start_seat: (\d+)")
SHORT = {"end_game_finish_round": "fr", "main_play_start_seat": "ss"}


def split_setup(spec):
    """'0,2,1/0,0,1;key=v;...' -> ('0,2,1/0,0,1', {'key': 'v', ...})."""
    head, *rest = [p.strip() for p in spec.split(";") if p.strip()]
    extras = {}
    for kv in rest:
        if "=" not in kv:
            sys.exit(f"setup '{spec}': '{kv}' is not key=value")
        k, v = (x.strip() for x in kv.split("=", 1))
        extras[k] = v
    return head, extras


def parse_setup(spec):
    """Goods part: 'base' -> None; '0,2,1/0,0,1' -> ('0,2,1', '0,0,1')."""
    spec, _ = split_setup(spec)
    if spec == "base":
        return None
    parts = spec.split("/")
    if len(parts) != 2:
        sys.exit(f"setup '{spec}': expected COMMON/RARE, e.g. 0,2,1/0,0,1, or 'base'")
    for p in parts:
        if not re.fullmatch(r"\d+(,\d+)*", p):
            sys.exit(f"setup '{spec}': '{p}' is not a comma-separated list of counts")
    return parts[0], parts[1]


def tag_for(spec):
    goods = parse_setup(spec)
    tag = "base" if goods is None else f"c{goods[0].replace(',', '')}_r{goods[1].replace(',', '')}"
    for k, v in split_setup(spec)[1].items():
        short = SHORT.get(k, "".join(w[0] for w in k.split("_")))
        val = {"true": "1", "false": "0"}.get(v.lower(), v.replace(",", ""))
        tag += f"_{short}{val}"
    return tag


def write_ini(base_ini, spec, path):
    """Copy base_ini with the setup's [Rules] settings replaced (or appended)."""
    goods = parse_setup(spec)
    lines = open(base_ini).read().splitlines()
    want = dict(split_setup(spec)[1])
    if goods is not None:
        want["starting_common_goods_by_seat"], want["starting_rare_goods_by_seat"] = goods
    seen = set()
    out, section, rules_end = [], None, None
    for line in lines:
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            if section == "Rules":
                rules_end = len(out)
            section = s[1:-1]
        key = s.split("=", 1)[0].strip() if "=" in s and not s.startswith(("#", ";")) else None
        if section == "Rules" and key in want:
            out.append(f"{key} = {want[key]}")
            seen.add(key)
            continue
        out.append(line)
    missing = [k for k in want if k not in seen]
    if missing:
        if rules_end is None:
            sys.exit(f"{base_ini}: no [Rules] section to add {missing} to")
        out[rules_end:rules_end] = [f"{k} = {want[k]}" for k in missing]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def read_result(log_path):
    if not os.path.exists(log_path):
        return None
    text = open(log_path, errors="replace").read()
    seats, played, applied = RE_SEATS.search(text), RE_PLAYED.search(text), RE_APPLIED.search(text)
    rules = RE_RULES.search(text)
    if not (seats and played):
        return None
    wins = [int(v) for v in re.findall(r"seat \d+: (\d+)", seats.group(1))]
    return dict(wins=wins, games=int(played.group(1)), decided=int(played.group(2)),
                timeouts=int(played.group(3)),
                applied=(applied.group(1), applied.group(2)) if applied else None,
                rules=(rules.group(1), rules.group(2)) if rules else None)


def main():
    ap = argparse.ArgumentParser(description="Seat balance of starting-goods setups (same model in every seat).")
    ap.add_argument("setups", nargs="+", help="'base' or COMMON/RARE lists, e.g. 0,2,1/0,0,1")
    ap.add_argument("--agent", default="mali_ba_agent_vD028.weights.h5")
    ap.add_argument("--config_file", default=os.path.join(HERE, "mali_ba.ini"),
                    help="Base ini to copy (default: the live mali_ba.ini, read only)")
    ap.add_argument("--games", type=int, default=504, help="Per setup; a multiple of 6")
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--uct_c", type=float, default=1.0)
    ap.add_argument("--heuristic_weight", type=float, default=0.10,
                    help="Guidance share (normalized blend). 0.10 = training's")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12345, help="Same for every setup: same boards")
    ap.add_argument("--rerun", action="store_true", help="Rerun setups that already have a result")
    ap.add_argument("--summary_only", action="store_true", help="Only summarize existing logs")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    agent_tag = os.path.basename(args.agent).replace("mali_ba_agent_v", "").replace(".weights.h5", "")
    h_tag = f"h{int(round(args.heuristic_weight * 100)):03d}"
    runs = []
    for spec in args.setups:
        parse_setup(spec)  # validate before anything runs
        tag = tag_for(spec)
        log = os.path.join(OUT_DIR, f"ab3_seats_{tag}_{agent_tag}_u{args.uct_c:g}_{h_tag}.log")
        runs.append((spec, tag, log))

    for spec, tag, log in runs:
        if args.summary_only:
            break
        if read_result(log) and not args.rerun:
            print(f"{spec}: already done ({os.path.relpath(log, HERE)}), skipping")
            continue
        ini = args.config_file
        if tag != "base":
            ini = os.path.join(OUT_DIR, f"mali_ba_seats_{tag}.ini")
            write_ini(args.config_file, spec, ini)
        cmd = [sys.executable, os.path.join(HERE, "ab_eval.py"),
               "--agent_a", args.agent, "--agent_b", args.agent, "--agent_c", args.agent,
               "--uct_c", str(args.uct_c), "--guidance_blend_normalized",
               "--heuristic_weight", str(args.heuristic_weight), "--sims", str(args.sims),
               "--games", str(args.games), "--workers", str(args.workers),
               "--seed", str(args.seed), "--config_file", ini]
        print(f"\n=== {spec}  ->  {os.path.relpath(log, HERE)}\n    {' '.join(cmd)}", flush=True)
        with open(log, "w") as f:
            proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            proc.wait()
        if proc.returncode != 0:
            print(f"ab_eval exited with code {proc.returncode} for {spec}; see {log}")

    print("\nSeat balance (share of decided games won by each seat, 95% interval; fair = 33.3% each)")
    print(f"agent {os.path.basename(args.agent)}, {args.sims} sims, uct_c {args.uct_c:g}, "
          f"guidance {args.heuristic_weight:g}, seed {args.seed}\n")
    w = max(16, max(len(spec) for spec, _, _ in runs))
    print(f"{'setup':<{w}} {'decided':>8} {'t/o':>4}   " + "   ".join(f"{'seat ' + str(s):^22}" for s in range(3))
          + f"   {'spread':>6}")
    for spec, tag, log in runs:
        r = read_result(log)
        if r is None:
            print(f"{spec:<{w}} (no result yet: {os.path.relpath(log, HERE)})")
            continue
        n = r["decided"]
        cells, shares = [], []
        for k in r["wins"]:
            lo, hi = wilson(k, n)
            shares.append(k / n if n else float("nan"))
            cells.append(f"{100 * k / n:5.1f}% [{100 * lo:4.1f},{100 * hi:4.1f}]" if n else "n/a")
        print(f"{spec:<{w}} {n:8d} {r['timeouts']:4d}   " + "   ".join(f"{c:^22}" for c in cells)
              + f"   {100 * (max(shares) - min(shares)):5.1f}")
        extras = split_setup(spec)[1]
        if extras.keys() & SHORT.keys():
            if r["rules"] is None:
                print(f"{'':<{w}} WARNING: the log has no 'end_game_finish_round' line; the engine "
                      f"may be an older build that ignores the new rules")
            else:
                got = {"end_game_finish_round": r["rules"][0], "main_play_start_seat": r["rules"][1]}
                for k in extras.keys() & SHORT.keys():
                    if extras[k].lower() != got[k]:
                        print(f"{'':<{w}} WARNING: the engine applied {k} = {got[k]}, not {extras[k]}")
        if parse_setup(spec) is not None:
            want = parse_setup(spec)
            if r["applied"] is None:
                print(f"{'':<{w}} WARNING: the log has no 'starting goods by seat' line; the engine "
                      f"may be an older build that ignores the lists")
            elif tuple(r["applied"]) != want:
                print(f"{'':<{w}} WARNING: the engine applied {r['applied']}, not {want}")
    print("\nspread = largest minus smallest seat share, in points (0 = perfectly even).")


if __name__ == "__main__":
    main()
