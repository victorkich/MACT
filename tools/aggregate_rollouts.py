"""Aggregate rollout records into the statistic the paper reports.

Table 2 gives, per map, the median win rate across training seeds with its standard
deviation, where each seed's win rate is measured over a fixed number of evaluation
episodes. This reproduces that: win rate per seed, then median and std across seeds.
"""
import argparse
import glob
import json
import os
import re
import statistics as st

# Table 2 of the paper — median (std) win rate %
PAPER = {
    "3s_vs_3z":         {"MACT": (80.0, 3.4),  "MARIE": (85.0, 21.8)},
    "so_many_baneling": {"MACT": (94.4, 7.9),  "MARIE": (73.0, 12.4)},
    "MMM":              {"MACT": (39.2, 32.6), "MARIE": (1.0, 1.6)},
    "8m":               {"MACT": (95.0, 5.0),  "MARIE": (72.0, 7.1)},
}
MAPKEY = {"3s3z": "3s_vs_3z", "smb": "so_many_baneling", "MMM": "MMM", "8m": "8m"}


def load(dirpath):
    """dirpath/<method>_<mapkey>_s<N>.json -> {(method, map): {seed: record}}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "*_s[0-9].json"))):
        m = re.match(r"(mact|marie)_(.+)_s(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        method, mk, seed = m.group(1).upper(), m.group(2), int(m.group(3))
        if mk not in MAPKEY:
            continue
        d = json.load(open(f))
        out.setdefault((method, MAPKEY[mk]), {})[seed] = d
    return out


def summarize(groups):
    rows = []
    for (method, mp), seeds in sorted(groups.items()):
        per_seed = []
        for s in sorted(seeds):
            d = seeds[s]
            n = d["n_total"]
            if n == 0:
                continue
            per_seed.append((s, 100.0 * d["n_won"] / n, n))
        if not per_seed:
            continue
        wrs = [w for _, w, _ in per_seed]
        med = st.median(wrs)
        sd = st.stdev(wrs) if len(wrs) > 1 else 0.0
        paper = PAPER.get(mp, {}).get(method)
        rows.append(dict(method=method, map=mp, n_seeds=len(wrs),
                         eps=[n for _, _, n in per_seed], per_seed=per_seed,
                         median=med, std=sd, mean=sum(wrs) / len(wrs),
                         paper_med=paper[0] if paper else None,
                         paper_std=paper[1] if paper else None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/rollouts")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    rows = summarize(load(args.dir))
    if not rows:
        print("no rollout files matched <method>_<map>_s<N>.json")
        return

    hdr = f"{'method':6s} {'map':18s} {'seeds':>5s} {'eps/seed':>9s} {'per-seed win %':<26s} {'median':>7s} {'std':>6s} | {'paper':>13s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ps = " ".join(f"{w:.0f}" for _, w, _ in r["per_seed"])
        eps = ",".join(str(e) for e in sorted(set(r["eps"])))
        paper = (f"{r['paper_med']:.1f} ±{r['paper_std']:.1f}"
                 if r["paper_med"] is not None else "—")
        print(f"{r['method']:6s} {r['map']:18s} {r['n_seeds']:5d} {eps:>9s} "
              f"{ps:<26s} {r['median']:7.1f} {r['std']:6.1f} | {paper:>13s}")

    print("\nper-seed detail")
    for r in rows:
        for s, w, n in r["per_seed"]:
            print(f"  {r['method']:6s} {r['map']:18s} seed{s}  {w:5.1f}%  ({n} episodes)")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
