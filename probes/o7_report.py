"""Cross-arm summary of the o7 spatial-variability results.

The analysis script prints one block per arm, which is unreadable as a
comparison. This transposes the same JSON into arm-by-arm tables, and puts the
controls next to the arms they price rather than in a separate section:
`rand_student` (same architecture and receptive field, random weights) bounds
how much of any density number is reachable with no learned content, and
`esmc`/`esmc_all` bound how much the 60-residue subsample distorts things.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["esmc", "esmc_all", "raw",
         "rand_student@mid_block", "rand_student@up_blocks[1]",
         "rand_student@up_blocks[0]",
         "dec_t261@up_blocks[0]", "dec_t500@up_blocks[0]", "dec_t900@up_blocks[0]",
         "cou_best@up_blocks[0]", "dec_t500@mid_block", "dec_t500@up_blocks[1]",
         "student@up_blocks[0]", "student_paperhead@up_blocks[0]"]


def sort_arms(arms):
    return sorted(arms, key=lambda a: (ORDER.index(a) if a in ORDER else 999, a))


def f(x, w=7, p=3):
    return " " * w if x is None else (f"{'nan':>{w}}" if x != x else f"{x:{w}.{p}f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", default="results/o7_spatial_variability.json,"
                    "results/o7_spatial_variability_ctrl.json",
                    help="comma-separated; later files merge in (missing ones skipped)")
    ap.add_argument("--corr", default="results/o7_spatial_correlogram.json,"
                    "results/o7_spatial_correlogram_ctrl.json")
    args = ap.parse_args()

    J = {"arms": {}}
    for part in (Path(x) for x in args.var.split(",") if x):
        if part.exists():
            d = json.loads(part.read_text())
            J["arms"].update(d["arms"])
            J.setdefault("n_chains", d.get("n_chains"))
            J.setdefault("split_chains", d.get("split_chains"))
    C = {"arms": {}}
    for part in (Path(x) for x in args.corr.split(",") if x):
        if part.exists():
            C["arms"].update(json.loads(part.read_text())["arms"])
    arms = sort_arms(set(J["arms"]) | set(C["arms"]))
    print(f"chains {J.get('n_chains', '?')}  split {J.get('split_chains', '?')}\n")

    print("=== A  VARIANCE HIERARCHY (% of within-protein variance; global = % of total) ===")
    print(f"{'arm':34s} {'dim':>5} {'global%':>8} {'50A':>7} {'25A':>7} {'10A':>7} {'local':>7}")
    for a in arms:
        v = J["arms"].get(a, {}).get("variance")
        if not v:
            continue
        w = v["frac_within"]
        print(f"{a:34s} {v['dim']:5d} {100*v['between']:8.2f} {100*w['50A']:7.3f} "
              f"{100*w['25A']:7.3f} {100*w['10A']:7.3f} {100*w['local']:7.2f}")

    for key, title in (("r2", "all pairs"), ("r2_seqex", "through-space |i-j|>8")):
        print(f"\n=== B  RIDGE R2, residue i -> residue j, vs 3D DISTANCE ({title}) ===")
        bins = None
        for a in arms:
            rows = J["arms"].get(a, {}).get("dist3d")
            if rows and bins is None:
                bins = [r["bin"] for r in rows]
        if bins is None:
            continue
        print(f"{'arm':34s} " + " ".join(f"{b:>7}" for b in bins))
        for a in arms:
            rows = J["arms"].get(a, {}).get("dist3d")
            if not rows:
                continue
            print(f"{a:34s} " + " ".join(f(r[key]) for r in rows))

    print("\n=== B2 RIDGE R2 vs SEQUENCE SEPARATION (through-chain reference) ===")
    bins = None
    for a in arms:
        rows = J["arms"].get(a, {}).get("seqsep")
        if rows and bins is None:
            bins = [r["bin"] for r in rows]
    if bins:
        print(f"{'arm':34s} " + " ".join(f"{b:>7}" for b in bins))
        for a in arms:
            rows = J["arms"].get(a, {}).get("seqsep")
            if rows:
                print(f"{a:34s} " + " ".join(f(r["r2"]) for r in rows))

    print("\n=== C  RIDGE R2, NEIGHBOURHOOD(mean+std, |i-j|>8) -> residue, vs RADIUS ===")
    rad = None
    for a in arms:
        rows = J["arms"].get(a, {}).get("neighbourhood")
        if rows and rad is None:
            rad = [f"{r['radius']:.0f}A" for r in rows]
    if rad:
        print(f"{'arm':34s} " + " ".join(f"{r:>7}" for r in rad))
        for a in arms:
            rows = J["arms"].get(a, {}).get("neighbourhood")
            if rows:
                print(f"{a:34s} " + " ".join(f(r["r2"]) for r in rows))

    print("\n=== D  CENTRED-COSINE HALF-DECAY L (A).  cos@ref = similarity in the "
          "shortest populated bin ===")
    print("    NOTE the band columns are metric-limited: per-protein mean removal caps "
          "measurable L at\n    ~diameter/4 (~15-17 A), so 50A and 25A bands read the "
          "same for EVERY arm incl. random weights.\n    Only the `feature` column "
          "discriminates.")
    print(f"{'arm':34s} {'L(feat)':>8} {'cos@ref':>8} {'L(50A)':>7} {'L(25A)':>7} "
          f"{'L(10A)':>7} {'L(local)':>8}")
    for a in arms:
        c = C["arms"].get(a, {}).get("correlogram")
        if not c:
            continue
        s = c["series"]
        cen = lambda k: "*" if s[k]["censored"] else " "
        print(f"{a:34s} {f(s['feature']['half_decay'],8,2)}{cen('feature')}"
              f"{f(s['feature']['ref_cos'],7,4)} "
              f"{f(s['50.0Å']['half_decay'],7,2)} {f(s['25.0Å']['half_decay'],7,2)} "
              f"{f(s['10.0Å']['half_decay'],7,2)} {f(s['private']['half_decay'],8,2)}")
    print("    * = censored (never reached half by the last bin; L is a lower bound)")
    print("    nan in L(feat) = cos@ref already at or below half of itself, i.e. no "
          "measurable spatial\n        correlation even between adjacent residues.")


if __name__ == "__main__":
    main()
