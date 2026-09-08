"""STEP 0b -- is a 1.0-point paired effect detectable AT ALL on this split?

Run this BEFORE any GPU spend. Gate condition 2 of PLAN_CLEANDIFT.md ("the null
pair must resolve better than 0.005") can otherwise only fail AFTER ~20
GPU-hours, which is the wrong order. Everything needed is already cached in
`data/o4_lab_parts`, so this is minutes of CPU.

It answers three questions:

  1. MDE -- the minimum detectable paired effect at the real cluster count. If
     that exceeds 1.0 point, the pre-registered endpoint is not measurable on
     this split and enlarging the test set is the prerequisite, not training.
  2. Design effect -- how much too narrow a residue-level bar would be. The plan
     asserts "roughly 3x"; this measures it instead of assuming it.
  3. Whether `--per-chain` is worth raising, via the between/within variance
     split and a direct 10-vs-20 residues-per-chain comparison. Only the WITHIN
     component shrinks with more residues per chain.

`lab3 - lab2` is the pre-existing analogue of the plan's `cou_seed1` null pair:
two arbitrary fixed global orientations, so the true difference is ~0 and any
measured gap is pure measurement noise.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from probes.o5_stats import (cluster_bootstrap_diff, mcnemar_exact, mdi,
                             variance_split)

ARMS = ("aligned", "lab", "lab2", "lab3", "labavg4", "labavg8", "raw")


def load_parts(feat_dir: Path, per_chain: int | None = None) -> dict:
    keys = ARMS + ("aa", "ss", "split", "cluster")
    buf: dict[str, list] = {k: [] for k in keys}
    for p in sorted(glob.glob(str(feat_dir / "*.npz"))):
        z = np.load(p, allow_pickle=True)
        missing = set(keys) - set(z.files)
        if missing:                      # stale cache from before an arm was added
            raise KeyError(f"{p} lacks {sorted(missing)}; delete and re-extract")
        n = len(z["aa"])
        sel = slice(None) if per_chain is None else slice(0, min(per_chain, n))
        for k in keys:
            buf[k].append(z[k][sel])
    return {k: np.concatenate(v) for k, v in buf.items()}


def fit_correctness(F: dict, task: str, arms=ARMS) -> tuple[dict, dict]:
    """Per-residue correctness on the TEST split for each arm."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sp = F["split"].astype(str)
    tr, te = sp == "train", sp == "test"
    y = F[task]
    ok, acc = {}, {}
    for arm in arms:
        X = F[arm]
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        ok[arm] = (pred == y[te])
        acc[arm] = float(ok[arm].mean())
    return ok, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", type=Path, default=Path("data/o4_lab_parts"))
    ap.add_argument("--task", default="ss", choices=["ss", "aa"])
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--bar", type=float, default=0.010,
                    help="the pre-registered PASS effect size to test feasibility of")
    ap.add_argument("--out", type=Path, default=Path("results/o5_power_check.json"))
    args = ap.parse_args()

    # Pairs: one known-large, one near-null, one small, one vs the raw baseline.
    PAIRS = [("lab", "aligned"), ("lab3", "lab2"), ("labavg8", "labavg4"),
             ("aligned", "raw")]

    res: dict = {"_meta": {"task": args.task, "n_boot": args.n_boot,
                           "bar": args.bar, "feat_dir": str(args.feat_dir)}}
    for per_chain in (None, 10):
        tag = "per_chain_all" if per_chain is None else f"per_chain_{per_chain}"
        F = load_parts(args.feat_dir, per_chain)
        sp = F["split"].astype(str)
        te = sp == "test"
        clusters = F["cluster"].astype(str)[te]
        ok, acc = fit_correctness(F, args.task)
        n_cl = len(np.unique(clusters))
        print(f"\n=== {tag}: {int(te.sum())} test residues, {n_cl} test clusters ===",
              flush=True)
        print("  acc: " + "  ".join(f"{a}={acc[a]:.4f}" for a in ARMS), flush=True)
        block: dict = {"acc": acc, "n_test": int(te.sum()), "n_clusters": n_cl,
                       "pairs": {}}
        for a, b in PAIRS:
            r = cluster_bootstrap_diff(ok[a], ok[b], clusters, n_boot=args.n_boot)
            r["mde_95"] = mdi(r)
            r["mcnemar"] = mcnemar_exact(ok[a], ok[b])
            r["variance"] = variance_split(ok[a], ok[b], clusters)
            block["pairs"][f"{a}-{b}"] = r
            print(f"  {a:9s} - {b:9s} {r['diff']:+.4f}  "
                  f"CI [{r['lo95']:+.4f},{r['hi95']:+.4f}]  "
                  f"SE_cl {r['se_cluster']:.4f} (naive {r['se_residue_naive']:.4f}, "
                  f"deff {r['design_effect']:.2f})  MDE {r['mde_95']:.4f}"
                  f"{'  *' if r['excludes_zero'] else ''}", flush=True)
        res[tag] = block

    # --- verdict -----------------------------------------------------------
    mdes = [res["per_chain_all"]["pairs"][k]["mde_95"] for k in res["per_chain_all"]["pairs"]]
    mde = float(np.median(mdes))
    deffs = [res["per_chain_all"]["pairs"][k]["design_effect"]
             for k in res["per_chain_all"]["pairs"]]
    null = res["per_chain_all"]["pairs"]["lab3-lab2"]
    se_all = np.median([res["per_chain_all"]["pairs"][k]["se_cluster"]
                        for k in res["per_chain_all"]["pairs"]])
    se_10 = np.median([res["per_chain_10"]["pairs"][k]["se_cluster"]
                       for k in res["per_chain_10"]["pairs"]])
    verdict = {
        "median_mde_95": mde,
        "bar": args.bar,
        "bar_detectable": bool(mde <= args.bar),
        "median_design_effect": float(np.median(deffs)),
        "null_pair_diff": null["diff"],
        "null_pair_resolves_005": bool(abs(null["diff"]) < 0.005),
        "se_halving_from_doubling_residues_per_chain": float(se_10 / se_all)
        if se_all > 0 else float("nan"),
        "per_chain_verdict": (
            "raising --per-chain helps: SE shrank materially when residues doubled"
            if se_all < 0.9 * se_10 else
            "raising --per-chain is NOT worth it: SE barely moved when residues "
            "doubled, so between-cluster variance dominates. Spend on clusters."),
    }
    # --- projection: what scale WOULD resolve the bar? ---------------------
    # SE of a cluster-bootstrap mean scales ~ 1/sqrt(n_clusters), which dominates;
    # the residues-per-chain axis is measured directly above and folded in as a
    # multiplier rather than assumed.
    n_cl_now = res["per_chain_all"]["n_clusters"]
    se_now = float(se_all)
    se_target = args.bar / 1.96
    # per-chain gain: measured SE ratio for 10 -> 20 residues, extrapolated to 60
    # as (ratio)^log2(3), i.e. the same per-doubling gain applied 1.58 doublings.
    per_doubling = float(se_all / se_10) if se_10 > 0 else 1.0
    gain_20_to_60 = per_doubling ** (np.log2(3.0))
    se_at_60 = se_now * gain_20_to_60
    need_ratio = (se_at_60 / se_target) ** 2
    verdict["projection"] = {
        "se_now_at_20_per_chain": se_now,
        "measured_se_ratio_per_doubling_of_residues": per_doubling,
        "projected_se_at_60_per_chain": se_at_60,
        "se_needed_for_bar": se_target,
        "test_clusters_now": n_cl_now,
        "test_clusters_needed_at_60_per_chain": float(n_cl_now * need_ratio),
        "test_clusters_in_full_split": 100,
        "projected_mde_on_full_split_at_60": float(
            1.96 * se_at_60 * np.sqrt(n_cl_now / 100.0)),
        "note": "the full 1500-chain split has 100 test clusters; this projects the "
                "MDE there from the measured SE on this 395-chain subset",
    }

    res["_verdict"] = verdict
    print("\n=== VERDICT ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    if not verdict["bar_detectable"]:
        print("\n  *** A 1.0-point effect is NOT resolvable on this split. "
              "Per PLAN step 0b: STOP and enlarge the test cluster count. ***")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
