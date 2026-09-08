"""Paired, cluster-aware statistics for the CleanDIFT arms comparison.

WHY THIS EXISTS. Every arm in `o5_cleandift_arms.py` is evaluated on the SAME
residues, so the quantity with the smallest variance -- and the only one that
should ever be quoted -- is the PAIRED difference. Two independent binomial SEs
(the `o4_lab_arms.py` convention) throw most of that power away: 0.775 +- 0.012
vs 0.768 +- 0.012 reads as "overlapping" when the paired difference may be a
clean, tight -0.007.

And the error bar must resample CLUSTERS, not residues. 20-60 residues from one
chain are strongly correlated (same map, same resolution, same fold), so a
residue-level bootstrap or a binomial SE understates the CI. `design_effect`
below measures that inflation directly rather than assuming a factor.

The estimator is the ratio form -- sum(correct) / sum(n) over resampled clusters
-- which matches how overall accuracy is computed, so clusters contribute in
proportion to their residue count.
"""

from __future__ import annotations

import numpy as np


def cluster_bootstrap_diff(ok_a, ok_b, clusters, n_boot: int = 2000,
                           seed: int = 0) -> dict:
    """Paired accuracy difference (a - b) with a cluster-bootstrap 95% CI.

    `ok_a`/`ok_b` are per-residue correctness booleans for the two arms on the
    SAME residues in the SAME order; `clusters` is the per-residue cluster id.
    """
    ok_a = np.asarray(ok_a).astype(np.float64)
    ok_b = np.asarray(ok_b).astype(np.float64)
    clusters = np.asarray(clusters)
    if not (len(ok_a) == len(ok_b) == len(clusters)):
        raise ValueError(f"length mismatch: {len(ok_a)}, {len(ok_b)}, {len(clusters)}")
    d = ok_a - ok_b
    uniq, inv = np.unique(clusters, return_inverse=True)
    # Per-cluster sufficient statistics, so one bootstrap draw costs O(n_clusters).
    sums = np.bincount(inv, weights=d, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)

    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    # Two-sided bootstrap p-value: how much of the distribution sits on the
    # wrong side of zero. Floored at 1/n_boot -- never report p = 0.
    tail = min(float((boot <= 0.0).mean()), float((boot >= 0.0).mean()))
    p = min(1.0, max(2.0 * tail, 1.0 / n_boot))

    sd_cluster = float(boot.std(ddof=1))
    sd_residue = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {
        "diff": obs, "lo95": lo, "hi95": hi, "p": p,
        "se_cluster": sd_cluster, "se_residue_naive": sd_residue,
        # >1 means a residue-level bar is too narrow by this factor in SE terms.
        "design_effect": float(sd_cluster / sd_residue) if sd_residue > 0 else float("nan"),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "n_clusters": int(len(uniq)), "n_residues": int(len(d)),
    }


def mdi(res: dict) -> float:
    """Minimum detectable paired effect: the smallest |diff| whose 95% CI would
    exclude 0, given the measured cluster-level SE. Reported so a gate can be
    checked for feasibility BEFORE the run rather than after it."""
    return 1.96 * res["se_cluster"]


def mcnemar_exact(ok_a, ok_b) -> dict:
    """Exact McNemar on discordant pairs. SECONDARY ONLY -- it assumes
    independent residues, so it is anticonservative here by roughly
    `design_effect`. Reported for continuity with the wider literature."""
    ok_a = np.asarray(ok_a).astype(bool)
    ok_b = np.asarray(ok_b).astype(bool)
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    try:
        from scipy.stats import binomtest
        p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    except Exception:
        p = float("nan")
    return {"a_only": b, "b_only": c, "p_exact": p,
            "note": "anticonservative: assumes independent residues"}


def variance_split(ok_a, ok_b, clusters) -> dict:
    """Between- vs within-cluster variance of the paired difference.

    This is what decides `--per-chain`: adding residues per chain only shrinks
    the WITHIN component. If between dominates, more residues per chain buy
    almost nothing and the money should go to more clusters instead.
    """
    d = np.asarray(ok_a).astype(np.float64) - np.asarray(ok_b).astype(np.float64)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    cnts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    means = np.bincount(inv, weights=d, minlength=len(uniq)) / cnts
    grand = float(d.mean())
    between = float(np.sum(cnts * (means - grand) ** 2) / cnts.sum())
    within = float(np.sum((d - means[inv]) ** 2) / cnts.sum())
    tot = between + within
    return {"between": between, "within": within,
            "between_frac": float(between / tot) if tot > 0 else float("nan"),
            "mean_cluster_size": float(cnts.mean())}
