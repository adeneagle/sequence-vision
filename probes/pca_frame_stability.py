"""Is PCA frame averaging viable? The cheap half of the question, CPU only.

Frame averaging (Puny et al., ICLR 2022) gives EXACT rotation invariance by
averaging features over the sign-flip frames of a PCA basis -- provided the basis
itself is well defined. Two independent failure modes:

  (1) washout   -- averaging over frames whose features are dissimilar destroys
                   the signal. Needs a GPU; not tested here.
  (2) degeneracy-- if two inertia eigenvalues are close, the AXES are unstable,
                   not merely their signs. Sign averaging cannot repair this: a
                   near-degenerate pair rotates freely, so the frame *set* is not
                   equivariant and invariance is lost. CPU only -- tested here.

If (2) fails there is no point spending GPU on (1).

Note the failure is about stability under PERTURBATION, not under exact rotation.
PCA is exactly equivariant for a rigidly rotated point set regardless of
degeneracy. What breaks is that two similar structures -- or the same structure
with coordinate noise -- get frames that differ by a large rotation, making the
target a discontinuous function across the dataset.

    pixi run python probes/pca_frame_stability.py --noise 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from probes.stability import load_ca


def pca_frame(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal axes (rows) and eigenvalues, descending."""
    xc = x - x.mean(0)
    cov = xc.T @ xc / len(xc)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return v[:, order].T, w[order]


def axis_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-axis angle (deg) between two frames, after resolving sign ambiguity."""
    dots = np.abs(np.sum(A * B, axis=1)).clip(0, 1)   # |cos| removes the sign flip
    return np.degrees(np.arccos(dots))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--noise", type=float, default=0.5,
                    help="Angstrom sigma of Ca perturbation (~model coordinate error)")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("results/pca_frame_stability.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = list(csv.DictReader(open(args.manifest)))
    recs = []

    for row in rows:
        try:
            ca = load_ca(row["cif_path"])
            if len(ca) < 50:
                continue
            F0, w = pca_frame(ca)
            # relative eigenvalue gaps -- small gap => unstable axis pair
            gap12 = (w[0] - w[1]) / w[0]
            gap23 = (w[1] - w[2]) / w[0]

            angs = []
            for _ in range(args.trials):
                pert = ca + rng.normal(0, args.noise, ca.shape)
                F1, _ = pca_frame(pert)
                angs.append(axis_angles(F0, F1))
            angs = np.array(angs)                     # [trials, 3]

            recs.append({
                "emdb_id": row["emdb_id"], "n": len(ca),
                "eig": w.tolist(), "gap12": float(gap12), "gap23": float(gap23),
                "min_gap": float(min(gap12, gap23)),
                "max_axis_angle": float(angs.max()),
                "median_axis_angle": float(np.median(angs)),
            })
        except Exception as exc:
            print(f"{row['emdb_id']}: {type(exc).__name__}: {exc}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(args.out, "w"), indent=2)

    mg = np.array([r["min_gap"] for r in recs])
    ma = np.array([r["max_axis_angle"] for r in recs])
    md = np.array([r["median_axis_angle"] for r in recs])

    print(f"\n{len(recs)} structures, Ca perturbation sigma = {args.noise} A, "
          f"{args.trials} trials each\n")
    print("Relative eigenvalue gap  min(gap12, gap23):")
    print(f"  median {np.median(mg):.3f}   p10 {np.percentile(mg,10):.3f}   "
          f"min {mg.min():.3f}")
    print(f"  fraction with gap < 0.10 (near-degenerate): {np.mean(mg < 0.10):.2f}")
    print(f"  fraction with gap < 0.05                  : {np.mean(mg < 0.05):.2f}")
    print("\nAxis rotation induced by the perturbation (degrees):")
    print(f"  median-of-medians {np.median(md):.2f}   median-of-max {np.median(ma):.2f}   "
          f"worst {ma.max():.1f}")
    print(f"  fraction of structures with any axis moving > 10 deg: {np.mean(ma > 10):.2f}")
    print(f"  fraction > 30 deg                                   : {np.mean(ma > 30):.2f}")

    print("\nWorst 5 by axis movement:")
    for r in sorted(recs, key=lambda r: -r["max_axis_angle"])[:5]:
        print(f"  {r['emdb_id']:<12} n={r['n']:<5} min_gap={r['min_gap']:.3f} "
              f"max_axis_angle={r['max_axis_angle']:6.1f} deg")

    frac_bad = float(np.mean(ma > 10))
    print("\nVERDICT: ", end="")
    if frac_bad > 0.2:
        print(f"PCA frames are UNSTABLE for {frac_bad:.0%} of structures at "
              f"{args.noise} A noise.\n  Frame averaging over sign flips does NOT repair this -- "
              "it fixes signs, not axis\n  directions. Not worth GPU time on the washout question.")
    else:
        print(f"PCA frames are stable for {1-frac_bad:.0%} of structures. "
              "Washout test is worth running.")


if __name__ == "__main__":
    main()
