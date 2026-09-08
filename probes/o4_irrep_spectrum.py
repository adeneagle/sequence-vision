"""O4 Task 1: does the CryoFM2 channel covariance show (2l+1) degeneracies?

If rotating the input acts on channel space by a global orthogonal Q_R, then over
a rotation-INVARIANT sample of voxels the second moment C = E[f f^T] satisfies
Q_R C Q_R^T = C, so C commutes with every Q_R. By Schur, on each isotypic
component C = (m_l x m_l Hermitian) (x) I_{2l+1}, i.e. **every eigenvalue of C has
multiplicity exactly (2l+1) -- always ODD**. Clusters of 3/5/7 near-equal
eigenvalues are then evidence for l=1/2/3. This needs no rotations at all, which
is why it is the cheap pre-test.

TWO DESIGN POINTS THAT DECIDE WHETHER THE TEST MEANS ANYTHING:

1. **LAB-frame voxels, not local backbone frames.** The alignment/O1 caches are
   cut in each residue's N-CA-C frame, so their feature distribution is
   deliberately CANONICALISED and NOT rotation-invariant; C would not commute with
   Q_R and the test would return a false negative. We sample from lab-frame
   feature volumes of deposited maps, whose orientations are arbitrary, so the
   aggregate sample is approximately rotation-invariant.
2. **The random-weight null is load-bearing, not hygiene.** Any 512-dim
   covariance has accidental near-degeneracies in its tail, and "clusters of 3, 5,
   7" is exactly the kind of pattern that is easy to see and easy to imagine. Only
   the pretrained-vs-random DIFFERENCE is evidence. This control has reversed
   conclusions twice in this project.

Also controlled: voxels within `--margin` of the box faces are excluded (U-Net
padding breaks equivariance there), and voxels are drawn from the high-density
region (a rotation-invariant, scalar criterion) so the covariance is not dominated
by solvent.

Statistic: sort eigenvalues descending, group consecutive ones whose relative gap
is below a tolerance, and report the run-length histogram. Under the hypothesis
ALL multiplicities are odd, so the ODD-run fraction should exceed the null's.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def run_lengths(eig: np.ndarray, tol: float) -> list[int]:
    """Group consecutive (descending) eigenvalues with relative gap < tol."""
    runs, cur = [], 1
    for i in range(len(eig) - 1):
        denom = max(abs(eig[i]), 1e-12)
        if (eig[i] - eig[i + 1]) / denom < tol:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    return runs


def summarize(eig: np.ndarray, tag: str, tols=(0.005, 0.01, 0.02, 0.05)) -> dict:
    out = {}
    keep = eig[eig > eig.max() * 1e-6]        # drop numerically dead directions
    print(f"  [{tag}] {len(keep)} live dims of {len(eig)}; "
          f"top eigs {np.array2string(keep[:6], precision=3)}")
    for tol in tols:
        r = run_lengths(keep, tol)
        hist = {}
        for x in r:
            hist[str(x)] = hist.get(str(x), 0) + 1
        # A length-1 run is trivially odd, and singletons dominate, so an
        # odd-fraction over ALL runs just measures the singleton rate and says
        # nothing about (2l+1) structure. Restrict to runs of length >= 3, which
        # are the only ones the hypothesis predicts.
        multi = [x for x in r if x >= 3]
        odd_multi = sum(1 for x in multi if x % 2 == 1)
        out[f"tol{tol}"] = {
            "n_runs": len(r), "n_singleton": hist.get("1", 0),
            "n_runs_ge3": len(multi),
            "odd_frac_ge3": (odd_multi / len(multi)) if multi else None,
            "dims_in_runs_ge3": sum(multi),
            "runs_of_3": hist.get("3", 0), "runs_of_5": hist.get("5", 0),
            "runs_of_7": hist.get("7", 0),
            "hist": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))[:10])}
        of = out[f"tol{tol}"]["odd_frac_ge3"]
        print(f"    tol {tol:<6}: {len(r):4d} runs ({hist.get('1',0):4d} singletons), "
              f"{len(multi):3d} runs>=3, odd-frac(>=3) "
              f"{('%.3f' % of) if of is not None else '  n/a'}, "
              f"len3 {hist.get('3',0):3d} len5 {hist.get('5',0):3d} len7 {hist.get('7',0):3d}")
    return out


def collect(tap_obj, rows, args, dev):
    """Second moment of lab-frame features over high-density interior voxels."""
    from teachers.cryofm_tap import MODEL_VOXEL_SIZE, preprocess, sample_at
    from probes.stability import load_map
    from probes.o1_cryofm_benchmark import to_cubic_even

    C = None; n_tot = 0; mean = None; used = 0
    rng = np.random.default_rng(args.seed)
    for i, r in enumerate(rows):
        try:
            d = Path(r["pdb"]).parent
            vol, vs, _ = load_map(str(d / f"{d.name}_raw_emd.map"))
            norm = preprocess(to_cubic_even(vol), vs)
            fv = tap_obj.feature_volumes(
                norm, timestep=args.timestep,
                noise_level=(args.timestep / 1000.0) if args.couple else 0.0)[args.tap]
            F = fv.data if hasattr(fv, "data") else fv        # [C, d, d, d]
            F = np.asarray(F)
            Cc, D, H, W = F.shape
            m = args.margin
            if min(D, H, W) <= 2 * m + 2:
                raise ValueError("feature volume too small for the margin")
            # scalar, rotation-invariant selection: high-density interior voxels
            sub = F[:, m:D - m, m:H - m, m:W - m].reshape(Cc, -1)
            power = (sub ** 2).sum(0)
            thr = np.percentile(power, args.pct)
            idx = np.nonzero(power >= thr)[0]
            if len(idx) > args.per_map:
                idx = rng.choice(idx, args.per_map, replace=False)
            X = sub[:, idx].T.astype(np.float64)              # [n, C]
            C = X.T @ X if C is None else C + X.T @ X
            mean = X.sum(0) if mean is None else mean + X.sum(0)
            n_tot += len(X); used += 1
        except Exception as exc:
            if used < 8:
                print(f"    skip {r['key']}: {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(rows)}] {used} maps, {n_tot} voxels", flush=True)
    if C is None:
        raise SystemExit("no maps usable")
    C /= n_tot; mean = mean / n_tot
    return C, C - np.outer(mean, mean), n_tot, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--tap", default="up_blocks[1]")
    ap.add_argument("--timestep", type=int, default=261)
    ap.add_argument("--couple", action="store_true")
    ap.add_argument("--limit", type=int, default=60, help="maps")
    ap.add_argument("--per-map", type=int, default=4000, help="voxels per map")
    ap.add_argument("--pct", type=float, default=90.0,
                    help="keep voxels above this percentile of feature power "
                         "(concentrates on protein, not solvent; scalar criterion "
                         "so it does not break rotation invariance)")
    ap.add_argument("--margin", type=int, default=8,
                    help="feature cells excluded at each box face -- U-Net padding "
                         "breaks equivariance there")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/o4_irrep_spectrum.json"))
    args = ap.parse_args()

    from teachers.cryofm_tap import CryoFM2Tap
    rows = list(csv.DictReader(open(args.chains)))
    seen, uniq = set(), []
    for r in rows:                                  # one chain per EMDB entry
        if r["emd"] in seen:
            continue
        seen.add(r["emd"]); uniq.append(r)
    rows = uniq[: args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(rows)} maps | tap={args.tap} | t={args.timestep} "
          f"| {'COUPLED' if args.couple else 'decoupled'} | {dev}", flush=True)

    res = {}
    for arm, rand in (("pretrained", False), ("random", True)):
        print(f"\n=== {arm} ===", flush=True)
        tap_obj = CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                             batch_size=8, random_weights=rand)
        C2, Ccen, n_tot, used = collect(tap_obj, rows, args, dev)
        del tap_obj
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"  {used} maps, {n_tot} voxels")
        e2 = np.linalg.eigvalsh(C2)[::-1]
        ec = np.linalg.eigvalsh(Ccen)[::-1]
        res[arm] = {"n_voxels": int(n_tot), "n_maps": int(used),
                    "uncentered": summarize(e2, f"{arm} E[ff^T]"),
                    "centered": summarize(ec, f"{arm} centered"),
                    "eigs_uncentered_top64": [float(x) for x in e2[:64]]}

    p, q = res["pretrained"]["uncentered"], res["random"]["uncentered"]
    print("\n=== pretrained MINUS random (only the difference is evidence) ===")
    fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"
    for tol in ("tol0.005", "tol0.01", "tol0.02", "tol0.05"):
        print(f"  {tol:<10} runs>=3 {p[tol]['n_runs_ge3']:3d} vs {q[tol]['n_runs_ge3']:3d}"
              f"   odd-frac(>=3) {fmt(p[tol]['odd_frac_ge3'])} vs {fmt(q[tol]['odd_frac_ge3'])}"
              f"   len3 {p[tol]['runs_of_3']:3d} vs {q[tol]['runs_of_3']:3d}"
              f"   len5 {p[tol]['runs_of_5']:3d} vs {q[tol]['runs_of_5']:3d}"
              f"   len7 {p[tol]['runs_of_7']:3d} vs {q[tol]['runs_of_7']:3d}")
    res["_meta"] = {"tap": args.tap, "timestep": args.timestep,
                    "coupled": bool(args.couple), "margin": args.margin,
                    "pct": args.pct,
                    "hypothesis": "all eigenvalue multiplicities odd (2l+1)",
                    "note": "lab-frame voxels; local-frame caches would false-negative"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
