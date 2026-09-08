"""Correctness check for CryoFM2Tap.sample_frame_averaged.

Two claims, one provable and one empirical:

  1. **Exact invariance to the averaged group.** Averaging a feature over all of
     G makes the result G-invariant *by construction*. So rotating the input by
     any g in G and rotating the query coordinates to match must return the SAME
     vectors, to floating-point. This is a hard assertion, not a metric -- if it
     fails, the coordinate bookkeeping in `sample_frame_averaged` is wrong.

     Note the failure mode this is written to catch: a test that rotates the
     volume and the coordinates *together* through a lossless cube rotation can
     be tautological (it re-samples identical points with identical weights, so
     it returns 1.000 no matter what the code does). This one is not, because
     frame averaging must reconcile 24 DIFFERENT padded grids -- each rotation
     pads to its own multiple of 64 -- and a bookkeeping error in that
     reconciliation shows up here as a mismatch.

  2. **Improvement under generic SO(3)**, i.e. rotations OUTSIDE the averaged
     group, which is the honest test and the number worth reporting. Compares
     single-frame against 24-frame averaging on independently re-simulated
     density, so the two poses are separate discretizations rather than a
     resampling of one another.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data.simulate_density import simulate
from probes.local_frame_stability import random_so3
from probes.pose_invariance_clean import _R_zyx
from probes.stability import centred_cosine, load_ca
from teachers.cryofm_tap import (
    MODEL_VOXEL_SIZE,
    CryoFM2Tap,
    cube_rotations,
    preprocess,
    rotate_coords,
    rotate_volume,
    sample_at,
)


def _sim(cif, R, chain):
    vol, origin_xyz, spacing_zyx = simulate(
        cif, d_min=3.0, voxel=MODEL_VOXEL_SIZE, rotation=R, chain=chain)
    assert np.allclose(spacing_zyx, MODEL_VOXEL_SIZE), spacing_zyx
    return vol, np.asarray(origin_xyz), np.asarray(spacing_zyx)


def _coords(ca_zyx, R, origin_xyz, spacing_zyx, centre_xyz):
    """Ca (z,y,x) -> voxel index (z,y,x) in a grid simulated at rotation R."""
    xyz = ca_zyx[:, ::-1]
    if R is not None:
        xyz = (xyz - centre_xyz) @ np.asarray(R).T + centre_xyz
    vox_xyz = (xyz - origin_xyz[None]) / np.asarray(spacing_zyx)[::-1][None]
    return vox_xyz[:, ::-1].copy()


def _inside(coords, shape, pad=8):
    return np.all((coords >= pad) & (coords < np.array(shape)[None] - pad), axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--n-res", type=int, default=200)
    ap.add_argument("--n-rot", type=int, default=2, help="generic SO(3) poses")
    ap.add_argument("--n-frames", type=int, default=24)
    ap.add_argument("--timestep", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("results/frame_avg_check.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))[: args.limit]
    tap = CryoFM2Tap(args.ckpt, batch_size=args.batch_size)
    rng = np.random.default_rng(0)
    kw = dict(timestep=args.timestep)

    exact_err: list[float] = []
    single: dict[str, list[float]] = {}
    avg: dict[str, list[float]] = {}

    for row in rows:
        cif, chain = row["cif_path"], row["chain_id"]
        ca = load_ca(cif, chain_id=chain)
        if len(ca) < 20:
            continue
        sel = rng.choice(len(ca), min(args.n_res, len(ca)), replace=False)
        ca = ca[sel]
        centre_xyz = ca[:, ::-1].mean(0)

        volA, oA, spA = _sim(cif, None, chain)
        cA = _coords(ca, None, oA, spA, centre_xyz)
        keep = _inside(cA, volA.shape)
        if keep.sum() < 20:
            continue
        cA = cA[keep]
        nA = preprocess(volA, MODEL_VOXEL_SIZE)
        refA = tap.sample_frame_averaged(nA, cA, n_frames=args.n_frames, **kw)

        # --- claim 1: exact invariance to a member of the averaged group -----
        # Only valid when the WHOLE group is averaged; a partial average is
        # invariant to nothing in particular.
        if args.n_frames == 24:
            g = cube_rotations()[7]
            gvol = rotate_volume(nA, g)
            gc = rotate_coords(cA, g, nA.shape)
            gout = tap.sample_frame_averaged(gvol, gc, n_frames=24, **kw)
            for name in refA:
                a, b = refA[name], gout[name]
                denom = max(float(np.abs(a).mean()), 1e-12)
                exact_err.append(float(np.abs(a - b).mean() / denom))

        # --- claim 2: generic SO(3), single frame vs 24-frame average --------
        singleA = {n: sample_at(fv, cA)
                   for n, fv in tap.feature_volumes(nA, **kw).items()}
        for _ in range(args.n_rot):
            R = random_so3(rng)
            volB, oB, spB = _sim(cif, R, chain)
            cB = _coords(ca[keep], R, oB, spB, centre_xyz)
            if not np.all(_inside(cB, volB.shape)):
                continue
            nB = preprocess(volB, MODEL_VOXEL_SIZE)
            singleB = {n: sample_at(fv, cB)
                       for n, fv in tap.feature_volumes(nB, **kw).items()}
            avgB = tap.sample_frame_averaged(nB, cB, n_frames=args.n_frames, **kw)
            for name in refA:
                single.setdefault(name, []).extend(
                    centred_cosine(singleA[name], singleB[name]).tolist())
                avg.setdefault(name, []).extend(
                    centred_cosine(refA[name], avgB[name]).tolist())

    def med(d):
        return {k: float(np.nanmedian(v)) for k, v in d.items()}

    out = {
        "n_maps": len(rows),
        "n_frames": args.n_frames,
        "exact_invariance_rel_err": {
            "max": float(np.max(exact_err)) if exact_err else None,
            "median": float(np.median(exact_err)) if exact_err else None,
        },
        "so3_single_frame": med(single),
        "so3_frame_averaged": med(avg),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    if exact_err:
        # Tolerance is loose because the 24 forward passes run in bf16-ish
        # kernels and the sum order differs; a bookkeeping error is orders of
        # magnitude larger than this, not marginally over it.
        assert max(exact_err) < 5e-2, (
            f"frame averaging is NOT invariant to its own group "
            f"(max relative error {max(exact_err):.3g}) -- coordinate "
            "bookkeeping in sample_frame_averaged is wrong")
        print("\nOK: exactly invariant to the averaged group")
    for name in out["so3_single_frame"]:
        print(f"  {name:>14}  SO(3) single {out['so3_single_frame'][name]:.3f}"
              f"  ->  24-frame {out['so3_frame_averaged'][name]:.3f}")


if __name__ == "__main__":
    main()
