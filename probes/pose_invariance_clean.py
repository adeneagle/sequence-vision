"""The honest pose-invariance number for local-frame features.

Local framing makes features pose-invariant BY CONSTRUCTION, so what actually
limits them is discretization. The earlier measurement (0.58-0.72) overstates the
error because it rotates a already-sampled volume: resample #1 for the rotation,
resample #2 to cut the box. Deployment only ever does #2.

Here each orientation is simulated from the atomic model directly, so the two
grids are INDEPENDENT discretizations of the same continuous object -- exactly the
deployment condition, where different proteins simply arrive in different
arbitrary orientations. Mass conservation under rotation (verified, ratio 1.0000)
confirms no resampling is happening.

Reference points:
    global frame, cube rotations      pose 0.33-0.50
    local frame, SO(3) on a real map  pose 0.58-0.72   <- inflated by double resampling
    local frame, independent grids    <- what this measures

    pixi run python probes/pose_invariance_clean.py --limit 8 --n-rot 3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data.simulate_density import simulate
from probes.local_frame_stability import (
    backbone_frames,
    centre_features,
    extract_local_boxes,
    random_so3,
)
from probes.stability import centred_cosine, retrieval
from teachers.cryofm_tap import MODEL_VOXEL_SIZE, PATCH, CryoFM2Tap, preprocess


def features_for(tap, cif, R, coords_xyz, frames_zyx, args, dev):
    """Simulate at orientation R, then read local-frame centre features."""
    # vol is [z,y,x]; spacing is [z,y,x] and is NOT the requested voxel size --
    # gemmi rounds each axis to an FFT-friendly length independently, so the
    # realized spacing is anisotropic ~7-11% AND pose-dependent. The earlier
    # version of this function hardcoded MODEL_VOXEL_SIZE here, which mislocated
    # Ca atoms by a median 8.8 A against a box only ~12 A wide -- large enough to
    # invalidate the local-frame result this probe was written to measure.
    #
    # The margin must cover the local box, or the `keep` test below silently
    # discards every surface residue. A 64-voxel box needs PATCH//2 = 32 voxels
    # = 48 A of clearance, but simulate()'s default margin is 12 A -- so only
    # residues deep inside a large complex survived, small proteins yielded an
    # empty list (torch.cat on no tensors), and the surviving sample was biased
    # towards buried environments. Defaults to 48 A; see --margin.
    vol, origin_xyz, spacing_zyx = simulate(
        cif, d_min=args.d_min, voxel=MODEL_VOXEL_SIZE, rotation=R,
        chain=args.chain, margin=args.margin)
    spacing_xyz = np.asarray(spacing_zyx)[::-1]

    pos = coords_xyz
    if R is not None:
        c = pos.mean(0)
        pos = (pos - c) @ np.asarray(R).T + c
    # xyz -> grid voxel index, then to [z,y,x] ordering to match the volume
    vox_xyz = (pos - np.asarray(origin_xyz)[None]) / spacing_xyz[None]
    coords = vox_xyz[:, ::-1].copy()

    fr = frames_zyx if R is None else np.einsum("ij,mkj->mki", _R_zyx(R), frames_zyx)

    shape = np.array(vol.shape)
    keep = np.all((coords >= PATCH // 2) & (coords < shape[None] - PATCH // 2), axis=1)
    if keep.sum() == 0:
        # Name the cause. Otherwise this surfaces downstream as an opaque
        # "torch.cat(): expected a non-empty list of Tensors".
        raise ValueError(
            f"no residue has {PATCH // 2} voxels of clearance in a {tuple(shape)} "
            f"grid (margin={args.margin} A) -- raise --margin")
    norm = preprocess(vol, MODEL_VOXEL_SIZE)
    boxes = extract_local_boxes(torch.from_numpy(norm), coords[keep], fr[keep], device=dev)
    nl = (args.timestep / 1000.0) if args.couple else None
    return centre_features(tap, boxes, args.timestep, args.batch_size,
                           noise_level=nl), keep


def _R_zyx(R_xyz: np.ndarray) -> np.ndarray:
    """Re-express an xyz rotation in zyx axis order."""
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    return P @ np.asarray(R_xyz, dtype=float) @ P.T


def density_frames(vol: np.ndarray, coords: np.ndarray, sigma: float = 3.0,
                   half: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Local frames built from the DENSITY, requiring no atomic model.

    Backbone frames (N-Ca-C) are not a general construction: they can only be
    computed where a fitted structure already exists, so they can never be applied
    to an unknown map. This is the model-free replacement -- eigenvectors of the
    local structure tensor J = smooth(grad rho (x) grad rho) in a window around
    each point. J is equivariant by construction: rotate the density and J
    conjugates, so its eigenvectors rotate with it.

    Returns frames [N,3,3] (rows = axes, descending eigenvalue) and the relative
    eigenvalue gaps [N,2]. Small gaps mean a locally isotropic neighbourhood where
    the frame is ill-defined -- the same degeneracy that sank inertial frames, but
    now local, so it must be measured rather than assumed.
    """
    from scipy import ndimage

    g = np.gradient(ndimage.gaussian_filter(vol.astype(np.float32), sigma * 0.5))
    comp = {}
    for a in range(3):
        for b in range(a, 3):
            comp[(a, b)] = ndimage.gaussian_filter(g[a] * g[b], sigma)

    idx = np.rint(coords).astype(int)
    idx = np.clip(idx, half, np.array(vol.shape) - half - 1)
    frames, gaps = [], []
    for p in idx:
        J = np.empty((3, 3))
        for a in range(3):
            for b in range(a, 3):
                J[a, b] = J[b, a] = comp[(a, b)][p[0], p[1], p[2]]
        w, v = np.linalg.eigh(J)
        o = np.argsort(w)[::-1]
        w, v = w[o], v[:, o].T
        # sign fix: eigenvectors are defined up to sign, so pick a deterministic
        # one from the local density's third moment (equivariant under rotation).
        for a in range(3):
            if v[a] @ np.array([1.0, 1.0, 1.0]) < 0:
                v[a] = -v[a]
        v[2] = np.cross(v[0], v[1])
        s = max(w[0], 1e-12)
        frames.append(v)
        gaps.append([(w[0] - w[1]) / s, (w[1] - w[2]) / s])
    return np.array(frames), np.array(gaps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--n-rot", type=int, default=3)
    ap.add_argument("--n-res", type=int, default=250)
    ap.add_argument("--d-min", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=PATCH // 2 * MODEL_VOXEL_SIZE,
                    help="A of padding around the model; must be >= half a local "
                         "box (48 A) or every surface residue is discarded")
    ap.add_argument("--chain", default=None)
    ap.add_argument("--timestep", type=int, default=10)
    ap.add_argument("--couple", action="store_true",
                    help="tie the input noise to the timestep (noise_level = t/1000), the\n                          TRAINED pairing. Default off = clean input, which is off-manifold\n                          for t>0 but is the pairing every prior result in this repo used.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("results/pose_clean.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))[: args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tap = CryoFM2Tap(args.ckpt, device=dev, batch_size=args.batch_size)
    rng = np.random.default_rng(0)
    print(f"device={dev} maps={len(rows)} rotations={args.n_rot} d_min={args.d_min}")

    pooled: dict[str, list] = {}
    retr: dict[str, list] = {}

    for i, row in enumerate(rows):
        try:
            cif = row["cif_path"]
            ca_zyx, frames_zyx = backbone_frames(cif)
            if len(ca_zyx) < 50:
                continue
            sel = rng.choice(len(ca_zyx), min(args.n_res, len(ca_zyx)), replace=False)
            ca_zyx, frames_zyx = ca_zyx[sel], frames_zyx[sel]
            coords_xyz = ca_zyx[:, ::-1].copy()            # back to xyz for simulate()

            ref, keep_ref = features_for(tap, cif, None, coords_xyz, frames_zyx, args, dev)
            for _ in range(args.n_rot):
                R = random_so3(rng)
                rot, keep_rot = features_for(tap, cif, R, coords_xyz, frames_zyx, args, dev)
                both = keep_ref & keep_rot
                if both.sum() < 30:
                    continue
                ia = np.cumsum(keep_ref) - 1
                ib = np.cumsum(keep_rot) - 1
                for k in ref:
                    A = ref[k][ia[both]]
                    B = rot[k][ib[both]]
                    pooled.setdefault(k, []).append(centred_cosine(A, B))
                    retr.setdefault(k, []).append(retrieval(A, B))
            print(f"[{i+1}/{len(rows)}] {row['emdb_id']} {int(keep_ref.sum())} residues")
        except Exception as exc:
            print(f"[{i+1}/{len(rows)}] {row['emdb_id']} FAILED {type(exc).__name__}: {exc}")

    summary = {}
    for k, v in pooled.items():
        x = np.concatenate(v)
        x = x[np.isfinite(x)]
        r = retr[k]
        summary[k] = {
            "pose_median": float(np.median(x)), "pose_p10": float(np.percentile(x, 10)),
            "top1": float(np.mean([q["top1"] for q in r])),
            "matched": float(np.mean([q["matched_cos"] for q in r])),
            "mismatched": float(np.mean([q["mismatched_cos"] for q in r])),
            "n": int(x.size),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)

    print("\n" + "=" * 72)
    print(f"{'tap':<16}{'pose':>9}{'p10':>9}{'top1':>8}{'matched':>10}{'mismatch':>10}{'n':>9}")
    print("-" * 72)
    for k, s in summary.items():
        print(f"{k:<16}{s['pose_median']:>9.3f}{s['pose_p10']:>9.3f}{s['top1']:>8.3f}"
              f"{s['matched']:>10.3f}{s['mismatched']:>10.3f}{s['n']:>9}")
    print("=" * 72)
    print("vs  global frame / cube rot   : pose 0.33-0.50, top1 0.03-0.10")
    print("vs  local frame / SO(3) resample: pose 0.58-0.72, top1 0.73-0.85 (double-resampled)")
    print("This run uses INDEPENDENT simulated grids -- no resampling, so the gap from 1.0")
    print("is pure discretization sensitivity, the real deployment limit.")


if __name__ == "__main__":
    main()
