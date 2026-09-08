"""Isolate the patch-stitching effect: aligned vs unaligned, same map, same pass.

A review agent measured the alignment fix as worth +0.24 per-residue pose
consistency. Re-running the Phase 0a gate on this repo after the fix moved it by
<0.01 (25 maps, 8 rotations, 380k residues). One of the two is wrong about the
magnitude, and the pose metric is too indirect to say which -- it convolves the
stitching effect with everything else in the measurement.

This measures the stitching effect and nothing else:

  1. How many residues actually LAND in a misregistered region? The bug affects
     the last patch of each axis, so the answer depends on where a map's atoms
     sit relative to a 64-voxel lattice -- it is a property of the data, not just
     of the code. If it is small, both results can be right.
  2. How much does a feature CHANGE when the misregistration is removed? Cosine
     between aligned and unaligned features at the same Ca, split by whether the
     residue is in an affected region. Unaffected residues are the internal
     control: they must score ~1.0, or the two arms differ for some other reason
     and the comparison is void.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from teachers.cryofm_tap import (
    MODEL_VOXEL_SIZE,
    PATCH,
    TAP_STRIDES,
    CryoFM2Tap,
    preprocess,
    sample_at,
)
from probes.stability import load_ca, load_map


def affected_mask(shape, coords, stride) -> np.ndarray:
    """True where a residue sits in a patch whose origin was misregistered.

    Mirrors GridPatches3D: starts step by 64, plus a final `n - 64` per axis if
    that is off the grid. A start is misregistered iff it is not a multiple of
    the tap stride.
    """
    bad_spans = []
    for n in shape:
        starts = list(range(0, max(n - PATCH + 1, 1), PATCH))
        if not starts:
            starts = [0]
        if starts[-1] != n - PATCH and n >= PATCH:
            starts.append(n - PATCH)
        bad_spans.append([(s, s + PATCH) for s in starts if s % stride != 0])
    out = np.zeros(len(coords), dtype=bool)
    for ax in range(3):
        for lo, hi in bad_spans[ax]:
            out |= (coords[:, ax] >= lo) & (coords[:, ax] < hi)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--timestep", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--taps", nargs="*",
                    default=["mid_block", "up_blocks[0]", "up_blocks[1]"])
    ap.add_argument("--out", type=Path, default=Path("results/patch_alignment_ab.json"))
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tap = CryoFM2Tap(args.ckpt, taps=tuple(args.taps), device=dev,
                     batch_size=args.batch_size)
    rows = list(csv.DictReader(open(args.manifest)))[: args.limit]

    acc = {t: {"aff": [], "un": [], "frac": []} for t in args.taps}
    for i, row in enumerate(rows):
        try:
            vol, vs, origin_A = load_map(row["map_path"])
            ca = load_ca(row["cif_path"])
            norm = preprocess(vol, vs)
            # Same convention as probes/stability.py: preprocess() has already
            # resampled to MODEL_VOXEL_SIZE, so divide by that, not by `vs`.
            coords = (ca - origin_A[None]) / MODEL_VOXEL_SIZE
            shape = np.array(norm.shape)
            keep = np.all((coords >= 1) & (coords < shape[None] - 2), axis=1)
            coords = coords[keep]
            if len(coords) < 50:
                continue
            # Same correspondence assertion -- without it an origin error looks
            # like a null result.
            idx = np.rint(coords).astype(int)
            contrast = float(norm[idx[:, 0], idx[:, 1], idx[:, 2]].mean() - norm.mean())
            if contrast < 0.5:
                print(f"[{i+1}/{len(rows)}] {row['emdb_id']} SKIP contrast {contrast:+.2f}")
                continue

            fa = tap.feature_volumes(norm, timestep=args.timestep, align_patches=True)
            fu = tap.feature_volumes(norm, timestep=args.timestep, align_patches=False)
            for t in args.taps:
                A, U = sample_at(fa[t], coords), sample_at(fu[t], coords)
                num = (A * U).sum(1)
                den = np.linalg.norm(A, axis=1) * np.linalg.norm(U, axis=1)
                cos = num / np.maximum(den, 1e-12)
                # Mask is computed on the UNALIGNED grid -- that is the one whose
                # patch starts were off-lattice.
                m = affected_mask(norm.shape, coords, TAP_STRIDES[t])
                acc[t]["aff"].extend(cos[m].tolist())
                acc[t]["un"].extend(cos[~m].tolist())
                acc[t]["frac"].append(float(m.mean()))
            print(f"[{i+1}/{len(rows)}] {row['emdb_id']} {len(coords)} residues "
                  f"grid {tuple(norm.shape)}")
        except Exception as exc:
            print(f"[{i+1}/{len(rows)}] {row['emdb_id']} FAILED "
                  f"{type(exc).__name__}: {exc}")

    out = {}
    for t in args.taps:
        d = acc[t]
        out[t] = {
            "frac_residues_affected": float(np.mean(d["frac"])) if d["frac"] else None,
            "cos_affected": float(np.median(d["aff"])) if d["aff"] else None,
            "cos_unaffected": float(np.median(d["un"])) if d["un"] else None,
            "n_affected": len(d["aff"]),
            "n_unaffected": len(d["un"]),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 76)
    print(f"{'tap':<16}{'%residues':>11}{'cos affected':>15}{'cos control':>14}")
    print("-" * 76)
    for t, r in out.items():
        f = "n/a" if r["frac_residues_affected"] is None else f"{r['frac_residues_affected']:.1%}"
        ca_ = "n/a" if r["cos_affected"] is None else f"{r['cos_affected']:.4f}"
        cu = "n/a" if r["cos_unaffected"] is None else f"{r['cos_unaffected']:.4f}"
        print(f"{t:<16}{f:>11}{ca_:>15}{cu:>14}")
    print("=" * 76)
    print("'cos control' is unaffected residues: MUST be ~1.0, else the two arms")
    print("differ for some reason other than stitching and this test is void.")
    print("'cos affected' near 1.0 => the misregistration barely changed the")
    print("feature, so the bug was real but cosmetic on this data.")


if __name__ == "__main__":
    main()
