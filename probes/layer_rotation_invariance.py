"""Rotation invariance of CryoFM2 features AS A FUNCTION OF LAYER.

Rotate the structure, re-simulate from scratch, recompute features, and ask how
much a residue's feature vector survives -- for every tap, not just the three we
happen to use.

Two framings, run together because they answer different questions:

  global  Rotate the molecule, forward the whole map, sample at the residue's
          new coordinates. This is the network's RAW non-equivariance: the thing
          `RotCube24` augmentation was supposed to buy and (measured earlier)
          does not, since augmentation makes the denoising TASK equivariant
          without making any internal activation invariant.
  local   Cut a 64^3 box in the residue's own N-CA-C frame first. Pose-invariant
          BY CONSTRUCTION, so whatever is left is pure discretization error --
          the floor for any frame-based construction.

Rotations are generic SO(3) with INDEPENDENT re-simulation, never cube
rotations: the octahedral group is exactly CryoFM2's augmentation group and
flatters it roughly 2x at every granularity measured.

Every cosine is reported against its mismatched floor (feature of residue i vs
residue j != i). An absolute similarity without that floor is uninterpretable --
raw cosines on high-dim CNN activations sit near 0.99 for any pair.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data.simulate_density import model_centre, simulate
from probes.homolog_diagnostic_residue import chain_backbone
from probes.local_frame_stability import (
    centre_features,
    extract_local_boxes,
    random_so3,
)
from probes.pose_invariance_clean import _R_zyx
from probes.stability import centred_cosine
from teachers.cryofm_tap import (
    MODEL_VOXEL_SIZE,
    PATCH,
    TAP_STRIDES,
    CryoFM2Tap,
    preprocess,
    sample_at,
)

ALL_TAPS = ("conv_in", "down_blocks[0]", "down_blocks[1]", "down_blocks[2]",
            "down_blocks[3]", "mid_block", "up_blocks[0]", "up_blocks[1]",
            "up_blocks[2]", "up_blocks[3]")


def _sim(pdb, chain, R, margin, d_min):
    vol, origin, spacing = simulate(pdb, d_min=d_min, voxel=MODEL_VOXEL_SIZE,
                                    rotation=R, chain=chain, margin=margin)
    assert np.allclose(spacing, MODEL_VOXEL_SIZE), spacing
    return vol, np.asarray(origin), np.asarray(spacing)


def _coords(ca_zyx, frames, pdb, chain, R, origin, spacing):
    """Rotate Ca + frames about simulate()'s own centre, then to voxel indices."""
    ca, fr = ca_zyx, frames
    if R is not None:
        ctr = model_centre(pdb, chain)[::-1]                 # xyz -> zyx
        Rz = _R_zyx(R)
        ca = (ca - ctr) @ Rz.T + ctr
        fr = np.einsum("ij,mkj->mki", Rz, fr)
    return (ca - origin[::-1][None]) / spacing[None], fr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--limit", type=int, default=12, help="structures")
    ap.add_argument("--n-rot", type=int, default=2)
    ap.add_argument("--n-res", type=int, default=60, help="residues per structure")
    ap.add_argument("--d-min", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--timestep", type=int, default=10,
                    help="flow-matching operating point. The default of 10 is 1%% noise, which\n                          is BELOW the bottom of SD's schedule (~3%% at t=0) and ~40x under the\n                          diffusion-features optimum (FM t ~ 420). Every result logged from this\n                          probe before 2026-08-26 is a t=10 slice; sweep it.")
    ap.add_argument("--box-chunk", type=int, default=16)
    ap.add_argument("--no-local", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("results/layer_rotation_invariance.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.chains)))
    rows = [r for r in rows if 80 <= int(r["n_obs"]) <= 400][: args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tap = CryoFM2Tap(args.ckpt, taps=ALL_TAPS, device=dev, batch_size=args.batch_size)
    rng = np.random.default_rng(0)
    margin = PATCH // 2 * MODEL_VOXEL_SIZE      # local boxes need 48 A clearance
    print(f"{len(rows)} structures x {args.n_rot} SO(3) rotations, "
          f"{args.n_res} residues each | device={dev}")

    acc = {f"{m}:{t}": {"match": [], "mis": []}
           for m in ("global", "local") for t in ALL_TAPS}

    for i, r in enumerate(rows):
        try:
            seq, ca, fr = chain_backbone(r["pdb"], r["chain"])
            volA, oA, spA = _sim(r["pdb"], r["chain"], None, margin, args.d_min)
            cA, frA = _coords(ca, fr, r["pdb"], r["chain"], None, oA, spA)
            shape = np.array(volA.shape)
            keep = np.all((cA >= PATCH // 2) & (cA < shape[None] - PATCH // 2), axis=1)
            idx = np.where(keep)[0]
            if len(idx) < 20:
                print(f"  [{i+1}] SKIP {r['key']}: only {len(idx)} residues with clearance")
                continue
            if len(idx) > args.n_res:
                idx = rng.choice(idx, args.n_res, replace=False)
            normA = preprocess(volA, MODEL_VOXEL_SIZE)

            gA = {t: sample_at(fv, cA[idx]) for t, fv in
                  tap.feature_volumes(normA, timestep=args.timestep).items()}
            lA = None
            if not args.no_local:
                bx = extract_local_boxes(torch.from_numpy(normA), cA[idx], frA[idx],
                                         device=dev, chunk=args.box_chunk)
                lA = centre_features(tap, bx, args.timestep, args.batch_size)
                del bx
                tap._buf.clear()
                if dev == "cuda":
                    torch.cuda.empty_cache()

            for _ in range(args.n_rot):
                R = random_so3(rng)
                volB, oB, spB = _sim(r["pdb"], r["chain"], R, margin, args.d_min)
                cB, frB = _coords(ca, fr, r["pdb"], r["chain"], R, oB, spB)
                shB = np.array(volB.shape)
                if not np.all(np.all((cB[idx] >= PATCH // 2)
                                     & (cB[idx] < shB[None] - PATCH // 2), axis=1)):
                    continue
                normB = preprocess(volB, MODEL_VOXEL_SIZE)

                gB = {t: sample_at(fv, cB[idx]) for t, fv in
                      tap.feature_volumes(normB, timestep=args.timestep).items()}
                for t in ALL_TAPS:
                    acc[f"global:{t}"]["match"].extend(centred_cosine(gA[t], gB[t]).tolist())
                    perm = rng.permutation(len(idx))
                    acc[f"global:{t}"]["mis"].extend(
                        centred_cosine(gA[t], gB[t][perm]).tolist())

                if not args.no_local:
                    bx = extract_local_boxes(torch.from_numpy(normB), cB[idx], frB[idx],
                                             device=dev, chunk=args.box_chunk)
                    lB = centre_features(tap, bx, args.timestep, args.batch_size)
                    del bx
                    tap._buf.clear()
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                    for t in ALL_TAPS:
                        acc[f"local:{t}"]["match"].extend(
                            centred_cosine(lA[t], lB[t]).tolist())
                        perm = rng.permutation(len(idx))
                        acc[f"local:{t}"]["mis"].extend(
                            centred_cosine(lA[t], lB[t][perm]).tolist())
            print(f"  [{i+1}/{len(rows)}] {r['key']} {len(idx)} residues")
        except Exception as exc:
            print(f"  [{i+1}/{len(rows)}] SKIP {r['key']} {type(exc).__name__}: {exc}")

    res = {}
    for k, d in acc.items():
        if not d["match"]:
            continue
        m, f = float(np.nanmedian(d["match"])), float(np.nanmedian(d["mis"]))
        res[k] = {"matched": m, "mismatched": f, "separation": m - f,
                  "n": len(d["match"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 78)
    print(f"{'tap':<17}{'A/token':>8}{'global':>10}{'floor':>9}"
          f"{'local':>10}{'floor':>9}")
    print("-" * 78)
    for t in ALL_TAPS:
        g, l = res.get(f"global:{t}"), res.get(f"local:{t}")
        aa = TAP_STRIDES[t] * MODEL_VOXEL_SIZE
        gs = f"{g['matched']:+.3f}" if g else "n/a"
        gf = f"{g['mismatched']:+.3f}" if g else ""
        ls = f"{l['matched']:+.3f}" if l else "n/a"
        lf = f"{l['mismatched']:+.3f}" if l else ""
        print(f"{t:<17}{aa:>8.1f}{gs:>10}{gf:>9}{ls:>10}{lf:>9}")
    print("=" * 78)
    print("global = rotate the molecule, forward the whole map, sample at the new")
    print("         coordinates -> the network's RAW non-equivariance.")
    print("local  = box cut in the residue's backbone frame -> pose-invariant by")
    print("         construction, so the residual is pure discretization.")
    print("Generic SO(3) with independent re-simulation, never cube rotations.")


if __name__ == "__main__":
    main()
