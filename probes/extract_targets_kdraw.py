"""K-draw noise marginalisation: is the coupled arm's advantage recoverable?

The coupled/decoupled split left a puzzle. Coupled features are BETTER features
(more pose-stable at the peak, more residue-discriminative) yet WORSE regression
targets by 2.7x -- and the seqwin3 control pinned why: the coupled feature is
f(structure, eps) where eps is a noise realisation, so part of it is
unpredictable from sequence BY CONSTRUCTION, for any predictor.

That suggests the advantage is not lost, only unreachable by SINGLE-DRAW targets.
E_eps[f(x_t, t)] is deterministic in the structure AND on-manifold, so in
principle it has coupled quality with decoupled predictability -- a combination
neither single arm offers. This tests it directly, before anyone builds a
distillation to achieve the same thing.

Three targets, same chains, same t, same taps, one pass:
    decoupled   f(x_0, t)                 -- the current choice
    coupled1    f(x_t, t), one draw       -- the arm that lost 2.7x
    coupledK    mean_k f(x_t^(k), t)      -- the marginalised target

Written as a SEPARATE SCRIPT rather than a flag on extract_density_targets.py:
two 3h40m extractions are running against that file right now, and I corrupted
it once today. Not worth the risk for a variant experiment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from data.simulate_density import simulate
from probes.homolog_diagnostic_residue import chain_backbone
from probes.local_frame_stability import centre_features, extract_local_boxes
from teachers.cryofm_tap import MODEL_VOXEL_SIZE, PATCH, CryoFM2Tap, preprocess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--outroot", type=Path, default=Path("data/kdraw"))
    ap.add_argument("--tap", default="up_blocks[0]")
    ap.add_argument("--timestep", type=int, default=250,
                    help="250 is where coupled pose peaked, i.e. where the "
                         "coupled quality advantage is largest and therefore "
                         "where marginalisation has the most to recover")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--d-min", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=PATCH // 2 * MODEL_VOXEL_SIZE)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--box-chunk", type=int, default=16)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.chains)))[: args.limit]
    dirs = {n: args.outroot / n for n in ("decoupled", "coupled1", "coupledK")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tap = CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                     batch_size=args.batch_size)
    nl = args.timestep / 1000.0
    from collections import Counter
    print(f"{len(rows)} chains | t={args.timestep} (noise_level={nl}) | K={args.k} "
          f"| tap={args.tap} | {dev}", flush=True)
    print(f"splits: {dict(Counter(r['split'] for r in rows))}", flush=True)

    done = skipped = 0
    for i, r in enumerate(rows):
        if all((dirs[n] / f"{r['key']}.npz").exists() for n in dirs):
            done += 1
            continue
        try:
            seq, ca, fr = chain_backbone(r["pdb"], r["chain"])
            if seq != r["seq"]:
                raise ValueError("observed sequence differs from the chain list")
            vol, origin, spacing = simulate(
                r["pdb"], d_min=args.d_min, voxel=MODEL_VOXEL_SIZE,
                chain=r["chain"], margin=args.margin)
            coords = (ca - np.asarray(origin)[::-1][None]) / np.asarray(spacing)[None]
            shape = np.array(vol.shape)
            keep = np.all((coords >= PATCH // 2)
                          & (coords < shape[None] - PATCH // 2), axis=1)
            if keep.sum() < 20:
                raise ValueError(f"only {int(keep.sum())} residues have clearance")
            norm = preprocess(vol, MODEL_VOXEL_SIZE)
            boxes = extract_local_boxes(torch.from_numpy(norm), coords[keep],
                                        fr[keep], device=dev, chunk=args.box_chunk)

            dec = centre_features(tap, boxes, args.timestep,
                                  args.batch_size)[args.tap]
            # K INDEPENDENT draws: centre_features seeds its generator from
            # noise_seed, so distinct seeds give distinct eps. Draw 0 doubles as
            # the single-draw arm, so this costs K forwards, not K+1.
            draws = [centre_features(tap, boxes, args.timestep, args.batch_size,
                                     noise_level=nl, noise_seed=s)[args.tap]
                     for s in range(args.k)]
            out = {"decoupled": dec, "coupled1": draws[0],
                   "coupledK": np.mean(draws, axis=0)}
            for n, f in out.items():
                np.savez_compressed(dirs[n] / f"{r['key']}.npz",
                                    feats=f.astype(np.float32), keep=keep, seq=seq)
            del boxes, draws
            tap._buf.clear()
            if dev == "cuda":
                torch.cuda.empty_cache()
            done += 1
        except Exception as exc:
            skipped += 1
            print(f"  [{i+1}/{len(rows)}] SKIP {r['key']} {type(exc).__name__}: {exc}",
                  flush=True)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(rows)}] {done} done, {skipped} skipped", flush=True)
    print(f"done: {done} cached, {skipped} skipped -> {args.outroot}/"
          "{decoupled,coupled1,coupledK}")


if __name__ == "__main__":
    main()
