"""Cache per-residue density targets for the alignment experiment.

Target = CryoFM2 `up_blocks[0]` activation at the centre of a 64^3 box cut in
each residue's N-CA-C backbone frame. Chosen because the per-residue ladder
measured it best of the CryoFM constructions: pose consistency 0.826 under
generic SO(3) with independent re-simulation, pose noise 0.151 of total variance,
and 0.447 of its variance tracks sequence at homolog resolution.

Correspondence is ASSUMED here -- the frame comes from the fitted model. That is
legitimate for defining a training TARGET (the sequence model never sees
coordinates at inference), and with fold-then-fit it is also available for an
unknown map whenever the sequence is known.

Writes one .npz per chain: feats [n_obs, C] aligned index-for-index with the
OBSERVED residue list from chain_backbone(), plus the observed sequence so the
ESM-C side can be checked against it rather than trusted.
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
    ap.add_argument("--outdir", type=Path, default=Path("data/density_targets"))
    ap.add_argument("--tap", default="up_blocks[0]")
    ap.add_argument("--taps", nargs="*", default=None,
                    help="extract SEVERAL taps in one pass. The hooks all fire on the "
                         "same forward, so extra taps are nearly free -- and the "
                         "vector-regression objective wants up_blocks[0] while a "
                         "RELATIONAL objective wants mid_block/down_blocks[2], so "
                         "extracting both at once saves a second 3h40m run. "
                         "Overrides --tap; each tap is stored under its own key.")
    ap.add_argument("--d-min", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=PATCH // 2 * MODEL_VOXEL_SIZE)
    ap.add_argument("--timestep", type=int, default=10)
    ap.add_argument("--couple", action="store_true",
                    help="tie input noise to the timestep (noise_level = t/1000), the TRAINED\n                          pairing. Default off = clean input, which is what every cached\n                          target in data/density_targets was built with.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--box-chunk", type=int, default=16,
                    help="local boxes resampled at once. The grid tensor is "
                         "chunk x 64^3 x 3 in float64 = ~400 MB at chunk 64, "
                         "which drove the allocator to <3 MB free on an 80 GB "
                         "card; 16 keeps peak ~100 MB.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.chains)))
    if args.limit:
        rows = rows[: args.limit]
    args.outdir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    taps = tuple(args.taps) if args.taps else (args.tap,)
    tap = CryoFM2Tap(args.ckpt, taps=taps, device=dev, batch_size=args.batch_size)
    print(f"{len(rows)} chains | taps={list(taps)} | t={args.timestep} "
          f"| couple={args.couple} | device={dev}")

    done = skipped = 0
    for i, r in enumerate(rows):
        out = args.outdir / f"{r['key']}.npz"
        if out.exists():
            done += 1
            continue
        try:
            seq, ca, fr = chain_backbone(r["pdb"], r["chain"])
            if seq != r["seq"]:
                raise ValueError("observed sequence differs from the chain list; "
                                 "the two sides would be misaligned")
            vol, origin, spacing = simulate(
                r["pdb"], d_min=args.d_min, voxel=MODEL_VOXEL_SIZE,
                chain=r["chain"], margin=args.margin)
            coords = (ca - np.asarray(origin)[::-1][None]) / np.asarray(spacing)[None]
            shape = np.array(vol.shape)
            keep = np.all((coords >= PATCH // 2)
                          & (coords < shape[None] - PATCH // 2), axis=1)
            if keep.sum() < 20:
                raise ValueError(f"only {int(keep.sum())} residues have box clearance")
            norm = preprocess(vol, MODEL_VOXEL_SIZE)
            boxes = extract_local_boxes(torch.from_numpy(norm), coords[keep],
                                        fr[keep], device=dev, chunk=args.box_chunk)
            allf = centre_features(tap, boxes, args.timestep, args.batch_size,
                                    noise_level=(args.timestep / 1000.0)
                                    if args.couple else None)
            feats = allf[taps[0]]
            # Free before the next chain: boxes is [n_res, 1, 64, 64, 64] on GPU
            # and the hook buffer holds a batch of activations. Without this the
            # allocator walks up to OOM over a long run.
            del boxes
            tap._buf.clear()
            if dev == "cuda":
                torch.cuda.empty_cache()
            # `keep` is carried, not applied: downstream must know WHICH observed
            # residues have targets so it can index ESM-C identically.
            # `feats` keeps the original key so existing consumers load
            # unchanged; a multi-tap run adds one `feats::<tap>` key beside it.
            payload = {"feats": feats.astype(np.float32), "keep": keep, "seq": seq}
            for t_ in taps:
                payload[f"feats::{t_}"] = allf[t_].astype(np.float32)
            np.savez_compressed(out, **payload)
            done += 1
        except Exception as exc:
            skipped += 1
            print(f"  [{i+1}/{len(rows)}] SKIP {r['key']} {type(exc).__name__}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] {done} done, {skipped} skipped")
    print(f"done: {done} cached, {skipped} skipped -> {args.outdir}")


if __name__ == "__main__":
    main()
