"""All-intermediate-layer centre features from the CleanDIFT student.

The stored o5 parts hold only three taps (`mid_block`, `up_blocks[0]`,
`up_blocks[1]`), which is enough for the tap comparison in the main o7 analysis
but not for a LAYER SWEEP. This re-forwards the same residues and captures every
named block output of the CryoFM2 UNet:

    conv_in · down_blocks[0..3] · mid_block · up_blocks[0..3]

Two things make the output drop-in compatible with `o7_spatial_variability.py`:
the residue selection is the SAME rng replay used everywhere else (so
`data/o7_coords.npz` applies unchanged), and each part carries `aa`/`ss`/`split`/
`cluster` under the same names, so `Corpus` reads this directory with no changes.

The selection is VERIFIED, not assumed: every chain's recovered `aa` and `ss` must
match the already-extracted `data/o5_multitap_parts` entry exactly, or the run
aborts. Pairing the wrong coordinates to the right features is the failure mode
this project keeps catching late.

`rand_student` (identical architecture, weights never loaded) is extracted
alongside by default. A layer sweep without it invites the exact misreading this
analysis exists to prevent: a box-based feature is spatially smoothed BY
CONSTRUCTION, and how much so varies strongly by layer.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

ALL_TAPS = ("conv_in", "down_blocks[0]", "down_blocks[1]", "down_blocks[2]",
            "down_blocks[3]", "mid_block", "up_blocks[0]", "up_blocks[1]",
            "up_blocks[2]", "up_blocks[3]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--vol-dir", type=Path, default=Path("data/cleandift_vols"))
    ap.add_argument("--ref-dir", type=Path, default=Path("data/o5_multitap_parts"),
                    help="existing parts used to VERIFY the residue replay")
    ap.add_argument("--feat-dir", type=Path, default=Path("data/o7_alltap_parts"))
    ap.add_argument("--student", default="data/cleandift_runs/distill_t1000_s0/best.pt")
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--max-chains", type=int, default=10 ** 9,
                    help="stop after this many successfully extracted chains")
    ap.add_argument("--per-chain", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--box-chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-rand", action="store_true")
    args = ap.parse_args()

    from probes.o5_arms_multitap import centre2_multi
    from probes.o5_boxes import chain_boxes
    from teachers.cryofm_tap import CryoFM2Tap

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.feat_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.chains)))[: args.limit]
    rows.sort(key=lambda r: r["emd"])

    arms = {"student": args.student}
    if not args.no_rand:
        arms["rand_student"] = None
    keys = [f"{a}@{t}" for a in arms for t in ALL_TAPS]
    print(f"{len(rows)} candidate chains | taps {list(ALL_TAPS)} | arms "
          f"{list(arms)} | {dev}", flush=True)

    taps = {}
    for nm, ck in arms.items():
        # No `stop_after`: the sweep needs the WHOLE forward, including up_blocks[3].
        taps[nm] = CryoFM2Tap(args.ckpt, taps=ALL_TAPS, device=dev,
                              batch_size=args.batch_size,
                              random_weights=(ck is None),
                              student_ckpt=ck)

    rng = np.random.default_rng(args.seed)
    done = skipped = bad = 0
    for r in rows:
        if done >= args.max_chains:
            break
        part = args.feat_dir / f"{r['key']}.npz"
        ref = args.ref_dir / f"{r['key']}.npz"
        if part.exists():
            z = np.load(part, allow_pickle=True)
            if not set(keys) - set(z.files):
                done += 1
                continue
            part.unlink()
        try:
            boxes, aa, ss, _ = chain_boxes(r, args.vol_dir, args.per_chain, rng,
                                           dev, args.box_chunk)
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"  SKIP {r['key']} {type(exc).__name__}: {exc}", flush=True)
            continue
        # Verify the replay against the already-extracted parts before spending a
        # forward on it -- a silent mis-pairing would be invisible downstream.
        if ref.exists():
            zr = np.load(ref, allow_pickle=True)
            if not (np.array_equal(zr["aa"], aa) and np.array_equal(zr["ss"], ss)):
                bad += 1
                print(f"  MISMATCH {r['key']}: replay disagrees with {ref}", flush=True)
                continue
        out = {}
        for nm in arms:
            for k, v in centre2_multi(taps[nm], boxes, 0, args.batch_size,
                                      ALL_TAPS).items():
                out[f"{nm}@{k}"] = v
        del boxes
        if dev == "cuda":
            torch.cuda.empty_cache()
        np.savez(part, aa=aa, ss=ss, split=np.array([r["split"]] * len(aa)),
                 cluster=np.array([r["cluster"]] * len(aa)), **out)
        done += 1
        if done % 25 == 0:
            print(f"  [{done}] ok, {skipped} skipped, {bad} mismatched", flush=True)

    print(f"chains: {done} ok, {skipped} skipped, {bad} MISMATCHED", flush=True)
    if bad:
        raise SystemExit(f"FAILED: {bad} chains failed aa/ss verification.")
    if done == 0:
        raise SystemExit("FAILED: 0 chains extracted.")


if __name__ == "__main__":
    main()
