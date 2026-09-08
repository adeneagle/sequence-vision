"""ESM-C 600m per-residue embeddings for the alignment chain list.

Runs in the MAIN LAB ENV -- `esm` is not installed in this project's env and
adding it risks CryoFM's numpy<2.0 pin:

    /mnt/main0/projects/et-foundation-vision-model/aden/aden/.pixi/envs/default/bin/python3.10 \
        data/extract_esmc_chains.py

Embeds the OBSERVED sequence (residues with a complete N/CA/C backbone), NOT the
full SEQRES. That is deliberate: it makes index i of the embedding correspond to
observed residue i on the density side BY CONSTRUCTION, with no alignment step
that could silently drift. The cost is that ESM-C sees a sequence with internal
gaps where residues were unmodelled; for most chains that is a small perturbation
and it is the safer trade, given how much of this project's lost time came from
silent index misalignment.

Writes data/esmc_chains/<key>.npy of shape [n_obs, 1152], BOS/EOS stripped, with
a length assertion against the chain list -- load-bearing, not decorative.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("data/esmc_chains"))
    ap.add_argument("--model", default="esmc_600m")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    rows = list(csv.DictReader(open(args.chains)))
    if args.limit:
        rows = rows[: args.limit]
    args.outdir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(rows)} chains | model={args.model} | device={dev}", flush=True)

    model = ESMC.from_pretrained(args.model).to(dev)
    cfg = LogitsConfig(sequence=True, return_embeddings=True)

    done = skipped = 0
    for i, r in enumerate(rows):
        out = args.outdir / f"{r['key']}.npy"
        if out.exists():
            done += 1
            continue
        seq = r["seq"]
        if len(seq) > args.max_len:
            skipped += 1
            continue
        try:
            with torch.no_grad():
                t = model.encode(ESMProtein(sequence=seq))
                emb = model.logits(t, cfg).embeddings.squeeze(0)[1:-1]
            emb = emb.float().cpu().numpy()
            assert emb.shape[0] == len(seq), (
                f"{r['key']}: {emb.shape[0]} embeddings for {len(seq)} residues -- "
                "BOS/EOS handling or tokenisation changed; the density side would "
                "be misaligned")
            np.save(out, emb.astype(np.float32))
            done += 1
        except Exception as exc:
            skipped += 1
            print(f"  [{i+1}/{len(rows)}] SKIP {r['key']} {type(exc).__name__}: {exc}",
                  flush=True)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rows)}] {done} done, {skipped} skipped", flush=True)
    print(f"done: {done} cached, {skipped} skipped -> {args.outdir}")


if __name__ == "__main__":
    main()
