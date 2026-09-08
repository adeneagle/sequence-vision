"""Cache ESM-C 600m embeddings for the manifest's paired sequences.

Runs in the MAIN LAB ENV, not this project's env: `esm` is not installed here and
adding it risks CryoFM's numpy<2.0 pin. Invoke with

    pixi run --manifest-path ../../aden/aden/pixi.toml python data/extract_esmc.py

Writes data/esmc/<emdb_id>.npy of shape [L, 1152] (per-residue, BOS/EOS stripped)
plus data/esmc/index.json. Re-running skips what already exists.

The manifest pairs each map with its LARGEST chain, so the sequence embedded here
must be that same chain -- the density descriptor is pooled over exactly its Ca
atoms. A silent mismatch here would fabricate or destroy apparent alignment, so
the length assertion below is load-bearing, not decorative.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("data/esmc"))
    ap.add_argument("--model", default="esmc_600m")
    ap.add_argument("--max-len", type=int, default=2048, help="ESM-C context limit")
    args = ap.parse_args()

    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.manifest)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(rows)} sequences | model={args.model} | device={dev}")

    model = ESMC.from_pretrained(args.model).to(dev)
    cfg = LogitsConfig(sequence=True, return_embeddings=True)

    index, skipped = {}, []
    for r in tqdm(rows):
        eid, seq = r["emdb_id"], r["seq"]
        out = args.outdir / f"{eid}.npy"
        if out.exists():
            index[eid] = {"path": str(out), "len": len(seq), "chain_id": r.get("chain_id", "")}
            continue
        if len(seq) > args.max_len:
            skipped.append((eid, len(seq)))
            continue
        with torch.no_grad():
            t = model.encode(ESMProtein(sequence=seq))
            emb = model.logits(t, cfg).embeddings.squeeze(0)[1:-1]   # strip BOS/EOS
        emb = emb.float().cpu().numpy()
        # Load-bearing: a length mismatch means the wrong chain was embedded.
        assert emb.shape[0] == len(seq), f"{eid}: {emb.shape[0]} != {len(seq)}"
        np.save(out, emb)
        index[eid] = {"path": str(out), "len": len(seq), "chain_id": r.get("chain_id", "")}

    json.dump(index, open(args.outdir / "index.json", "w"), indent=2)
    print(f"\ncached {len(index)} / {len(rows)}")
    if skipped:
        print(f"skipped {len(skipped)} over the {args.max_len}-residue context limit: "
              f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
        print("NOTE: this truncates the sample toward shorter chains -- report it.")


if __name__ == "__main__":
    main()
