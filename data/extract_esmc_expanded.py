"""ESM-C layer 32 for the EXPANDED chain inventory (all chains, not the 1,500).

Runs in the MAIN LAB ENV -- `esm` is not installed in this project's env and
adding it risks CryoFM's numpy<2.0 pin:

    /mnt/main0/projects/et-foundation-vision-model/aden/aden/.pixi/envs/default/bin/python3.10 \
        data/extract_esmc_expanded.py

WHY LAYER 32, NOT THE FINAL LAYER AND NOT THE 36-LAYER MIX. Measured on this
project's own ridge probe: layer 32 is the best SINGLE layer (R2 0.184) against
the final layer's 0.169, and the full ESMFold2-style learned mix over all 36
reaches 0.220 -- of which only **+0.017** is the mixing itself (the rest is head
capacity, isolated by running layer 32 through the identical head). The 36-layer
stack for 17.5k chains would be ~36x this cache for that +0.017. So: layer 32
here for the expanded set, and the existing 36-layer `data/esmc_layers` cache
(1,500 chains) stays available for the mix ablation.

Note layer 32 is PRE-final-norm, unlike `data/esmc_chains` which stores the
post-norm final embedding. Measured to be a non-issue for this target (L35
pre-norm 0.167 vs post-norm final 0.169, ~0.002), but the consuming head applies
its own LayerNorm regardless -- load-bearing, since per-layer RMS spans 1.37..108
across depth and an un-normalised consumer would be decided by scale alone.

DEDUPED BY SEQUENCE, NOT BY CHAIN. 17,498 chains carry far fewer distinct
sequences (homo-oligomer copies share one), and the density target indexes
residues by (sequence, position), so one embedding per distinct sequence is
exactly what the loss consumes. Files are keyed by the sequence SHA1 that
`build_map_chains.py` already wrote, so the join is by content, not by name.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-chains", type=Path, default=Path("data/map_chains.csv"))
    ap.add_argument("--inv-dir", type=Path, default=Path("data/map_chains"))
    ap.add_argument("--outdir", type=Path, default=Path("data/esmc_seq32"))
    ap.add_argument("--model", default="esmc_600m")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    # Collect DISTINCT sequences across the whole inventory.
    rows = list(csv.DictReader(open(args.map_chains)))
    want: dict[str, str] = {}          # sha1 -> sequence
    seen_emd: set = set()
    for r in rows:
        emd = r["emd"]
        if emd in seen_emd:
            continue
        seen_emd.add(emd)
        d = np.load(args.inv_dir / f"{emd}.npz", allow_pickle=True)
        for s in d["seqs"]:
            s = str(s)
            if args.min_len <= len(s) <= args.max_len:
                want[hashlib.sha1(s.encode()).hexdigest()[:12]] = s
    keys = sorted(want)
    if args.limit:
        keys = keys[: args.limit]
    args.outdir.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(rows)} chains -> {len(want)} distinct sequences "
          f"({len(keys)} to do) | layer {args.layer} | {dev}", flush=True)

    model = ESMC.from_pretrained(args.model).to(dev)
    cfg = LogitsConfig(sequence=True, return_embeddings=True,
                       return_hidden_states=True)

    done = cached = skipped = 0
    t0 = time.time()
    for i, k in enumerate(keys):
        out = args.outdir / f"{k}.npy"
        if out.exists():
            cached += 1
            continue
        seq = want[k]
        try:
            with torch.no_grad():
                t = model.encode(ESMProtein(sequence=seq))
                r = model.logits(t, cfg)
                # hidden_states: [n_layers+1, B, L+2, 1152]; strip BOS/EOS.
                emb = r.hidden_states[args.layer].squeeze(0)[1:-1]
            emb = emb.float().cpu().numpy()
            assert emb.shape[0] == len(seq), (
                f"{k}: {emb.shape[0]} embeddings for {len(seq)} residues -- BOS/EOS "
                "handling or tokenisation changed; the density side would be "
                "misaligned, which is the failure mode this project has paid most for")
            tmp = out.with_suffix(".tmp.npy")
            with open(tmp, "wb") as fh:
                np.save(fh, emb.astype(np.float16))
            tmp.replace(out)
            done += 1
        except Exception as exc:
            skipped += 1
            if skipped <= 10:
                print(f"  SKIP {k} (len {len(seq)}) {type(exc).__name__}: {exc}",
                      flush=True)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(keys)}] {done} new, {cached} cached, "
                  f"{skipped} skipped, {(time.time()-t0)/60:.1f} min", flush=True)

    tot = sum(p.stat().st_size for p in args.outdir.glob("*.npy"))
    print(f"\nsequences: {done} new, {cached} cached, {skipped} skipped | "
          f"{tot/2**30:.2f} GiB | {(time.time()-t0)/60:.1f} min -> {args.outdir}")
    if done + cached == 0:
        raise SystemExit("FAILED: 0 sequences embedded.")
    if skipped > 0.5 * len(keys):
        raise SystemExit(f"FAILED: {skipped}/{len(keys)} skipped -- systematic.")


if __name__ == "__main__":
    main()
