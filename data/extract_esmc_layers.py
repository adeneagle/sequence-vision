"""Cache ALL ESM-C hidden layers per residue, for an ESMFold2-style layer mix.

ESMFold2 does not use the final layer -- it consumes every layer. From the
Apache-2.0 JAX translation of the Biohub reference
(escalante-bio/esmjfold2, src/esmjfold2/language_model.py, LanguageModelShim):

    lm_z    = base_z_linear(hidden_states)        # shared LayerNorm -> Linear, per layer
    weights = softmax(base_z_combine)             # learned, one scalar per layer
    lm_z    = einsum("blnd,n->bld", lm_z, weights)  # weighted sum over LAYERS

i.e. an ELMo-style learned scalar mix over all `num_layers + 1` representations
(the `81` in their shape comment is ESMC-6B's 80 layers + embeddings).

Two consequences drive this file:
  * The LayerNorm BEFORE mixing is load-bearing -- layers differ a lot in
    activation scale, and an un-normalised weighted sum is dominated by whichever
    layer has the largest norm. We store PRE-norm states and normalise in the
    head, so the normalisation stays trainable/inspectable.
  * The shared projection is linear, so it commutes with the mix:
    sum_n w_n Linear(LN(h_n)) = Linear(sum_n w_n LN(h_n)). Nothing is lost by
    storing the d_model states and mixing there.

Also motivated by this project's own prior finding: a layer sweep on ESM-C found
LAYER 24 the most structure-rich (~3x the coarse-band energy of the final layer),
while our alignment used the final layer purely because it is the SDK default.

Runs in the MAIN LAB ENV. Writes fp16 [n_obs, n_layers, 1152] per chain
(~17 MB for a 200-residue chain, ~25 GB total for 1500).
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
    ap.add_argument("--outdir", type=Path, default=Path("data/esmc_layers"))
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
    print(f"{len(rows)} chains | {args.model} | {dev}", flush=True)

    model = ESMC.from_pretrained(args.model).to(dev)
    cfg = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)

    done = skipped = 0
    n_layers = None
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
                o = model.logits(t, cfg)
                hs = o.hidden_states            # [n_layers, 1, L+2, d]
            h = hs.squeeze(1)[:, 1:-1, :]       # strip BOS/EOS -> [n_layers, L, d]
            h = h.permute(1, 0, 2).float().cpu().numpy()   # -> [L, n_layers, d]
            assert h.shape[0] == len(seq), (
                f"{r['key']}: {h.shape[0]} positions for {len(seq)} residues -- "
                "BOS/EOS handling changed; the density side would be misaligned")
            if n_layers is None:
                n_layers = h.shape[1]
                print(f"  hidden_states: {h.shape[1]} layers x {h.shape[2]} dim",
                      flush=True)
            np.save(out, h.astype(np.float16))
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
