"""Which ESM-C layer(s) predict the density feature? Sweep, then ESMFold2's mix.

Our alignment used ESM-C's FINAL layer only, because that is what
`LogitsConfig(return_embeddings=True)` returns -- never a decision. Two reasons
to revisit, pulling in opposite directions:

  * this project's own layer sweep found LAYER 24 the most structure-rich
    (~3x the coarse-band energy of the final layer), and our target is
    structural;
  * but hidden_states are PRE-final-norm, and the post-norm final embedding
    previously showed HIGHER linear cross-predictability (0.13 vs 0.01) -- and
    our head is nearly linear (an MLP bought only +0.018 over ridge).

So it is genuinely untested. `--mode sweep` answers it directly.

`--mode mix` reproduces ESMFold2's mechanism, verified against the Apache-2.0
JAX translation (escalante-bio/esmjfold2, language_model.py::LanguageModelShim):

    lm_z    = base_z_linear(hidden_states)          # shared LayerNorm -> Linear
    weights = softmax(base_z_combine)               # one learned scalar / layer
    lm_z    = einsum("blnd,n->bld", lm_z, weights)  # weighted sum over LAYERS

The LayerNorm before mixing is not cosmetic: measured per-layer RMS on our own
cache runs 1.37 -> 108 across the 36 layers, so an un-normalised weighted sum is
decided entirely by the last few layers regardless of the learned weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from probes.align_esmc_density import AA1, onehot, ridge_r2, seq_window


def load_split(rows, layer_dir: Path, tgt_dir: Path, layers, split):
    """(X [N, n_layers, d], Y [N, C]) for one split. `layers` selects a subset."""
    X, Y = [], []
    for r in rows:
        if r["split"] != split:
            continue
        e, t = layer_dir / f"{r['key']}.npy", tgt_dir / f"{r['key']}.npz"
        if not (e.exists() and t.exists()):
            continue
        h = np.load(e)                                   # [L, n_layers, d] fp16
        d = np.load(t, allow_pickle=True)
        feats, keep, seq = d["feats"], d["keep"], str(d["seq"])
        if len(h) != len(seq) or len(keep) != len(seq) or int(keep.sum()) != len(feats):
            continue
        X.append(h[keep][:, layers])
        Y.append(feats)
    if not X:
        return None, None
    return np.concatenate(X), np.concatenate(Y)


def mix_head(Xtr, Ytr, Xva, Yva, Xte, Yte, d_out=1024, hidden=2048, epochs=40,
             bs=2048, lr=1e-3, wd=1e-4):
    """ESMFold2 LanguageModelShim: shared LN->Linear per layer, softmax mix, head."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_layers, d_model = Xtr.shape[1], Xtr.shape[2]
    my, sy = Ytr.mean(0), Ytr.std(0)
    sy[sy < 1e-8] = 1.0

    class Shim(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(d_model)          # shared across layers
            self.proj = nn.Linear(d_model, d_out)    # shared across layers
            self.combine = nn.Parameter(torch.zeros(n_layers))
            self.head = nn.Sequential(nn.GELU(), nn.Linear(d_out, hidden),
                                      nn.GELU(), nn.Linear(hidden, Ytr.shape[1]))

        def forward(self, h):                        # h: [B, n_layers, d_model]
            z = self.proj(self.ln(h))                # [B, n_layers, d_out]
            w = torch.softmax(self.combine, 0)
            return self.head(torch.einsum("bnd,n->bd", z, w))

    net = Shim().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    ytr = torch.from_numpy(((Ytr - my) / sy).astype(np.float32))
    yva = torch.from_numpy(((Yva - my) / sy).astype(np.float32))
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            j = perm[i:i + bs].numpy()
            xb = torch.from_numpy(Xtr[j].astype(np.float32)).to(dev)
            opt.zero_grad()
            nn.functional.mse_loss(net(xb), ytr[j].to(dev)).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = sum(nn.functional.mse_loss(
                net(torch.from_numpy(Xva[i:i + bs].astype(np.float32)).to(dev)),
                yva[i:i + bs].to(dev), reduction="sum").item()
                for i in range(0, len(Xva), bs)) / yva.numel()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        P = np.concatenate([
            net(torch.from_numpy(Xte[i:i + bs].astype(np.float32)).to(dev)).cpu().numpy()
            for i in range(0, len(Xte), bs)]) * sy + my
        w = torch.softmax(net.combine.detach().cpu(), 0).numpy()
    r2 = 1.0 - ((Yte - P) ** 2).sum() / ((Yte - my) ** 2).sum()
    cos = float(np.mean(np.sum((Yte - my) * (P - my), 1) /
                        np.maximum(np.linalg.norm(Yte - my, axis=1)
                                   * np.linalg.norm(P - my, axis=1), 1e-9)))
    return float(r2), cos, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--layer-dir", type=Path, default=Path("data/esmc_layers"))
    ap.add_argument("--target-dir", type=Path, default=Path("data/density_targets"))
    ap.add_argument("--mode", choices=["sweep", "mix", "single"], default="sweep",
                    help="'single' runs ONE layer through the identical Shim head, "
                         "isolating the mix's contribution from the head's -- the "
                         "sweep uses ridge, so mix-vs-sweep otherwise conflates "
                         "layer mixing with nonlinearity")
    ap.add_argument("--layer", type=int, default=32,
                    help="which layer for --mode single (32 won the sweep)")
    ap.add_argument("--sweep-layers", type=int, nargs="*",
                    default=[0, 8, 16, 20, 24, 28, 32, 35])
    ap.add_argument("--max-train", type=int, default=0,
                    help="subsample training residues; 0 = all")
    ap.add_argument("--out", type=Path, default=Path("results/align_layer_mix.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.chains)))
    layers = (args.sweep_layers if args.mode == "sweep"
              else [args.layer] if args.mode == "single" else None)
    if layers is None:
        probe = next(p for p in args.layer_dir.glob("*.npy"))
        layers = list(range(np.load(probe).shape[1]))
    print(f"mode={args.mode}  layers={layers}")

    data = {s: load_split(rows, args.layer_dir, args.target_dir, layers, s)
            for s in ("train", "val", "test")}
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = (data[s] for s in ("train", "val", "test"))
    if Xtr is None or Xte is None:
        raise SystemExit("no usable chains -- is the layer cache built?")
    if args.max_train and len(Xtr) > args.max_train:
        sel = np.random.default_rng(0).choice(len(Xtr), args.max_train, replace=False)
        Xtr, Ytr = Xtr[sel], Ytr[sel]
    print(f"residues: train {len(Ytr)}  val {len(Yva)}  test {len(Yte)}  "
          f"X {Xtr.shape}  Y {Ytr.shape}")

    res = {}
    if args.mode == "sweep":
        for k, L in enumerate(layers):
            r2, cos = ridge_r2(Xtr[:, k].astype(np.float32), Ytr,
                               Xte[:, k].astype(np.float32), Yte,
                               Xva=Xva[:, k].astype(np.float32), Yva=Yva)
            res[f"layer{L}"] = {"r2": r2, "cos": cos}
            print(f"  layer {L:>2}   R2 {r2:+.4f}   cos {cos:+.4f}")
    else:
        r2, cos, w = mix_head(Xtr, Ytr, Xva, Yva, Xte, Yte)
        tag = "layer_mix" if args.mode == "mix" else f"single_layer{args.layer}"
        res[tag] = {"r2": r2, "cos": cos, "weights": [float(x) for x in w]}
        label = ("ESMFold2 layer mix" if args.mode == "mix"
                 else f"single layer {args.layer}, same head")
        print(f"  {label}   R2 {r2:+.4f}   cos {cos:+.4f}")
        if args.mode == "mix":
            top = np.argsort(w)[::-1][:6]
            print("  top layers by learned weight: "
                  + ", ".join(f"L{int(i)}={w[i]:.3f}" for i in top))
            print(f"  weight spread: min {w.min():.4f} max {w.max():.4f} "
                  f"(uniform {1/len(w):.4f}) -- a flat profile means the gain is "
                  "ensembling, not selection")

    res["_meta"] = {"mode": args.mode, "layers": [int(x) for x in layers],
                    "n_train": int(len(Ytr)), "n_test": int(len(Yte)),
                    "final_layer_reference_r2": 0.187}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print("\nreference: final-layer ESM-C + MLP reached R2 0.187 / cos 0.417;"
          "\nsequence-tracked 0.447; pose ceiling 0.849.")


if __name__ == "__main__":
    main()
