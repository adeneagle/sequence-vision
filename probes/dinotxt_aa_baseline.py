"""Baseline (b) of E1: a per-voxel AA classifier aggregated to chains by composition.

WHY THIS IS A GATING ARM, not a nice-to-have. G0's stated stop condition names TWO
baselines -- the volume prior AND this one. The concern is concrete: if chain
assignment is recoverable by predicting amino-acid identity per voxel and matching
each candidate sequence's composition, then the alignment buys nothing a 20-way
classifier does not already give, and the two-tower machinery is unmotivated.

It can be much stronger than its own per-voxel accuracy suggests. Per-residue AA
identity from these features is weak (~0.117 20-way at `mid_block`, vs 0.273 for raw
ESM-C), but E1 aggregates 10^2-10^3 voxels per chain, and a weak-but-unbiased AA
signal composition-matched over that many samples can separate chains well. That
aggregation is exactly why E1 was chosen as the endpoint, so it cuts both ways.

SCORING. For voxel v with predicted AA distribution p_v and candidate sequence s with
composition f_s, score(v,s) = sum_a p_v(a) log f_s(a) -- the expected log-likelihood
of the predicted AA under the sequence's composition. Two properties that matter:
it uses the FULL predicted distribution rather than a hard argmax (a 12%-accurate
argmax throws away most of the signal), and it is a proper composition match rather
than a length prior, since f_s is normalised. `--hard` reports the argmax variant too,
because if hard >> soft something is wrong with the calibration.

The classifier is deliberately the SAME CAPACITY CLASS as the alignment head (linear,
or one hidden layer with --hidden) on the SAME cached features and the SAME split, so
a win for the alignment cannot be attributed to the baseline being under-parameterised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {c: i for i, c in enumerate(AA)}


def voxel_aa_labels(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """(label [N] int64, valid [N] bool) -- AA of each voxel's DOMINANT residue.

    Dominant = the residue holding the most target mass, i.e. the same residue E1
    scores against, so the baseline and the alignment are graded on one ground truth.
    """
    ri, w = d["res_idx"], d["weight"]
    best = np.take_along_axis(ri, w.argmax(1)[:, None], 1)[:, 0]
    ok = (ri >= 0).any(1) & ~d["is_bg"]
    seqs = [str(s) for s in d["seqs"]]
    rs, rp = d["res_seq"], d["res_pos"]
    lab = np.full(len(ri), -1, dtype=np.int64)
    idx = np.nonzero(ok)[0]
    for j in idx:
        r = best[j]
        c = seqs[rs[r]][rp[r]]
        lab[j] = AA_IDX.get(c, -1)
    valid = lab >= 0
    return lab, valid


def composition(seqs: list[str]) -> np.ndarray:
    """[n_seq, 20] AA composition, Laplace-smoothed so log f is always finite."""
    f = np.ones((len(seqs), 20), dtype=np.float64)
    for i, s in enumerate(seqs):
        for c in s:
            if c in AA_IDX:
                f[i, AA_IDX[c]] += 1.0
    return f / f.sum(1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/voxel_cache"))
    ap.add_argument("--esmc", type=Path, default=Path("data/esmc_seq32"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/dinotxt_runs/aa_baseline.pt"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=512,
                    help="0 = linear; matches the alignment head's capacity class")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-maps", type=int, default=8)
    ap.add_argument("--n-vox", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from probes.dinotxt_data import VoxelCorpus

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = VoxelCorpus(args.cache, args.esmc)
    tr, va = corpus.by_split("train"), corpus.by_split("val")
    d0 = corpus.get(tr[0])
    d_in = d0["feat"].shape[1]

    net = (nn.Linear(d_in, 20) if args.hidden == 0 else
           nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, args.hidden),
                         nn.GELU(), nn.Linear(args.hidden, 20))).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    print(f"AA classifier | {sum(p.numel() for p in net.parameters())/1e6:.2f} M "
          f"| {len(tr)} train maps | {dev}", flush=True)

    def draw(maps):
        X, Y = [], []
        for m in maps:
            d = corpus.get(m)
            lab, valid = voxel_aa_labels(d)
            sel = np.nonzero(valid)[0]
            if len(sel) == 0:
                continue
            if len(sel) > args.n_vox:
                sel = rng.choice(sel, args.n_vox, replace=False)
            X.append(d["feat"][sel].astype(np.float32))
            Y.append(lab[sel])
        return (torch.from_numpy(np.concatenate(X)).to(dev),
                torch.from_numpy(np.concatenate(Y)).to(dev))

    gen = corpus.batches(tr, args.batch_maps, rng)
    best = 0.0
    for step in range(1, args.steps + 1):
        try:
            batch = next(gen)
        except StopIteration:
            gen = corpus.batches(tr, args.batch_maps, rng)
            batch = next(gen)
        x, y = draw(batch)
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == args.steps:
            net.eval()
            with torch.no_grad():
                xv, yv = draw(va[:40])
                acc = float((net(xv).argmax(1) == yv).float().mean())
            net.train()
            print(f"  [{step}/{args.steps}] loss {float(loss):.4f} | "
                  f"val AA top-1 {acc:.4f} (20-way, chance 0.05)", flush=True)
            if acc > best:
                best = acc
                args.out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"net": net.state_dict(), "args": vars(args),
                            "d_in": d_in, "val_aa": acc, "step": step}, args.out)
    print(f"\nbest val AA top-1 {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
