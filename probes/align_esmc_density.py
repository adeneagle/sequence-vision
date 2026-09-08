"""Can ESM-C predict the per-residue density feature? The actual experiment.

Correspondence is assumed (deposited or fitted model), so this is a plain
supervised regression: ESM-C [1152] -> CryoFM local-frame feature [C], one
example per observed residue, evaluated on held-out mmseqs CLUSTERS.

Read the controls before the headline. On this project's history a raw positive
here is close to guaranteed and would mean nothing:

  shuffled    ESM-C rows permuted against targets WITHIN the test set. Must land
              at ~0 R2. If it does not, the pipeline leaks and every other number
              is void. Reported first, deliberately.
  aa          predict from the central residue's 20-d one-hot alone.
  seqwin      predict from a +-3 one-hot window (140-d). THE bar to beat: the
              homolog diagnostic showed a large part of this target is plain
              local sequence, so ESM-C must beat seqwin to have earned its place.
  esmc        the model under test.

Two reference points from the diagnostic, both on the same target:
  * 0.447 of target variance tracks sequence at homolog resolution -- roughly
    what a model that generalises across families should reach;
  * pose noise is 0.151 of total variance, so ~0.85 is the hard ceiling for any
    pose-free predictor, not 1.0.

R2 is computed per target dimension against the TRAIN mean and averaged, which
is the honest version -- a per-test-set mean would flatter every arm equally.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

AA1 = "ARNDCQEGHILKMFPSTWYV"


def onehot(idx: np.ndarray, n: int = 20) -> np.ndarray:
    M = np.zeros((len(idx), n), dtype=np.float32)
    ok = idx >= 0
    M[np.arange(len(idx))[ok], idx[ok]] = 1.0
    return M


def seq_window(seq: str, w: int) -> np.ndarray:
    """[L, 20*(2w+1)] one-hot of the +-w sequence window at every position."""
    idx = np.array([AA1.index(c) if c in AA1 else -1 for c in seq])
    cols = []
    for off in range(-w, w + 1):
        shifted = np.full(len(idx), -1)
        lo, hi = max(0, -off), min(len(idx), len(idx) - off)
        shifted[lo:hi] = idx[lo + off: hi + off]
        cols.append(onehot(shifted))
    return np.concatenate(cols, 1)


def load(rows, esmc_dir: Path, tgt_dir: Path, w: int):
    """Per split: (ESM-C, seqwin, aa, target). Indices are matched by construction."""
    out = {}
    for r in rows:
        e = esmc_dir / f"{r['key']}.npy"
        t = tgt_dir / f"{r['key']}.npz"
        if not (e.exists() and t.exists()):
            continue
        emb = np.load(e)
        d = np.load(t, allow_pickle=True)
        feats, keep, seq = d["feats"], d["keep"], str(d["seq"])
        # Three independent checks that the two modalities describe the same
        # residues. Any one failing means the pairing is fabricated.
        if len(emb) != len(seq) or len(keep) != len(seq) or int(keep.sum()) != len(feats):
            continue
        sw = seq_window(seq, w)[keep]
        aa = onehot(np.array([AA1.index(c) if c in AA1 else -1 for c in seq]))[keep]
        out.setdefault(r["split"], []).append((emb[keep], sw, aa, feats))
    return {k: tuple(np.concatenate([b[i] for b in v]) for i in range(4))
            for k, v in out.items()}


def ridge_r2(Xtr, Ytr, Xte, Yte, alphas=(1.0, 10.0, 100.0, 1000.0), Xva=None, Yva=None):
    """Ridge with alpha picked on val; R2 vs the TRAIN mean, averaged over dims."""
    mx, sx = Xtr.mean(0), Xtr.std(0)
    sx[sx < 1e-8] = 1.0
    my = Ytr.mean(0)
    A = (Xtr - mx) / sx
    B = Ytr - my
    G = A.T @ A
    rhs = A.T @ B
    best, best_r2 = None, -np.inf
    for a in alphas:
        Wt = np.linalg.solve(G + a * np.eye(G.shape[0]), rhs)
        if Xva is not None and len(Xva):
            P = ((Xva - mx) / sx) @ Wt + my
            r2 = 1.0 - ((Yva - P) ** 2).sum() / ((Yva - my) ** 2).sum()
            if r2 > best_r2:
                best, best_r2 = Wt, r2
        else:
            best = Wt
    P = ((Xte - mx) / sx) @ best + my
    ss_res = ((Yte - P) ** 2).sum(0)
    ss_tot = ((Yte - my) ** 2).sum(0)
    r2 = 1.0 - ss_res.sum() / ss_tot.sum()
    cos = float(np.mean(np.sum((Yte - my) * (P - my), 1) /
                        np.maximum(np.linalg.norm(Yte - my, axis=1)
                                   * np.linalg.norm(P - my, axis=1), 1e-9)))
    return float(r2), cos


def mlp_r2(Xtr, Ytr, Xva, Yva, Xte, Yte, hidden=2048, epochs=40, bs=4096,
           lr=1e-3, wd=1e-4, device=None):
    """2-layer MLP head, early-stopped on val. R2 still vs the TRAIN mean."""
    import torch
    import torch.nn as nn
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mx, sx = Xtr.mean(0), Xtr.std(0)
    sx[sx < 1e-8] = 1.0
    my, sy = Ytr.mean(0), Ytr.std(0)
    sy[sy < 1e-8] = 1.0
    t = lambda a, m, s: torch.from_numpy(((a - m) / s).astype(np.float32))
    xtr, ytr = t(Xtr, mx, sx), t(Ytr, my, sy)
    xva, yva = t(Xva, mx, sx), t(Yva, my, sy)
    xte = t(Xte, mx, sx)

    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, Ytr.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xtr[j].to(dev)), ytr[j].to(dev))
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = sum(nn.functional.mse_loss(net(xva[i:i + bs].to(dev)),
                                            yva[i:i + bs].to(dev),
                                            reduction="sum").item()
                     for i in range(0, len(xva), bs)) / yva.numel()
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
        P = np.concatenate([net(xte[i:i + bs].to(dev)).cpu().numpy()
                            for i in range(0, len(xte), bs)]) * sy + my
    ss_res = ((Yte - P) ** 2).sum()
    ss_tot = ((Yte - my) ** 2).sum()
    cos = float(np.mean(np.sum((Yte - my) * (P - my), 1) /
                        np.maximum(np.linalg.norm(Yte - my, axis=1)
                                   * np.linalg.norm(P - my, axis=1), 1e-9)))
    return float(1.0 - ss_res / ss_tot), cos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--esmc-dir", type=Path, default=Path("data/esmc_chains"))
    ap.add_argument("--target-dir", type=Path, default=Path("data/density_targets"))
    ap.add_argument("--seq-window", type=int, default=3)
    ap.add_argument("--mlp", action="store_true",
                    help="also fit a 2-layer MLP head; the linear probe is a "
                         "lower bound, not the ceiling")
    ap.add_argument("--pca", type=int, default=0,
                    help="optional PCA on the target, 0 = full dimensionality")
    ap.add_argument("--out", type=Path, default=Path("results/align_esmc_density.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.chains)))
    data = load(rows, args.esmc_dir, args.target_dir, args.seq_window)
    for k in ("train", "test"):
        if k not in data:
            raise SystemExit(f"no usable chains in split '{k}' -- check that both "
                             "the ESM-C and density caches were built")
    Etr, Str, Atr, Ytr = data["train"]
    Ete, Ste, Ate, Yte = data["test"]
    Eva, Sva, Ava, Yva = data.get("val", (None,) * 4)
    print(f"residues: train {len(Ytr)}  val {0 if Yva is None else len(Yva)}  "
          f"test {len(Yte)}   target dim {Ytr.shape[1]}")

    if args.pca:
        mu = Ytr.mean(0)
        _, _, Vt = np.linalg.svd(Ytr - mu, full_matrices=False)
        P = Vt[: args.pca].T
        Ytr, Yte = (Ytr - mu) @ P, (Yte - mu) @ P
        if Yva is not None:
            Yva = (Yva - mu) @ P

    rng = np.random.default_rng(0)
    arms = {
        # First, deliberately: if this is not ~0 the rest is void.
        "shuffled": (Etr, Ete[rng.permutation(len(Ete))], Eva),
        "aa": (Atr, Ate, Ava),
        f"seqwin{args.seq_window}": (Str, Ste, Sva),
        "esmc": (Etr, Ete, Eva),
    }
    res = {}
    for name, (Xtr, Xte, Xva) in arms.items():
        r2, cos = ridge_r2(Xtr, Ytr, Xte, Yte, Xva=Xva, Yva=Yva)
        res[name] = {"r2": r2, "cos": cos, "dim": int(Xtr.shape[1])}
        print(f"  {name:<12} dim {Xtr.shape[1]:>5}   R2 {r2:+.4f}   cos {cos:+.4f}")

    if args.mlp and Yva is not None:
        for name, (Xtr, Xte, Xva) in (("seqwin%d+mlp" % args.seq_window,
                                       (Str, Ste, Sva)), ("esmc+mlp", (Etr, Ete, Eva))):
            r2, cos = mlp_r2(Xtr, Ytr, Xva, Yva, Xte, Yte)
            res[name] = {"r2": r2, "cos": cos, "dim": int(Xtr.shape[1])}
            print(f"  {name:<12} dim {Xtr.shape[1]:>5}   R2 {r2:+.4f}   cos {cos:+.4f}")

    res["_meta"] = {"n_train": len(Ytr), "n_test": len(Yte),
                    "target_dim": int(Ytr.shape[1]),
                    "diagnostic_sequence_tracked": 0.447,
                    "pose_ceiling": 0.849}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))

    print("\nreference: 0.447 of this target's variance tracks sequence at homolog")
    print("resolution; pose noise caps any pose-free predictor near 0.85.")
    # The shuffled arm is judged on COSINE, not R2. Predicting from permuted
    # features gives predictions uncorrelated with the targets, so residuals
    # exceed the total variance and R2 goes NEGATIVE by roughly the variance of
    # the predictions -- that is correct behaviour, not leakage. What must be ~0
    # is the correlation, and R2 must not be positive.
    sh = res["shuffled"]
    if abs(sh["cos"]) > 0.05 or sh["r2"] > 0.02:
        print(f"\n*** SHUFFLED CONTROL FAILED (cos {sh['cos']:+.4f}, "
              f"R2 {sh['r2']:+.4f}) -- there is leakage; ignore every other "
              "number. ***")
    else:
        print(f"\nshuffled control OK: cos {sh['cos']:+.4f} ~ 0 "
              f"(its R2 {sh['r2']:+.4f} is negative by construction)")
    if res["esmc"]["r2"] <= res[f"seqwin{args.seq_window}"]["r2"]:
        print("ESM-C does NOT beat the local sequence window: the PLM adds nothing "
              "here beyond +-3 residues of sequence.")


if __name__ == "__main__":
    main()
