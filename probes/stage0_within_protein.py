"""How much of the density-target R2 is BETWEEN-protein rather than within?

Every R2 reported so far (esmc 0.169, dirmom 0.401, ...) uses a single GLOBAL
train-mean baseline, so between-protein variance sits in the denominator. A model
that merely got each protein's average density character right would already
score above zero without resolving any residue. This is not a hypothetical
concern in this codebase: the sibling ESM project measured 11.6% between-protein
/ 88.4% within, and per-protein centring collapsed its `band_50 -> coord@50`
result from 0.80 to 0.028.

Three quantities, all on the same held-out test residues:

  between_frac   ss_between / ss_total of the TARGET. Equivalently, the global R2
                 achieved by an ORACLE that knows each protein's mean target
                 vector and nothing else -- so it is directly comparable to the
                 headline numbers. If between_frac > 0.187, an oracle knowing
                 only per-protein means beats our ESM-C model.
  r2_global      1 - ss_res / ss_tot(global train mean). Reproduces the headline.
  r2_within      1 - ss_res / ss_tot(per-protein mean). Asks: does the model beat
                 a predictor handed each test protein's own mean? NEGATIVE means
                 it does not, i.e. the global number was carried by protein-level
                 signal. The MODEL never sees protein means here; only the
                 BASELINE does, which is what makes this the honest contrast.
  r2_wcent       both X and Y per-protein centred, refit. Pure within-protein
                 predictability. Here the model IS given per-protein centring, so
                 this is "conditional on knowing the protein, how well do we
                 resolve residues".
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from probes.align_esmc_density import AA1, onehot, seq_window
from probes.stage0_oracle_neighbourhood import chain_oracle


def _one(row):
    key, pdb, chain, split = row["key"], row["pdb"], row["chain"], row["split"]
    t, e = Path(row["_tgt"]) / f"{key}.npz", Path(row["_esmc"]) / f"{key}.npy"
    if not (t.exists() and e.exists()):
        return None
    try:
        dd = np.load(t, allow_pickle=True)
        feats, keep, seq = dd["feats"], dd["keep"], str(dd["seq"])
        emb = np.load(e)
        if len(emb) != len(seq) or len(keep) != len(seq) or int(keep.sum()) != len(feats):
            return None
        o = chain_oracle(pdb, chain, keep)
        if o is None:
            return None
        coord, comp, dirm, comp_far, dirm_far, bb = o
        if len(coord) != len(feats):
            return None
        sw = seq_window(seq, 3)[keep]
        own = onehot(np.array([AA1.index(c) if c in AA1 else -1 for c in seq]))[keep]
        return (split, key, emb[keep].astype(np.float32), sw, own, coord, comp,
                dirm, bb, feats)
    except Exception:
        return None


def ridge_fit(Xtr, Ytr, Xva, Yva, alphas=(1.0, 10.0, 100.0, 1000.0)):
    """Standardise X, centre Y, sweep alpha on val. Returns predict()."""
    mx, sx = Xtr.mean(0), Xtr.std(0)
    sx[sx < 1e-8] = 1.0
    my = Ytr.mean(0)
    A, B = (Xtr - mx) / sx, Ytr - my
    G, rhs = A.T @ A, A.T @ B
    best, best_sse = None, np.inf
    for a in alphas:
        W = np.linalg.solve(G + a * np.eye(G.shape[0]), rhs)
        sse = (((Yva - (((Xva - mx) / sx) @ W + my)) ** 2)).sum()
        if sse < best_sse:
            best, best_sse = W, sse
    return lambda X: ((X - mx) / sx) @ best + my, my


def group_center(A, prot):
    """Subtract each protein's own mean from its rows."""
    out = np.empty_like(A)
    for p in np.unique(prot):
        m = prot == p
        out[m] = A[m] - A[m].mean(0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--target-dir", type=Path, default=Path("data/density_targets"))
    ap.add_argument("--esmc-dir", type=Path, default=Path("data/esmc_chains"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("results/stage0_within_protein.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.chains)))
    if args.limit:
        rows = rows[: args.limit]
    for r in rows:
        r["_tgt"], r["_esmc"] = str(args.target_dir), str(args.esmc_dir)

    from multiprocessing import Pool
    acc: dict[str, list] = {}
    ok = bad = 0
    with Pool(args.workers) as p:
        for i, res in enumerate(p.imap(_one, rows, chunksize=4)):   # ORDERED
            if res is None:
                bad += 1
            else:
                acc.setdefault(res[0], []).append(res[1:])
                ok += 1
            if (i + 1) % 300 == 0:
                print(f"  [{i+1}/{len(rows)}] {ok} ok, {bad} skipped", flush=True)
    print(f"chains: {ok} ok, {bad} skipped", flush=True)

    names = ("esmc", "sw", "own", "coord", "comp", "dirm", "bb", "y")
    S, PROT = {}, {}
    for s, v in acc.items():
        S[s] = {n: np.concatenate([b[j + 1] for b in v])
                for j, n in enumerate(names)}
        PROT[s] = np.concatenate([np.full(len(b[-1]), k, np.int32)
                                  for k, b in enumerate(v)])
    for s in ("train", "val", "test"):
        print(f"{s}: {len(S[s]['y'])} residues, {len(np.unique(PROT[s]))} proteins")

    yte, pte = S["test"]["y"], PROT["test"]
    my_train = S["train"]["y"].mean(0)
    ss_tot_global = ((yte - my_train) ** 2).sum()
    ss_tot_within = (group_center(yte, pte) ** 2).sum()
    between_frac = float(1.0 - ss_tot_within / ss_tot_global)
    print(f"\nTARGET variance decomposition on test:")
    print(f"  between-protein fraction = {between_frac:.4f}")
    print(f"  within-protein  fraction = {1-between_frac:.4f}")
    print(f"  => an oracle knowing ONLY each protein's mean scores "
          f"global R2 = {between_frac:.4f}")

    arms = {"esmc": ("esmc",), "coord": ("coord",), "shellcomp": ("comp",),
            "dirmom": ("dirm",), "seqwin3": ("sw",),
            "full_oracle": ("comp", "dirm", "coord", "own", "sw", "bb")}
    res = {"_between_frac": between_frac,
           "_n_test": int(len(yte)),
           "_n_test_proteins": int(len(np.unique(pte)))}
    print(f"\n{'arm':14s} {'r2_global':>10s} {'r2_within':>10s} {'r2_wcent':>10s}")
    print("-" * 48)
    for name, keys in arms.items():
        cat = lambda s: np.concatenate([S[s][k] for k in keys], 1)
        Xtr, Xva, Xte = cat("train"), cat("val"), cat("test")
        pred, _ = ridge_fit(Xtr, S["train"]["y"], Xva, S["val"]["y"])
        ss_res = ((yte - pred(Xte)) ** 2).sum()
        r2g = float(1.0 - ss_res / ss_tot_global)
        r2w = float(1.0 - ss_res / ss_tot_within)
        # pure within-protein: centre X and Y per protein on every split
        predc, _ = ridge_fit(group_center(Xtr, PROT["train"]),
                             group_center(S["train"]["y"], PROT["train"]),
                             group_center(Xva, PROT["val"]),
                             group_center(S["val"]["y"], PROT["val"]))
        ycte = group_center(yte, pte)
        r2c = float(1.0 - ((ycte - predc(group_center(Xte, pte))) ** 2).sum()
                    / (ycte ** 2).sum())
        res[name] = {"r2_global": r2g, "r2_within": r2w, "r2_wcent": r2c,
                     "dim": int(Xtr.shape[1])}
        print(f"{name:14s} {r2g:+10.4f} {r2w:+10.4f} {r2c:+10.4f}", flush=True)
        del Xtr, Xva, Xte

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")
    print("r2_within < 0  => the model loses to a per-protein-mean predictor.")


if __name__ == "__main__":
    main()
