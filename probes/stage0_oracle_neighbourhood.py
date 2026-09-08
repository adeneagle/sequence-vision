"""Stage 0: how much of the density target does the GROUND-TRUTH neighbourhood explain?

The gate before building any pairwise architecture. Our per-residue readout
`f(esm_i) -> CryoFM up_blocks[0] local-frame feature` sits at R2 0.187 while
0.447 of the target tracks sequence at homolog resolution. ESMFold2 says the
missing capacity is RELATIONAL: it consumes ESM-C only as a pair representation
(outer product -> triangle multiplication -> row-attention pooling), never as a
per-residue vector. Before paying for that, measure whether the gap is actually
neighbourhood-shaped.

THE ORACLE IS DELIBERATELY MATCHED TO WHAT A PAIR CHANNEL CAN REPRESENT.
A pair rep with row-attention pooling computes `sum_j w_ij * g(x_i, x_j)`. Given
a perfect distance oracle in that channel, the representable functions are
exactly `sum_j g(aa_j, d_ij)` -- aggregate over neighbours of (identity,
distance). That is arm `shellcomp`. So `shellcomp` is not an arbitrary rich
feature set; it is an upper bound on Stage 1 with the hard part (inferring
distances from sequence) replaced by ground truth.

Reading the ladder:
  coord      neighbour COUNTS per shell -- burial/packing only, no identity
  shellcomp  per-shell 20-d AA composition  <-- the pairwise-reachable bound
  dirmom     per-shell directional moments in the residue's OWN N-CA-C frame.
             NOT pairwise-reachable: needs the frame, i.e. needs the structure.
             The (dirmom - shellcomp) gap is the part of the target that is
             frame-directional and therefore out of reach from sequence pairs.
  full       everything + own AA + seqwin3: how neighbourhood-determined the
             target is at all.

Decision rule fixed BEFORE running, so this is not a post-hoc read:
  * shellcomp >> 0.187  -> the pairwise channel has real headroom; build Stage 1.
  * shellcomp ~= 0.187  -> perfect distances+identities add nothing; the residual
                           is frame-directional or side-chain detail, and no
                           sequence-driven pair architecture recovers it. Stop.
Also reported: `esmc + shellcomp`. If the oracle is largely ADDITIVE to ESM-C,
the information is genuinely missing from the per-residue readout rather than
already latent in it.

Honest limits of the oracle (both make it a LOWER bound on
"neighbourhood-determined variance", so they cannot manufacture a positive):
  * Ca positions + residue identity only. No side-chain rotamers, though the
    simulated density that produced the target used every atom.
  * neighbours = observed residues with a complete N/CA/C backbone, which is
    the residue list we have coordinates for; `simulate()` used the full chain.
The density was simulated PER CHAIN (`simulate(..., chain=...)`), so restricting
the neighbourhood to this chain is exact, not an approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from probes.align_esmc_density import AA1, onehot, ridge_r2, seq_window

# Shells in Angstrom. Resolution is concentrated where the tap is sensitive:
# up_blocks[0] has ~9-10 A half-decay, with a small tail out to the 46.5 A box
# edge, so the bins are fine below ~15 A and coarse beyond.
EDGES = np.array([0.0, 5.0, 6.5, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 48.0])
N_SHELL = len(EDGES) - 1
SEQ_NEAR = 4          # |i-j| <= SEQ_NEAR counts as sequence-adjacent


def chain_oracle(pdb: str, chain: str, keep: np.ndarray):
    """Oracle neighbourhood features for the kept residues of one chain.

    Returns (coord [n,S], comp [n,S*20], dirm [n,S*9]) or None on failure.
    Anchors are the kept residues; NEIGHBOURS are all observed residues.
    """
    from probes.homolog_diagnostic_residue import chain_backbone

    seq, ca, fr = chain_backbone(pdb, chain)
    L = len(seq)
    if L != len(keep) or L < 2:
        return None
    aa = np.array([AA1.index(c) if c in AA1 else -1 for c in seq])
    idx = np.nonzero(keep)[0]

    d = np.linalg.norm(ca[idx][:, None, :] - ca[None, :, :], axis=2)  # [n, L]
    # shell id per (anchor, neighbour); -1 = outside the outermost edge
    sh = np.searchsorted(EDGES, d, side="right") - 1
    sh[(d >= EDGES[-1])] = -1
    sh[np.arange(len(idx)), idx] = -1                      # drop self

    n = len(idx)
    coord = np.zeros((n, N_SHELL), np.float32)
    comp = np.zeros((n, N_SHELL, 20), np.float32)
    dirm = np.zeros((n, N_SHELL, 9), np.float32)
    # Sequence-adjacent neighbours are a confound: the direction to i+-1 in i's
    # own N-CA-C frame is essentially the backbone dihedrals, i.e. secondary
    # structure. The sibling project found >50% of short-range predictability
    # was backbone adjacency, so the far-only variants are mandatory, not
    # optional. `bb` isolates that near term explicitly.
    comp_far = np.zeros((n, N_SHELL, 20), np.float32)
    dirm_far = np.zeros((n, N_SHELL, 9), np.float32)
    bb = np.zeros((n, 4, 4), np.float32)

    # Displacements rotated into each anchor's own N-CA-C frame -- the same
    # frames used to cut the density boxes, so "direction" means the same thing
    # on both sides.
    for a in range(n):
        i = idx[a]
        m = sh[a] >= 0
        if not m.any():
            continue
        s = sh[a][m]
        np.add.at(coord[a], s, 1.0)
        va = aa[m]
        ok = va >= 0
        np.add.at(comp[a], (s[ok], va[ok]), 1.0)
        v = (ca[m] - ca[i]) @ fr[i].T                      # [k, 3] local frame
        u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
        quad = np.stack([u[:, 0] * u[:, 0], u[:, 1] * u[:, 1], u[:, 2] * u[:, 2],
                         u[:, 0] * u[:, 1], u[:, 0] * u[:, 2], u[:, 1] * u[:, 2]], 1)
        np.add.at(dirm[a, :, 0:3], s, u)
        np.add.at(dirm[a, :, 3:9], s, quad)

        # far-only: drop |i-j| <= SEQ_NEAR so tertiary packing is separated
        # from backbone conformation
        far = np.abs(np.nonzero(m)[0] - i) > SEQ_NEAR
        if far.any():
            sf = s[far]
            np.add.at(dirm_far[a, :, 0:3], sf, u[far])
            np.add.at(dirm_far[a, :, 3:9], sf, quad[far])
            vaf, okf = aa[m][far], (aa[m][far] >= 0)
            np.add.at(comp_far[a], (sf[okf], vaf[okf]), 1.0)

        # explicit backbone conformation: direction + distance to i-2,i-1,i+1,i+2
        for t, off in enumerate((-2, -1, 1, 2)):
            j = i + off
            if 0 <= j < L:
                w = (ca[j] - ca[i]) @ fr[i].T
                nw = np.linalg.norm(w)
                bb[a, t, 0:3] = w / max(nw, 1e-8)
                bb[a, t, 3] = nw

    return (coord, comp.reshape(n, -1), dirm.reshape(n, -1),
            comp_far.reshape(n, -1), dirm_far.reshape(n, -1), bb.reshape(n, -1))


def _one(row):
    """Worker: oracle + sequence features + target for one chain."""
    key, pdb, chain, split = row["key"], row["pdb"], row["chain"], row["split"]
    t = Path(row["_tgt"]) / f"{key}.npz"
    e = Path(row["_esmc"]) / f"{key}.npy"
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
        return (split, emb[keep].astype(np.float32), sw, own,
                coord, comp, dirm, comp_far, dirm_far, bb, feats)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--target-dir", type=Path, default=Path("data/density_targets"))
    ap.add_argument("--esmc-dir", type=Path, default=Path("data/esmc_chains"))
    ap.add_argument("--cache", type=Path, default=Path("data/stage0_oracle.npz"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 8))
    ap.add_argument("--mlp", action="store_true", help="also run MLP heads on key arms")
    ap.add_argument("--out", type=Path, default=Path("results/stage0_oracle.json"))
    args = ap.parse_args()

    if args.cache.exists():
        print(f"loading cached features from {args.cache}")
        z = np.load(args.cache)
        S = {s: {k: z[f"{s}_{k}"] for k in
                 ("esmc", "sw", "own", "coord", "comp", "dirm",
                  "comp_far", "dirm_far", "bb", "y")}
             for s in ("train", "val", "test")}
    else:
        rows = list(csv.DictReader(open(args.chains)))
        if args.limit:
            rows = rows[: args.limit]
        for r in rows:
            r["_tgt"], r["_esmc"] = str(args.target_dir), str(args.esmc_dir)
        print(f"{len(rows)} chains | {args.workers} workers | shells {EDGES.tolist()}",
              flush=True)
        from multiprocessing import Pool
        acc: dict[str, list] = {}
        ok = bad = 0
        with Pool(args.workers) as p:
            for i, res in enumerate(p.imap_unordered(_one, rows, chunksize=4)):
                if res is None:
                    bad += 1
                else:
                    acc.setdefault(res[0], []).append(res[1:])
                    ok += 1
                if (i + 1) % 200 == 0:
                    print(f"  [{i+1}/{len(rows)}] {ok} ok, {bad} skipped", flush=True)
        print(f"chains: {ok} ok, {bad} skipped", flush=True)
        names = ("esmc", "sw", "own", "coord", "comp", "dirm",
                 "comp_far", "dirm_far", "bb", "y")
        S = {s: {n: np.concatenate([b[j] for b in v])
                 for j, n in enumerate(names)} for s, v in acc.items()}
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.cache, **{f"{s}_{n}": a for s, d in S.items()
                                for n, a in d.items()})
        print(f"cached -> {args.cache}", flush=True)

    for s in ("train", "val", "test"):
        print(f"{s}: {len(S[s]['y'])} residues")
    ytr, yva, yte = (S[s]["y"] for s in ("train", "val", "test"))

    def cat(s, keys):
        return np.concatenate([S[s][k] for k in keys], 1)

    # Ladder. `shellcomp` is the pairwise-reachable bound; `dirmom`/`full` sit
    # above it and are NOT reachable from sequence pairs.
    arms = {
        # references, same split / residues
        "seqwin3":            ("sw",),
        "esmc":               ("esmc",),
        # --- reachable by ONE outer product + row pooling (no triangle stack)
        "coord":              ("coord",),
        "shellcomp":          ("comp",),
        "shellcomp+own":      ("comp", "own"),
        "shellcomp_far":      ("comp_far",),
        # --- frame-directional: needs relative geometry reconstructed
        "bb_only":            ("bb",),
        "dirmom":             ("dirm",),
        "dirmom_far":         ("dirm_far",),
        "bb+dirmom_far":      ("bb", "dirm_far"),
        "shellcomp+dirmom":   ("comp", "dirm"),
        "full_oracle":        ("comp", "dirm", "coord", "own", "sw", "bb"),
        # --- additivity with the learned per-residue readout
        "esmc+shellcomp":     ("esmc", "comp"),
        "esmc+bb":            ("esmc", "bb"),
        "esmc+full_oracle":   ("esmc", "comp", "dirm", "coord", "own", "sw", "bb"),
    }

    res = {}
    print(f"\n{'arm':22s} {'dim':>6s} {'R2':>9s} {'cos':>8s}")
    print("-" * 48)
    for name, keys in arms.items():
        Xtr, Xva, Xte = (cat(s, keys) for s in ("train", "val", "test"))
        r2, cos = ridge_r2(Xtr, ytr, Xte, yte, Xva=Xva, Yva=yva)
        res[name] = {"r2": r2, "cos": cos, "dim": int(Xtr.shape[1])}
        print(f"{name:22s} {Xtr.shape[1]:6d} {r2:+9.4f} {cos:+8.4f}", flush=True)
        del Xtr, Xva, Xte

    if args.mlp:
        from probes.align_esmc_density import mlp_r2
        print("\n--- MLP heads ---")
        for name in ("shellcomp", "dirmom_far", "full_oracle", "esmc+full_oracle"):
            keys = arms[name]
            Xtr, Xva, Xte = (cat(s, keys) for s in ("train", "val", "test"))
            r2, cos = mlp_r2(Xtr, ytr, Xva, yva, Xte, yte)
            res[name + "+mlp"] = {"r2": r2, "cos": cos, "dim": int(Xtr.shape[1])}
            print(f"{name+'+mlp':22s} {Xtr.shape[1]:6d} {r2:+9.4f} {cos:+8.4f}",
                  flush=True)
            del Xtr, Xva, Xte

    res["_meta"] = {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "target_dim": int(ytr.shape[1]), "shell_edges": EDGES.tolist(),
        "reference_esmc_ridge": 0.169, "reference_esmc_mlp": 0.187,
        "reference_seqwin3": 0.036,
        "diagnostic_sequence_tracked": 0.447, "pose_ceiling": 0.849,
        "seq_near_threshold": SEQ_NEAR,
        "single_outer_product_bound_arm": "shellcomp",
        "note": ("shellcomp bounds a single outer-product + row-pool head. "
                 "Triangle multiplicative updates are designed to reconstruct "
                 "relative geometry from pairwise distances, so they are NOT "
                 "bounded by shellcomp -- their ceiling is the dirmom arms."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")
    print("decision rule: shellcomp >> 0.187 -> build Stage 1; "
          "shellcomp ~= 0.187 -> the pair channel is not the missing lever.")


if __name__ == "__main__":
    main()
