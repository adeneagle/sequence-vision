"""O4 Task 2 (done properly): is there ONE Q_R per rotation, shared across structures?

The brief's claim is `f(R.x)(Rv) ~ Q_R f(x)(v)` with **Q_R a function of R ALONE**.
That is what makes `R -> Q_R` a homomorphism and lets channel space carry a
representation of SO(3).

`layer_relational_invariance.py` does NOT test this. Read at line ~189: `R =
random_so3(rng)` sits INSIDE the per-structure loop, so every structure gets its
own rotations, and `procrustes_cos(A, B)` then fits a FRESH Q for every
(structure, rotation) pair from ~30 residues after the held-out split. That
measures "for this structure and this rotation, does SOME orthogonal map exist" --
a strictly weaker claim that permits Q to depend on the input and therefore
cannot support any irrep decomposition. Its 0.857 is an optimistic upper bound.

This probe fixes both problems:
  * **Rotations are SHARED across structures**, so a single Q_R is identifiable.
  * **Q_R is fitted pooled over TRAIN structures and scored on HELD-OUT
    structures.** Held-out *residues* (the old split) leak, because residues of
    the same structure share its map; held-out *structures* do not.
  * Conditioning improves by the pooling factor: ~40 structures x 60 residues =
    2400 samples/rotation instead of 30, against an orthogonal group of dimension
    C(C-1)/2 (130k at C=512).
  * **Homomorphism check** on the same fitted maps: include R1, R2 and R1R2 in the
    shared set and compare Q_{R1R2} against Q_{R1} Q_{R2}. This residual bounds
    everything downstream (generators, Casimir, multiplicities).

Rotations use INDEPENDENT re-simulation of the deposited model (not resampling of
an already-sampled volume), so discretisation error is not charged to
equivariance -- stronger than the Fourier-rotation fix the brief suggests.

Reported per tap:
  per_struct   the OLD weak number (fresh Q per structure), for calibration
  pooled_in    pooled Q scored on the structures it was fitted on
  pooled_ho    **pooled Q scored on HELD-OUT structures -- the real quantity**
  homo         median cosine of (X Q1 Q2) vs Y_{R1R2}, i.e. does the map compose
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def _fit_q(X, Y):
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    return U @ Vt


def _med_cos(P, Y):
    num = (P * Y).sum(1)
    den = np.linalg.norm(P, axis=1) * np.linalg.norm(Y, axis=1)
    return float(np.median(num / np.maximum(den, 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--limit", type=int, default=40, help="structures")
    ap.add_argument("--n-res", type=int, default=60)
    ap.add_argument("--max-chain-len", type=int, default=400)
    ap.add_argument("--d-min", type=float, default=3.0)
    ap.add_argument("--timestep", type=int, default=261)
    ap.add_argument("--couple", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/o4_pooled_qr.json"))
    args = ap.parse_args()

    from teachers.cryofm_tap import (MODEL_VOXEL_SIZE, PATCH, CryoFM2Tap,
                                     preprocess, sample_at)
    from probes.layer_rotation_invariance import ALL_TAPS, _coords, _sim
    from probes.local_frame_stability import random_so3
    from probes.homolog_diagnostic_residue import chain_backbone

    rng = np.random.default_rng(args.seed)
    R1, R2 = random_so3(rng), random_so3(rng)
    ROTS = {"R1": R1, "R2": R2, "R1R2": R1 @ R2, "R3": random_so3(rng)}
    print(f"shared rotations: {list(ROTS)} (R1R2 included for the homomorphism check)")

    rows = [r for r in csv.DictReader(open(args.chains))
            if int(r["n_obs"]) <= args.max_chain_len][: args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tap = CryoFM2Tap(args.ckpt, taps=ALL_TAPS, device=dev, batch_size=args.batch_size)
    margin = PATCH // 2 * MODEL_VOXEL_SIZE
    nl = (args.timestep / 1000.0) if args.couple else 0.0
    print(f"{len(rows)} structures | t={args.timestep} "
          f"| {'COUPLED' if args.couple else 'decoupled'} | {dev}", flush=True)

    # per structure: {tap: {'id': [n,C], 'R1': [n,C], ...}}
    store: list[dict] = []
    for i, r in enumerate(rows):
        try:
            seq, ca, fr = chain_backbone(r["pdb"], r["chain"])
            volA, oA, spA = _sim(r["pdb"], r["chain"], None, margin, args.d_min)
            cA, _ = _coords(ca, fr, r["pdb"], r["chain"], None, oA, spA)
            sh = np.array(volA.shape)
            keep = np.where(np.all((cA >= PATCH // 2)
                                   & (cA < sh[None] - PATCH // 2), axis=1))[0]
            if len(keep) < 25:
                raise ValueError("too few residues with clearance")
            if len(keep) > args.n_res:
                keep = rng.choice(keep, args.n_res, replace=False)
            g = {t: {"id": sample_at(fv, cA[keep])} for t, fv in tap.feature_volumes(
                preprocess(volA, MODEL_VOXEL_SIZE), timestep=args.timestep,
                noise_level=nl).items()}
            ok = True
            for name, R in ROTS.items():
                volB, oB, spB = _sim(r["pdb"], r["chain"], R, margin, args.d_min)
                cB, _ = _coords(ca, fr, r["pdb"], r["chain"], R, oB, spB)
                shB = np.array(volB.shape)
                if not np.all(np.all((cB[keep] >= PATCH // 2)
                                     & (cB[keep] < shB[None] - PATCH // 2), axis=1)):
                    ok = False
                    break
                for t, fv in tap.feature_volumes(
                        preprocess(volB, MODEL_VOXEL_SIZE), timestep=args.timestep,
                        noise_level=nl).items():
                    g[t][name] = sample_at(fv, cB[keep])
            if ok:
                store.append(g)
        except Exception as exc:
            print(f"  skip {r['key']}: {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(rows)}] {len(store)} structures kept", flush=True)

    n = len(store)
    if n < 8:
        raise SystemExit(f"only {n} structures usable")
    ntr = max(4, int(0.6 * n))
    # RANK GUARD. Orthogonal Procrustes needs X^T Y full rank, i.e. at least as
    # many pooled TRAIN samples as channels. Below that, Q is underdetermined and
    # pooled_ho is garbage by construction -- the smoke (240 samples, 512 ch)
    # showed the signature: pooled_in 0.851 vs pooled_ho 0.327. Warn loudly per
    # tap rather than silently reporting a false negative.
    n_train_samples = ntr * args.n_res
    widest = max(store[0][t]["id"].shape[1] for t in ALL_TAPS)
    print(f"pooled train samples: {n_train_samples}; widest tap {widest} channels; "
          f"ratio {n_train_samples / widest:.2f}x"
          + ("  ** RANK DEFICIENT for the widest taps -- raise --limit **"
             if n_train_samples < 2 * widest else "  (ok)"))
    tr, ho = list(range(ntr)), list(range(ntr, n))
    print(f"\n{n} structures: {len(tr)} fit / {len(ho)} held out", flush=True)

    ctr = lambda M: M - M.mean(0, keepdims=True)
    res = {}
    print(f"\n{'tap':16s} {'per_struct':>11s} {'pooled_in':>10s} {'pooled_ho':>10s} {'homo':>8s}")
    print("-" * 62)
    for t in ALL_TAPS:
        # OLD weak number: fresh Q per structure, held-out residues
        ps = []
        for g in store:
            X, Y = ctr(g[t]["id"].astype(np.float64)), ctr(g[t]["R1"].astype(np.float64))
            m = len(X); p = rng.permutation(m); f, e = p[:m // 2], p[m // 2:]
            ps.append(_med_cos(X[e] @ _fit_q(X[f], Y[f]), Y[e]))
        # POOLED: one Q_R over train structures, scored on held-out structures
        Q, sc_in, sc_ho = {}, {}, {}
        for name in ROTS:
            Xtr = np.concatenate([ctr(store[k][t]["id"].astype(np.float64)) for k in tr])
            Ytr = np.concatenate([ctr(store[k][t][name].astype(np.float64)) for k in tr])
            Q[name] = _fit_q(Xtr, Ytr)
            sc_in[name] = _med_cos(Xtr @ Q[name], Ytr)
            Xho = np.concatenate([ctr(store[k][t]["id"].astype(np.float64)) for k in ho])
            Yho = np.concatenate([ctr(store[k][t][name].astype(np.float64)) for k in ho])
            sc_ho[name] = _med_cos(Xho @ Q[name], Yho)
        # HOMOMORPHISM: does Q_{R1} Q_{R2} reproduce Q_{R1R2}?
        Xho = np.concatenate([ctr(store[k][t]["id"].astype(np.float64)) for k in ho])
        Y12 = np.concatenate([ctr(store[k][t]["R1R2"].astype(np.float64)) for k in ho])
        homo = _med_cos(Xho @ Q["R1"] @ Q["R2"], Y12)
        C_t = store[0][t]["id"].shape[1]
        res[t] = {"channels": int(C_t),
                  "train_samples_per_channel": float(n_train_samples / C_t),
                  "rank_ok": bool(n_train_samples >= 2 * C_t),
                  "per_struct": float(np.median(ps)),
                  "pooled_in": {k: float(v) for k, v in sc_in.items()},
                  "pooled_ho": {k: float(v) for k, v in sc_ho.items()},
                  "pooled_ho_median": float(np.median(list(sc_ho.values()))),
                  "homo_Q1Q2_vs_R1R2": float(homo),
                  "direct_R1R2_ho": float(sc_ho["R1R2"])}
        print(f"{t:16s} {np.median(ps):11.3f} "
              f"{np.median(list(sc_in.values())):10.3f} "
              f"{np.median(list(sc_ho.values())):10.3f} {homo:8.3f}", flush=True)

    res["_meta"] = {"n_structures": n, "n_fit": len(tr), "n_heldout": len(ho),
                    "n_res": args.n_res, "timestep": args.timestep,
                    "coupled": bool(args.couple), "rotations": list(ROTS),
                    "note": ("per_struct = the old weak number (fresh Q per structure). "
                             "pooled_ho is the real quantity: one Q_R shared across "
                             "structures, scored on structures never used to fit it. "
                             "homo compares Q_R1 Q_R2 against the R1R2 features.")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")
    print("If pooled_ho << per_struct, Q depends on the STRUCTURE and there is no "
          "single representation to decompose -- O4 Tasks 2-6 collapse.")


if __name__ == "__main__":
    main()
