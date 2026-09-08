"""Does the RELATIONSHIP between residues survive rotation? (not each vector)

Per-residue invariance is the wrong question for a spatially-informative
feature. If rotating the input maps every feature by one consistent transform,
`f_i(B) ~ Q f_i(A)`, then per-residue cosine collapses while EVERY pairwise
relationship is intact -- and pairwise relationships are what a spatial
representation is for. `up_blocks[0]` scores 0.094 per-residue in the global
framing; that number cannot distinguish "destroyed" from "rigidly re-expressed".

Five metrics, in increasing tolerance to nuisance transforms:

  cos        median per-residue centred cosine. Strict identity. (what we had)
  procrustes same, AFTER fitting the single best ORTHOGONAL map A->B. The gap
             cos -> procrustes is precisely the part of the apparent loss that
             is a global rotation of feature space rather than damage.
  cka        linear CKA between the two feature sets. Invariant to any
             orthogonal transform and isotropic scaling: the standard measure of
             "same representational geometry".
  rsa        Spearman correlation between the two residue x residue similarity
             matrices (off-diagonal). Rank-based, so also robust to monotone
             distortion.
  knn        mean overlap of each residue's top-k feature neighbours across
             poses. The most operational: are the same residues still close?

And one that asks whether the geometry is SPATIAL at all, not merely stable:

  spatial    Spearman corr between feature distance and 3D distance, per pose.
             A feature can be perfectly pose-stable and carry no geometry;
             this separates the two. Reported per pose plus the across-pose gap.

Global framing throughout (rotate the molecule, forward the whole map, sample at
the new coordinates) -- the local frame already has high per-residue invariance
by construction, so it has nothing to hide.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from data.simulate_density import simulate
from probes.homolog_diagnostic_residue import chain_backbone
from probes.layer_rotation_invariance import ALL_TAPS, _coords, _sim
from probes.local_frame_stability import random_so3
from probes.stability import centred_cosine
from teachers.cryofm_tap import (
    MODEL_VOXEL_SIZE,
    PATCH,
    CryoFM2Tap,
    preprocess,
    sample_at,
)


def _centre(X):
    return X - X.mean(0, keepdims=True)


def linear_cka(X, Y):
    """CKA on centred features: invariant to orthogonal transforms + scaling."""
    X, Y = _centre(X.astype(np.float64)), _centre(Y.astype(np.float64))
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / max(den, 1e-12))


def procrustes_cos(X, Y, rng=None, held_out=True):
    """Median per-residue cosine after the best ORTHOGONAL map X -> Y.

    HELD OUT BY DEFAULT, and that is not optional. Q is C x C (512 x 512 at
    up_blocks[0]) fitted from ~60 residues, so an in-sample Procrustes score is
    wildly overparameterised and can be near-perfect on pure noise. Fitting on
    half the residues and scoring the other half is what distinguishes "the
    rotation is a genuine global re-expression" from "512^2 free parameters can
    map any 60 points onto any other 60 points".
    """
    Xc, Yc = _centre(X.astype(np.float64)), _centre(Y.astype(np.float64))
    n = len(Xc)
    if held_out and n >= 8:
        rng = rng or np.random.default_rng(0)
        perm = rng.permutation(n)
        fit, ev = perm[: n // 2], perm[n // 2:]
    else:
        fit = ev = np.arange(n)
    U, _, Vt = np.linalg.svd(Xc[fit].T @ Yc[fit], full_matrices=False)
    Q = U @ Vt
    XQ = Xc[ev] @ Q
    num = (XQ * Yc[ev]).sum(1)
    den = np.linalg.norm(XQ, axis=1) * np.linalg.norm(Yc[ev], axis=1)
    return float(np.median(num / np.maximum(den, 1e-12)))


def _sim_mat(X):
    Xc = _centre(X.astype(np.float64))
    Xc /= np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
    return Xc @ Xc.T


def rsa(X, Y):
    """Spearman between the two residue x residue similarity matrices."""
    A, B = _sim_mat(X), _sim_mat(Y)
    iu = np.triu_indices(len(A), k=1)
    return float(spearmanr(A[iu], B[iu]).statistic)


def knn_overlap(X, Y, k=10):
    A, B = _sim_mat(X), _sim_mat(Y)
    np.fill_diagonal(A, -np.inf)
    np.fill_diagonal(B, -np.inf)
    ka = np.argsort(-A, axis=1)[:, :k]
    kb = np.argsort(-B, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ka, kb)]))


def spatial_fidelity(X, xyz):
    """Spearman(-feature similarity, 3D distance): is the geometry encoded at all?"""
    S = _sim_mat(X)
    D = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    iu = np.triu_indices(len(S), k=1)
    return float(spearmanr(-S[iu], D[iu]).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--n-rot", type=int, default=3)
    ap.add_argument("--n-res", type=int, default=60)
    ap.add_argument("--min-chain-len", type=int, default=0,
                    help="lower bound on chain length. Set it EQUAL across an "
                         "n_res sweep, or each point silently draws a different "
                         "structure population (n_res=400 would see only large "
                         "chains) and you measure protein size, not saturation.")
    ap.add_argument("--max-chain-len", type=int, default=400,
                    help="upper bound on chain length; raise it for large --n-res, "
                         "or the residue budget silently caps at the chain size")
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--d-min", type=float, default=3.0)
    ap.add_argument("--couple", action="store_true",
                    help="tie input noise to the timestep (noise_level = t/1000), the TRAINED\n                          pairing. Default off = clean input. NOTE every relational number\n                          logged before 2026-08-27 is DECOUPLED, and since the two arms are\n                          near-orthogonal at matched t, decoupled results do NOT transfer.")
    ap.add_argument("--timestep", type=int, default=10,
                    help="flow-matching operating point. Default 10 = 1%% noise, below the\n                          bottom of SD's schedule; every relational number logged before\n                          2026-08-27 is a t=10 slice.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path,
                    default=Path("results/layer_relational_invariance.json"))
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.chains))
            if max(80, args.min_chain_len, args.n_res) <= int(r["n_obs"])
            <= args.max_chain_len]
    rows = rows[: args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tap = CryoFM2Tap(args.ckpt, taps=ALL_TAPS, device=dev, batch_size=args.batch_size)
    rng = np.random.default_rng(0)
    margin = PATCH // 2 * MODEL_VOXEL_SIZE
    print(f"{len(rows)} structures x {args.n_rot} SO(3) rotations, "
          f"{args.n_res} residues | k={args.knn} | {dev}")

    acc = {t: {m: [] for m in ("cos", "procrustes", "procrustes_insample", "cka",
                               "rsa", "knn", "spatialA", "spatialB")}
           for t in ALL_TAPS}
    for i, r in enumerate(rows):
        try:
            seq, ca, fr = chain_backbone(r["pdb"], r["chain"])
            volA, oA, spA = _sim(r["pdb"], r["chain"], None, margin, args.d_min)
            cA, _ = _coords(ca, fr, r["pdb"], r["chain"], None, oA, spA)
            sh = np.array(volA.shape)
            keep = np.where(np.all((cA >= PATCH // 2)
                                   & (cA < sh[None] - PATCH // 2), axis=1))[0]
            if len(keep) < 25:
                continue
            if len(keep) > args.n_res:
                keep = rng.choice(keep, args.n_res, replace=False)
            xyz = ca[keep][:, ::-1]                      # zyx -> xyz, unrotated
            gA = {t: sample_at(fv, cA[keep]) for t, fv in
                  tap.feature_volumes(preprocess(volA, MODEL_VOXEL_SIZE),
                                      timestep=args.timestep,
                                      noise_level=(args.timestep/1000.0) if args.couple else 0.0).items()}

            for _ in range(args.n_rot):
                R = random_so3(rng)
                volB, oB, spB = _sim(r["pdb"], r["chain"], R, margin, args.d_min)
                cB, _ = _coords(ca, fr, r["pdb"], r["chain"], R, oB, spB)
                shB = np.array(volB.shape)
                if not np.all(np.all((cB[keep] >= PATCH // 2)
                                     & (cB[keep] < shB[None] - PATCH // 2), axis=1)):
                    continue
                gB = {t: sample_at(fv, cB[keep]) for t, fv in
                      tap.feature_volumes(preprocess(volB, MODEL_VOXEL_SIZE),
                                          timestep=args.timestep,
                                          noise_level=(args.timestep/1000.0) if args.couple else 0.0).items()}
                for t in ALL_TAPS:
                    A, B = gA[t], gB[t]
                    a = acc[t]
                    a["cos"].append(float(np.nanmedian(centred_cosine(A, B))))
                    a["procrustes"].append(procrustes_cos(A, B, rng))
                    a["procrustes_insample"].append(
                        procrustes_cos(A, B, rng, held_out=False))
                    a["cka"].append(linear_cka(A, B))
                    a["rsa"].append(rsa(A, B))
                    a["knn"].append(knn_overlap(A, B, args.knn))
                    a["spatialA"].append(spatial_fidelity(A, xyz))
                    a["spatialB"].append(spatial_fidelity(B, xyz))
            print(f"  [{i+1}/{len(rows)}] {r['key']} {len(keep)} residues")
        except Exception as exc:
            print(f"  [{i+1}/{len(rows)}] SKIP {r['key']} {type(exc).__name__}: {exc}")

    res = {t: {m: float(np.nanmedian(v)) for m, v in d.items() if v}
           for t, d in acc.items() if d["cos"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 88)
    print(f"{'tap':<17}{'cos':>8}{'proc(ho)':>10}{'proc(in)':>10}{'CKA':>8}"
          f"{'RSA':>8}{'kNN':>8}{'spatA':>8}{'spatB':>8}")
    print("-" * 88)
    for t in ALL_TAPS:
        if t not in res:
            continue
        v = res[t]
        print(f"{t:<17}{v['cos']:>8.3f}{v['procrustes']:>10.3f}"
              f"{v['procrustes_insample']:>10.3f}{v['cka']:>8.3f}"
              f"{v['rsa']:>8.3f}{v['knn']:>8.3f}{v['spatialA']:>8.3f}"
              f"{v['spatialB']:>8.3f}")
    print("=" * 88)
    print("proc(ho) = HELD-OUT Procrustes: fit Q on half the residues, score the")
    print("           other half. THE number. proc(in) is in-sample and is shown")
    print("           only to expose the overfitting gap -- Q is 512x512 fitted")
    print("           from ~60 points, so in-sample is near-meaningless.")
    print("cos -> proc(ho) : the part of the apparent loss that is a global")
    print("                  ROTATION of feature space, harmless for relations")
    print("CKA / RSA / kNN   : is the representational GEOMETRY preserved?")
    print("spatialA/B        : does feature similarity track 3D distance at all")
    print("                    (a stable feature carrying no geometry is useless)")


if __name__ == "__main__":
    main()
