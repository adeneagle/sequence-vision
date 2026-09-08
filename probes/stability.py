"""Phase 0a: is a CryoFM2 activation a well-defined target for a sequence model?

Three stability measurements, each answering a question that must be YES before
any alignment training is worth attempting:

  pose    -- does the feature at a residue survive re-orienting the map?
             A sequence model has NO pose. If the feature swings with map
             orientation it is unpredictable from sequence in principle.
             THIS IS THE GATE: median pose-cosine < ~0.5 kills the track.

  noise   -- the network is v(x_t, t), so features depend on the noise draw.
             How much of the feature is signal vs. which noise we happened to
             sample?

  tiling  -- the map is cut into 64^3 patches. Does a residue's feature depend
             on where the patch boundaries happened to fall?

Cosines are computed against a per-tap mean-centred reference, because raw
high-dimensional activations share a large common component that makes any two
of them look similar (cosine ~0.99 regardless). Centring removes that offset so
the number reflects residue-specific agreement, not the shared bias.

    pixi run python probes/stability.py --manifest data/manifest.csv --limit 10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gemmi
import mrcfile
import numpy as np
import torch

from teachers.cryofm_tap import (
    MODEL_VOXEL_SIZE,
    CryoFM2Tap,
    cube_rotations,
    preprocess,
    rotate_coords,
    rotate_volume,
    sample_at,
)


def load_map(path: str) -> tuple[np.ndarray, float, np.ndarray]:
    with mrcfile.open(path, permissive=True) as m:
        vol = np.array(m.data, dtype=np.float32)  # [z, y, x]
        vs = float(m.voxel_size.x)
        org = np.array([m.header.origin.z, m.header.origin.y, m.header.origin.x], dtype=np.float64)
        nstart = np.array([m.header.nzstart, m.header.nystart, m.header.nxstart], dtype=np.float64)
    origin_A = org if np.any(org) else nstart * vs
    return vol, vs, origin_A


def load_ca(path: str, chain_id: str | None = None) -> np.ndarray:
    """Ca positions as [N, 3] in (z, y, x) order.

    `chain_id=None` takes EVERY chain of model 0. That is correct for a
    map-level measurement (pose stability compares a map against itself), but it
    is WRONG for any cross-modal pairing: the manifest carries a single chain's
    sequence, while these entries average 5.15 chains and reach 28. On the
    current manifest 14 of 27 have `n_unique_seq > 1` and only 3 are monomers, so
    an unfiltered descriptor summarises protein the paired sequence does not
    describe. Pass `chain_id` (or restrict the manifest to single-unique-sequence
    entries, where the extra chains are copies of the same sequence) before
    reporting any sequence<->density number.
    """
    st = gemmi.read_structure(path)
    st.setup_entities()
    ca = []
    for model in st:
        for chain in model:
            if chain_id is not None and chain.name != chain_id:
                continue
            for res in chain:
                a = res.find_atom("CA", "*")
                if a is not None:
                    ca.append([a.pos.z, a.pos.y, a.pos.x])
        break
    if len(ca) == 0:
        raise ValueError(f"{path}: no CA atoms"
                         + (f" for chain {chain_id!r}" if chain_id else ""))
    return np.array(ca, dtype=np.float64)


def centred_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine between two [N, C] sets after removing the shared mean."""
    mu = 0.5 * (A.mean(0, keepdims=True) + B.mean(0, keepdims=True))
    a, b = A - mu, B - mu
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-8) & (nb > 1e-8)
    out = np.full(len(A), np.nan)
    out[ok] = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
    return out


def retrieval(A: np.ndarray, B: np.ndarray, coords_A: np.ndarray | None = None,
              max_n: int = 3000) -> dict:
    """Can a residue still be identified from its feature after a perturbation?

    An absolute cosine is uninterpretable on its own -- high-dimensional activations
    share a large common component, so both matched AND mismatched pairs can score
    high. What matters is whether the matched pair is *more* similar than mismatched
    ones. For each residue i we rank all j by similarity to A[i] and ask where the
    true match i lands.

    Returns top-1 / top-5 accuracy, median percentile rank, and -- crucially -- the
    mismatched-pair cosine, which is the floor the matched cosine must beat.
    """
    n = len(A)
    if n > max_n:                                  # subsample; the matrix is n^2
        sel = np.random.default_rng(0).choice(n, max_n, replace=False)
        A, B, n = A[sel], B[sel], max_n
        if coords_A is not None:
            coords_A = coords_A[sel]
    mu = 0.5 * (A.mean(0, keepdims=True) + B.mean(0, keepdims=True))
    a = A - mu
    b = B - mu
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    S = a @ b.T                                    # [n, n] cosine matrix

    diag = np.diag(S).copy()
    off = S[~np.eye(n, dtype=bool)]
    # rank of the true match within each row (0 = best)
    rank = (S > diag[:, None]).sum(1)
    out = {
        "n": int(n),
        "matched_cos": float(np.median(diag)),
        "mismatched_cos": float(np.median(off)),
        "top1": float((rank == 0).mean()),
        "top5": float((rank < 5).mean()),
        "median_pct_rank": float(np.median(rank) / max(n - 1, 1)),
    }

    # Discriminates the two explanations for poor retrieval:
    #   spatially smooth  -> the top-1 hit is a near neighbour (small distance)
    #   grid-artifact noise -> the top-1 hit is anywhere (distance ~ random pair)
    # Without this, low top-1 is ambiguous between "coarse but real" and "noise".
    if coords_A is not None:
        best = np.argmax(S, axis=1)
        d_top1 = np.linalg.norm(coords_A - coords_A[best], axis=1) * MODEL_VOXEL_SIZE
        rng = np.random.default_rng(0)
        d_rand = np.linalg.norm(coords_A - coords_A[rng.permutation(n)], axis=1) * MODEL_VOXEL_SIZE
        out["top1_dist_A"] = float(np.median(d_top1))
        out["random_dist_A"] = float(np.median(d_rand))
    return out


def summarise(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }


def run_map(tap: CryoFM2Tap, row: dict, args, rots: list[np.ndarray]) -> dict | None:
    vol, vs, origin_A = load_map(row["map_path"])
    ca = load_ca(row["cif_path"])
    if len(ca) == 0:
        return None

    norm = preprocess(vol, vs)
    coords = (ca - origin_A[None]) / MODEL_VOXEL_SIZE
    shape = np.array(norm.shape)
    inside = np.all((coords >= 1) & (coords < shape[None] - 2), axis=1)
    if inside.sum() < 30:
        return None
    coords = coords[inside]

    # Correspondence assertion: CA must sit in density, else origin handling is wrong.
    idx = np.rint(coords).astype(int)
    contrast = float(norm[idx[:, 0], idx[:, 1], idx[:, 2]].mean() - norm.mean())
    if contrast < 0.5:
        print(f"  {row['emdb_id']}: SKIP, CA/bulk density contrast only {contrast:+.2f}")
        return None

    ref = {k: sample_at(v, coords) for k, v in
           tap.feature_volumes(norm, timestep=args.timestep, noise_seed=0).items()}

    res: dict[str, dict[str, list]] = {
        k: {"pose": [], "noise_draw": [], "noise_shift": [], "tiling": []} for k in ref
    }

    # --- pose: lossless cube rotations of the volume + matching coord rotation
    retr: dict[str, list] = {k: [] for k in ref}
    for R in rots:
        rvol = rotate_volume(norm, R)
        rcoords = rotate_coords(coords, R, norm.shape)
        fvs = tap.feature_volumes(rvol, timestep=args.timestep, noise_seed=0)
        for k, fv in fvs.items():
            rot_feat = sample_at(fv, rcoords)
            res[k]["pose"].append(centred_cosine(ref[k], rot_feat))
            retr[k].append(retrieval(ref[k], rot_feat, coords_A=coords))

    # --- noise. Two distinct quantities, previously conflated:
    #   noise_shift: clean reference vs a noisy extraction -- sensitivity to ADDING noise.
    #   noise_draw : two noisy extractions at the SAME level, different seeds -- sensitivity
    #                to WHICH noise was drawn. This is the one the gate cares about, since it
    #                bounds how reproducible the target is at a fixed operating point.
    noisy = [
        tap.feature_volumes(norm, timestep=args.timestep, noise_seed=s,
                            noise_level=args.noise_level)
        for s in range(1, args.n_noise + 2)
    ]
    for k in ref:
        base = sample_at(noisy[0][k], coords)
        for fvs in noisy[1:]:
            res[k]["noise_draw"].append(centred_cosine(base, sample_at(fvs[k], coords)))
        res[k]["noise_shift"].append(centred_cosine(ref[k], base))

    # --- tiling: shift the volume so patch boundaries land elsewhere
    for sh in args.tiling_shifts:
        svol = np.pad(norm, ((sh, 0), (sh, 0), (sh, 0)), mode="constant")
        fvs = tap.feature_volumes(svol, timestep=args.timestep, noise_seed=0)
        for k, fv in fvs.items():
            res[k]["tiling"].append(centred_cosine(ref[k], sample_at(fv, coords + sh)))

    out = {k: {m: np.concatenate(v) for m, v in d.items() if v} for k, d in res.items()}
    for k in out:
        out[k]["_retrieval"] = retr[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--timestep", type=int, default=10)
    ap.add_argument("--n-rot", type=int, default=8, help="cube rotations to test (<=24)")
    ap.add_argument("--n-noise", type=int, default=3)
    ap.add_argument("--noise-level", type=float, default=0.1)
    ap.add_argument("--tiling-shifts", type=int, nargs="*", default=[16, 32])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("results/stability.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))[: args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  maps={len(rows)}  timestep={args.timestep}  rotations={args.n_rot}")

    tap = CryoFM2Tap(args.ckpt, device=dev, batch_size=args.batch_size)
    rots = cube_rotations()[1 : args.n_rot + 1]   # skip identity

    pooled: dict[str, dict[str, list]] = {}
    for i, row in enumerate(rows):
        print(f"[{i+1}/{len(rows)}] {row['emdb_id']} ({row['resolution']}A, box {row['box']})")
        try:
            r = run_map(tap, row, args, rots)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        if r is None:
            continue
        for k, d in r.items():
            for metric, vals in d.items():
                pooled.setdefault(k, {}).setdefault(metric, []).append(vals)

    retr_summary = {}
    for k, d in pooled.items():
        recs = [x for sub in d.pop("_retrieval", []) for x in sub]
        if recs:
            retr_summary[k] = {f: float(np.mean([r[f] for r in recs]))
                               for f in ("matched_cos", "mismatched_cos", "top1", "top5",
                                         "median_pct_rank", "top1_dist_A", "random_dist_A")
                               if f in recs[0]}

    summary = {k: {m: summarise(np.concatenate(v)) for m, v in d.items()}
               for k, d in pooled.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(args) | {"manifest": str(args.manifest), "out": str(args.out)},
               "summary": summary, "retrieval": retr_summary},
              open(args.out, "w"), indent=2, default=str)

    print("\n" + "=" * 68)
    print(f"{'tap':<16}{'metric':<10}{'median':>9}{'p10':>9}{'p90':>9}{'n':>9}")
    print("-" * 68)
    for k, d in summary.items():
        for m, s in d.items():
            if s.get("n"):
                print(f"{k:<16}{m:<10}{s['median']:>9.3f}{s['p10']:>9.3f}{s['p90']:>9.3f}{s['n']:>9}")
    print("=" * 68)
    print("\nPOSE RETRIEVAL -- is a residue still identifiable from its feature after rotation?")
    print("An absolute cosine means nothing without the mismatched floor to compare against.")
    print(f"\n{'tap':<16}{'matched':>9}{'mismatch':>10}{'top1':>8}{'pctrank':>9}"
          f"{'d(top1)A':>10}{'d(rand)A':>10}")
    print("-" * 74)
    for k, r in retr_summary.items():
        print(f"{k:<16}{r['matched_cos']:>9.3f}{r['mismatched_cos']:>10.3f}"
              f"{r['top1']:>8.3f}{r['median_pct_rank']:>9.4f}"
              f"{r.get('top1_dist_A', float('nan')):>10.1f}{r.get('random_dist_A', float('nan')):>10.1f}")
    print("=" * 68)
    for k, r in retr_summary.items():
        # Identity preserved = matched clearly beats mismatched AND retrieval is far
        # better than chance. top1 is the operational criterion; the absolute cosine
        # threshold used earlier was arbitrary and is reported only for context.
        sep = r["matched_cos"] - r["mismatched_cos"]
        ok = r["top1"] >= 0.5 or r["median_pct_rank"] <= 0.01
        print(f"GATE {k:<16} sep {sep:+.3f}  top1 {r['top1']:.3f}  "
              f"{'PASS -- identity survives rotation' if ok else 'FAIL -- identity lost'}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
