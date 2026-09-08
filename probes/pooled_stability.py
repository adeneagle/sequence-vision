"""Does pooling to a STRUCTURE-level descriptor rescue the pose problem?

Phase 0a found per-residue CryoFM2 features are only 31-51% pose-invariant, which
caps R^2 for any sequence model. Hypothesis: the orientation/grid noise is roughly
independent across residues, so mean+std pooling over ~1000 residues averages it
away at ~sqrt(N) while shared structural signal survives -- taking a whole-map
descriptor to >0.95 pose-invariance.

This measures that directly, and adds the control that matters for the downstream
task: cross-map retrieval, WITH a length-matched variant. Map size correlates with
sequence length, so "match this map to its sequence" can be largely solved by size
alone. Unless retrieval survives length matching, it is measuring nothing.

    pixi run python probes/pooled_stability.py --limit 12 --n-rot 4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from probes.stability import load_ca, load_map
from teachers.cryofm_tap import (
    MODEL_VOXEL_SIZE,
    CryoFM2Tap,
    cube_rotations,
    preprocess,
    rotate_coords,
    rotate_volume,
    sample_at,
)


def pool(feat: np.ndarray) -> np.ndarray:
    """mean+std pool over residues -> one descriptor per map."""
    return np.concatenate([feat.mean(0), feat.std(0)])


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--timestep", type=int, default=10)
    ap.add_argument("--n-rot", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("results/pooled_stability.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))[: args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tap = CryoFM2Tap(args.ckpt, device=dev, batch_size=args.batch_size)
    rots = cube_rotations()[1 : args.n_rot + 1]
    print(f"device={dev} maps={len(rows)} rotations={args.n_rot}")

    # per tap: reference descriptor per map, and rotated descriptors per map
    ref: dict[str, list] = {}
    rot: dict[str, list] = {}
    nres, sizes, ids = [], [], []

    for i, row in enumerate(rows):
        try:
            vol, vs, origin_A = load_map(row["map_path"])
            ca = load_ca(row["cif_path"])
            norm = preprocess(vol, vs)
            coords = (ca - origin_A[None]) / MODEL_VOXEL_SIZE
            shape = np.array(norm.shape)
            keep = np.all((coords >= 1) & (coords < shape[None] - 2), axis=1)
            coords = coords[keep]
            if len(coords) < 30:
                continue
            idx = np.rint(coords).astype(int)
            if norm[idx[:, 0], idx[:, 1], idx[:, 2]].mean() - norm.mean() < 0.5:
                continue

            base = tap.feature_volumes(norm, timestep=args.timestep, noise_seed=0)
            for k, fv in base.items():
                ref.setdefault(k, []).append(pool(sample_at(fv, coords)))

            per_rot: dict[str, list] = {}
            for R in rots:
                fvs = tap.feature_volumes(rotate_volume(norm, R), timestep=args.timestep,
                                          noise_seed=0)
                rc = rotate_coords(coords, R, norm.shape)
                for k, fv in fvs.items():
                    per_rot.setdefault(k, []).append(pool(sample_at(fv, rc)))
            for k, v in per_rot.items():
                rot.setdefault(k, []).append(v)

            nres.append(len(coords))
            sizes.append(int(row["box"]))
            ids.append(row["emdb_id"])
            print(f"[{i+1}/{len(rows)}] {row['emdb_id']} {len(coords)} residues")
        except Exception as exc:
            print(f"[{i+1}/{len(rows)}] {row['emdb_id']} FAILED {type(exc).__name__}: {exc}")

    nres = np.array(nres)
    out = {}
    for k in ref:
        Rf = np.stack(ref[k])                       # [M, 2C]
        M = len(Rf)
        # --- pose invariance of the POOLED descriptor
        pose = [cos(Rf[m], rot[k][m][r]) for m in range(M) for r in range(len(rots))]

        # --- cross-map retrieval: does a rotated map still match its own reference?
        hits, hits_lenmatched = [], []
        for m in range(M):
            for r in range(len(rots)):
                q = rot[k][m][r]
                sims = np.array([cos(q, Rf[j]) for j in range(M)])
                hits.append(int(np.argmax(sims) == m))
                # length-matched pool: only compete against maps of similar residue count
                near = np.where(np.abs(np.log(nres / nres[m])) < 0.2)[0]
                if len(near) > 2:
                    hits_lenmatched.append(int(near[np.argmax(sims[near])] == m))

        # --- how much of the pooled descriptor is just size?
        size_corr = float(np.corrcoef(np.linalg.norm(Rf - Rf.mean(0), axis=1), nres)[0, 1])

        out[k] = {
            "n_maps": int(M),
            "pose_pooled_median": float(np.median(pose)),
            "pose_pooled_p10": float(np.percentile(pose, 10)),
            "retrieval_top1": float(np.mean(hits)),
            "retrieval_top1_lenmatched": float(np.mean(hits_lenmatched)) if hits_lenmatched else None,
            "chance": 1.0 / M,
            "descriptor_norm_vs_nres_corr": size_corr,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"maps": ids, "n_residues": nres.tolist(), "results": out},
              open(args.out, "w"), indent=2)

    print("\n" + "=" * 78)
    print(f"{'tap':<16}{'pose(pooled)':>14}{'p10':>8}{'retr':>8}{'retr(len)':>11}"
          f"{'chance':>9}{'sizecorr':>10}")
    print("-" * 78)
    for k, r in out.items():
        lm = r["retrieval_top1_lenmatched"]
        print(f"{k:<16}{r['pose_pooled_median']:>14.3f}{r['pose_pooled_p10']:>8.3f}"
              f"{r['retrieval_top1']:>8.3f}{(f'{lm:.3f}' if lm is not None else 'n/a'):>11}"
              f"{r['chance']:>9.3f}{r['descriptor_norm_vs_nres_corr']:>10.3f}")
    print("=" * 78)
    print("\nCompare pose(pooled) against the PER-RESIDUE ceilings from Phase 0a:")
    print("  mid_block 0.46 | up_blocks[0] 0.31 | up_blocks[1] 0.51")
    print("A large jump confirms pooling averages away the orientation noise.")
    print("retr(len) is the number that matters -- unmatched retrieval can be size alone.")


if __name__ == "__main__":
    main()
