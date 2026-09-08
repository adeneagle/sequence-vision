"""Recover each cached voxel's 3D coordinate WITHOUT re-running the encoder.

The voxel cache stores features and targets but not positions, and token mixing
(§8.18.3) needs positions. A full rebuild is ~19.5 h of GPU; this is ~minutes of
CPU, because `sample_voxels(pool="model")` is a pure function of
(volume SHAPE, atom coordinates, seeded RNG) -- it never reads the volume's
VALUES. That is also why every low-pass cache came out with byte-identical
`res_idx`: the geometry does not depend on the density at all.

CORRECTNESS IS ASSERTED, NOT ASSUMED. Recomputed coordinates are pushed back
through `soft_targets` and the result must reproduce the cached `res_idx`
EXACTLY and `weight` to 0.0 absolute error. If the sampler or its defaults ever
drift, this fails loudly instead of silently attaching wrong positions to
features -- which would look like a modelling result rather than a bug.

COORDINATES ARE IN THE ROTATED FRAME, matching the features. Storing them in a
canonical (unrotated) frame would make any spatial module rotation-invariant by
construction and would not be deployable: at inference the map arrives in an
arbitrary frame. The cost is that voxels from different rotations of the same
map are in DIFFERENT frames, so a mixing model must draw each batch from a
single rotation (`--rot-per-batch` in the trainer).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def recover(emd: str, args) -> dict:
    from probes.build_voxel_cache import rotations_for
    from probes.voxel_sampler import sample_voxels, soft_targets
    from teachers.cryofm_tap import MODEL_VOXEL_SIZE, rotate_coords

    e = int(emd)
    vol = np.load(args.vol_dir / f"{e:04d}.npy", mmap_mode="r")   # header only
    meta = json.loads((args.vol_dir / f"{e:04d}.json").read_text())
    origin = np.asarray(meta["origin_zyx"])
    inv = dict(np.load(args.inv_dir / f"{emd}.npz", allow_pickle=True))
    atom_vox = (inv["atom_xyz"].astype(np.float64) - origin[None]) / MODEL_VOXEL_SIZE

    C, RI, W, ROT = [], [], [], []
    for k, R in enumerate(rotations_for(emd, args.n_rot)):
        rng = np.random.default_rng(args.seed * 1000003 + e * 97 + k)
        if k == 0:
            av, shape = atom_vox, vol.shape
        else:
            av = rotate_coords(atom_vox, R, vol.shape)
            shape = tuple(np.abs(R).astype(int) @ np.array(vol.shape))
        inv_k = dict(inv)
        inv_k["atom_xyz"] = (av * MODEL_VOXEL_SIZE + origin[None]).astype(np.float32)
        c, _ = sample_voxels(np.empty(shape, dtype=np.float32), args.n_vox, rng,
                             pool="model", atom_vox=av, bg_frac=args.bg_frac,
                             min_sep=args.min_sep)
        ri, w, _ = soft_targets(c, inv_k, origin, sigma=args.sigma,
                                n_max=args.n_max)
        C.append(c); RI.append(ri); W.append(w)
        ROT.append(np.full(len(c), k, dtype=np.int16))

    xyz = np.concatenate(C).astype(np.float32) * MODEL_VOXEL_SIZE
    return {"xyz": xyz, "rot": np.concatenate(ROT),
            "_ri": np.concatenate(RI), "_w": np.concatenate(W)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/voxel_cache"))
    ap.add_argument("--vol-dir", type=Path, default=Path("data/cleandift_vols"))
    ap.add_argument("--inv-dir", type=Path, default=Path("data/map_chains"))
    ap.add_argument("--map-chains", type=Path, default=Path("data/map_chains.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/voxel_coords"))
    # These MUST match the cache build or the assertion below will fire.
    ap.add_argument("--n-vox", type=int, default=3000)
    ap.add_argument("--n-rot", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--bg-frac", type=float, default=0.15)
    ap.add_argument("--min-sep", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1,
                    help="run N copies with --shard 0..N-1; each writes a "
                         "disjoint stride, so they cannot race on a file")
    args = ap.parse_args()

    keys = sorted({r["emd"] for r in csv.DictReader(open(args.map_chains))},
                  key=int)
    if args.limit:
        keys = keys[: args.limit]
    keys = keys[args.shard::args.nshards]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = cached = 0
    t0 = time.time()
    for i, emd in enumerate(keys):
        out = args.out_dir / f"{emd}.npz"
        src = args.cache / f"{emd}.npz"
        if out.exists():
            cached += 1
            continue
        if not src.exists():
            skipped += 1
            continue
        try:
            r = recover(emd, args)
            d = np.load(src, allow_pickle=True)
            assert np.array_equal(d["res_idx"], r["_ri"]), (
                f"{emd}: recovered res_idx differs from the cache -- the sampler "
                "or its defaults have drifted; coordinates would be WRONG")
            assert np.abs(d["weight"] - r["_w"]).max() == 0.0, \
                f"{emd}: recovered weights differ from the cache"
            assert np.array_equal(d["rot"], r["rot"]), f"{emd}: rotation mismatch"
            np.savez_compressed(out, xyz=r["xyz"], rot=r["rot"])
            done += 1
        except Exception as exc:                              # noqa: BLE001
            print(f"  {emd}: {type(exc).__name__}: {exc}", flush=True)
            skipped += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(keys)}] done {done} cached {cached} "
                  f"skipped {skipped} | {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"done {done} | cached {cached} | skipped {skipped} | "
          f"{(time.time()-t0)/60:.1f} min -> {args.out_dir}")


if __name__ == "__main__":
    main()
