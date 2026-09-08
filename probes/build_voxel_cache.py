"""Cache per-voxel CleanDIFT-student features + soft residue targets, per map.

THE POINT OF CACHING. Both towers are frozen and the student consumes a CLEAN
input at a single LEARNED timestep, so its features are a pure function of
(map, rotation) -- there is no noise draw and no `t` to choose. Precomputing them
turns DINO.txt training into a ~1-2 M parameter head over cached tensors: minutes
per run instead of hours, and dozens of ablations become affordable. That is the
LiT regime dino.txt itself exploits.

Budget: ~2-5k voxels/map x 256 ch x fp16 x 1,147 maps ~ 1.5-3 GB per rotation.
Caching whole feature VOLUMES instead would be ~134 MB/map (154 GB for one
rotation) and is not worth it.

ROTATIONS ARE THE AUGMENTATION, AND THEY ARE CUBE ROTATIONS -- read this before
quoting any invariance number off this cache. `rotate_volume` implements ONLY the
octahedral group: it picks `argmax|R[a]|` per row, so handing it a generic SO(3)
matrix silently snaps to the nearest axis permutation while `rotate_coords`
applies the true `R`, and the volume and the coordinates then disagree. Lossless
transpose+flip is also the only rotation that costs nothing and adds no
interpolation blur.

The cost: the octahedral group is CryoFM2's OWN pretraining augmentation
(`RotCube24`, p=1.0), so it is the easy case, and this cache CANNOT be used to
measure generic-SO(3) invariance -- that needs a real resampling path. For
AUGMENTATION the distinction looks unimportant (G2b: cube -0.006 SS vs generic
SO(3) -0.005), but the two claims are different and only the first is supported
here.

fp16 IS SAFE HERE, unlike in `build_vol_cache`. There the concern was that fp16's
~5e-4 relative error would be a confound for sub-1% effects on the PREPROCESSED
INPUT, which then propagates through the whole network. These are output features
consumed by a head that immediately standardises them; the same argument does not
apply, and 2x the cache for 5e-4 is a bad trade at this size.

WHAT THIS DOES *NOT* DO: choose voxels model-free. See `voxel_sampler`'s module
docstring -- the density criterion was measured to select solvent on real maps
(fraction on-protein 0.05-1.00 across 8 maps), so the default pool is
model-defined. The atomic model is a labelling instrument here, exactly as it is
for the target; making the model-free pool work is the open blocker on
inference-time deployment and is deliberately not on the G0 path.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch


def rotations_for(emd: str, k: int) -> list:
    """K distinct CUBE rotations for a map, identity first.

    Identity first so K=1 reproduces the unrotated lab frame exactly, making the
    no-augmentation arm a strict subset rather than a different sample. Seeded per
    map so a requeued job regenerates the SAME rotations -- otherwise a preempted
    run would silently mix two augmentation sets into one cache.
    """
    from teachers.cryofm_tap import cube_rotations

    allr = cube_rotations()
    ident = next(j for j, R in enumerate(allr) if np.allclose(R, np.eye(3)))
    if k <= 1:
        return [allr[ident]]
    others = [R for j, R in enumerate(allr) if j != ident]
    rng = np.random.default_rng(int(emd) * 7919)
    pick = rng.choice(len(others), min(k - 1, len(others)), replace=False)
    return [allr[ident]] + [others[j] for j in pick]


def lowpass(vol: np.ndarray, res_a: float, voxel_a: float,
            edge: float = 0.1) -> np.ndarray:
    """Soft-edge Fourier low-pass to `res_a` Angstrom, then RE-NORMALISE.

    G4 asks what E1 does at 4-10 A, the resolution band where the training-free
    rivals fail (ModelAngelo per-residue identification: 49% top-1 at 4-5 A,
    **0%** at 5-10 A). That band is also where CryoFM2 is most out of
    distribution, so this arm is a genuine test, not a formality.

    Two implementation points that would otherwise confound the result:
      * RE-NORMALISATION IS REQUIRED. Filtering removes high-frequency power, so
        the 99.999th percentile drops and an unrenormalised volume is not just
        lower-resolution but systematically DIMMER -- the arm would then measure
        contrast loss as if it were resolution loss. The CryoFM affine is undone,
        the filter applied, and the affine re-derived on the filtered volume,
        which is what a genuinely lower-resolution deposited map would look like.
      * A SPHERICALLY SYMMETRIC filter COMMUTES EXACTLY with the octahedral
        rotations used as augmentation, so filtering once before the rotation
        loop is equivalent to filtering each rotation and is not an ordering bug.

    `edge` is the raised-cosine transition width as a fraction of the cutoff; a
    hard cutoff would ring (Gibbs) and put structure where there is none.
    """
    from teachers.cryofm_tap import CRYOEM_DENSITY_MEAN, CRYOEM_DENSITY_STD
    raw = vol.astype(np.float32) * CRYOEM_DENSITY_STD + CRYOEM_DENSITY_MEAN
    kc = voxel_a / float(res_a)                       # cycles/voxel at cutoff
    if kc >= 0.5:
        raise SystemExit(
            f"--lowpass {res_a} A is at or beyond Nyquist for {voxel_a} A/voxel "
            f"(kc={kc:.3f}); nothing would be filtered")
    ks = [np.fft.fftfreq(n) for n in raw.shape[:-1]] + \
         [np.fft.rfftfreq(raw.shape[-1])]
    k = np.sqrt(sum(x.reshape([-1 if i == j else 1 for j in range(3)]) ** 2
                    for i, x in enumerate(ks)))
    lo, hi = kc * (1 - edge), kc * (1 + edge)
    m = np.clip((hi - k) / (hi - lo), 0.0, 1.0)
    m = 0.5 * (1 - np.cos(np.pi * m))                 # raised cosine
    out = np.fft.irfftn(np.fft.rfftn(raw) * m, s=raw.shape).astype(np.float32)
    q = np.percentile(out, 99.999)
    if q <= 0:
        raise ValueError(f"non-positive 99.999th percentile after low-pass ({q})")
    return ((out / q) - CRYOEM_DENSITY_MEAN) / CRYOEM_DENSITY_STD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-chains", type=Path, default=Path("data/map_chains.csv"))
    ap.add_argument("--inv-dir", type=Path, default=Path("data/map_chains"))
    ap.add_argument("--vol-dir", type=Path, default=Path("data/cleandift_vols"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/voxel_cache"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--student",
                    default="data/cleandift_runs/distill_t1000_s0_paperhead/best.pt")
    ap.add_argument("--teacher", action="store_true",
                    help="frozen CryoFM2 instead of the student (control arm)")
    ap.add_argument("--teacher-t", type=int, default=750)
    ap.add_argument("--random-weights", action="store_true",
                    help="untrained network: the mandatory floor arm (P4)")
    ap.add_argument("--tap", default="up_blocks[1]")
    ap.add_argument("--lowpass", type=float, default=0.0,
                    help="G4: low-pass volumes to this resolution (A) before "
                         "the forward pass. 0 = off.")
    ap.add_argument("--n-vox", type=int, default=3000, help="foreground per rotation")
    ap.add_argument("--n-rot", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--bg-frac", type=float, default=0.15)
    ap.add_argument("--min-sep", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from probes.voxel_sampler import chain_quota, sample_voxels, soft_targets
    from teachers.cryofm_tap import MODEL_VOXEL_SIZE, CryoFM2Tap, sample_at

    rows = list(csv.DictReader(open(args.map_chains)))
    maps: dict[str, dict] = {}
    for r in rows:
        maps.setdefault(r["emd"], {"split": r["split"], "chains": []})["chains"].append(r)
    keys = sorted(maps, key=int)
    if args.limit:
        keys = keys[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tap = CryoFM2Tap(
        args.ckpt, taps=(args.tap,), device=dev, batch_size=args.batch_size,
        stop_after=args.tap, random_weights=args.random_weights,
        student_ckpt=None if (args.teacher or args.random_weights) else args.student,
    )
    arm = ("random" if args.random_weights
           else "teacher" if args.teacher else "student")
    print(f"{len(keys)} maps | tap={args.tap} | arm={arm} | K={args.n_rot} | "
          f"{args.n_vox} fg/rot | {dev}", flush=True)

    done = cached = skipped = 0
    t0 = time.time()
    for i, emd in enumerate(keys):
        out = args.out_dir / f"{emd}.npz"
        if out.exists():
            cached += 1
            continue
        try:
            e = int(emd)
            vol = np.asarray(np.load(args.vol_dir / f"{e:04d}.npy", mmap_mode="r"))
            meta = json.loads((args.vol_dir / f"{e:04d}.json").read_text())
            if args.lowpass:
                vol = lowpass(vol, args.lowpass, MODEL_VOXEL_SIZE)
            origin = np.asarray(meta["origin_zyx"])
            inv = dict(np.load(args.inv_dir / f"{emd}.npz", allow_pickle=True))
            atom_vox = (inv["atom_xyz"].astype(np.float64) - origin[None]) / MODEL_VOXEL_SIZE

            F, RI, W, WB, BG, Q, ROT = [], [], [], [], [], [], []
            for k, R in enumerate(rotations_for(emd, args.n_rot)):
                rng = np.random.default_rng(args.seed * 1000003 + e * 97 + k)
                if k == 0:
                    v, av = vol, atom_vox
                else:
                    from teachers.cryofm_tap import rotate_coords, rotate_volume
                    # Guard the silent-snap failure described in the docstring.
                    assert np.allclose(np.abs(R) @ np.ones(3), 1.0) and \
                        np.allclose(np.abs(R).sum(0), 1.0) and \
                        np.allclose(np.abs(R), np.round(np.abs(R))), (
                        f"R is not a signed permutation matrix:\n{R}\n"
                        "rotate_volume would snap it to the nearest axis "
                        "permutation while rotate_coords applies the true R")
                    v = rotate_volume(vol, R)
                    av = rotate_coords(atom_vox, R, vol.shape)
                inv_k = dict(inv)
                # soft_targets works in Angstroms about `origin`; feed it a
                # rotated atom set expressed the same way.
                inv_k["atom_xyz"] = (av * MODEL_VOXEL_SIZE + origin[None]).astype(np.float32)

                c, bg = sample_voxels(v, args.n_vox, rng, pool="model",
                                      atom_vox=av, bg_frac=args.bg_frac,
                                      min_sep=args.min_sep)
                ri, w, wb = soft_targets(c, inv_k, origin, sigma=args.sigma,
                                         n_max=args.n_max)
                q = chain_quota(c, inv_k, origin)
                fv = tap.feature_volumes(v, timestep=args.teacher_t)[args.tap]
                f = sample_at(fv, c).astype(np.float16)
                del fv
                F.append(f); RI.append(ri); W.append(w); WB.append(wb)
                BG.append(bg); Q.append(q)
                ROT.append(np.full(len(c), k, dtype=np.int16))

            tmp = out.with_suffix(".tmp.npz")
            with open(tmp, "wb") as fh:
                np.savez_compressed(
                    fh,
                    feat=np.concatenate(F), res_idx=np.concatenate(RI),
                    weight=np.concatenate(W), w_bg=np.concatenate(WB),
                    is_bg=np.concatenate(BG), quota=np.concatenate(Q),
                    rot=np.concatenate(ROT),
                    res_seq=inv["res_seq"], res_pos=inv["res_pos"],
                    seqs=inv["seqs"], split=np.array(maps[emd]["split"]),
                    allow_pickle=True,
                )
            tmp.replace(out)
            done += 1
        except Exception as exc:
            skipped += 1
            if skipped <= 10:
                print(f"  SKIP {emd} {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 25 == 0:
            el = (time.time() - t0) / 60
            print(f"  [{i+1}/{len(keys)}] {done} new, {cached} cached, "
                  f"{skipped} skipped, {el:.1f} min", flush=True)

    tot = sum(p.stat().st_size for p in args.out_dir.glob("*.npz"))
    print(f"\nmaps: {done} new, {cached} cached, {skipped} skipped | "
          f"{tot/2**30:.2f} GiB | {(time.time()-t0)/60:.1f} min -> {args.out_dir}")
    if done + cached == 0:
        raise SystemExit("FAILED: 0 maps cached.")
    if skipped > 0.5 * len(keys):
        raise SystemExit(f"FAILED: {skipped}/{len(keys)} skipped -- systematic.")


if __name__ == "__main__":
    main()
