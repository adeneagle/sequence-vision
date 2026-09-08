"""Does imposed rotation invariance destroy task signal? Frame-averaging vs single frame.

Tests a claim I made and could not support: that rotation augmentation would force a
head to discard ~70% of the per-residue feature content. That number came from the
POINTWISE pose cosine (~0.29 at up_blocks[1]) -- and this project separately showed
pointwise cosine understates preserved information by up to 10x (cos 0.064 -> held-out
Procrustes 0.820). So the bound was quoted in the wrong metric.

The quantity that actually matters is how much *task* signal survives invariance, so
measure that directly: run the O1 benchmark (secondary structure, amino-acid identity,
real maps, cluster split) on features averaged over K of the 24 octahedral rotations.

  K=1   single frame (no imposed invariance)
  K=24  exactly invariant to the octahedral group, by construction

Averaging is CUMULATIVE over one set of 24 forwards, so K=1,4,8,24 all come from the
same compute -- giving the saturation curve, not just the endpoints. Cube rotations are
transpose+flip, hence lossless: no interpolation error is charged to invariance.

READOUT DETAIL THAT MATTERS. The box is 64^3 (even), so the Ca sits at continuous index
31.5 -- between voxels. A flip maps cell k -> d-1-k, so reading the single cell `d//2`
samples half a cell off the Ca, in a direction that ROTATES WITH THE FRAME. Reading the
mean of the central 2^3 feature cells is flip-symmetric by construction, so every arm
(including K=1) samples the same symmetric neighbourhood and the only variable is the
number of rotations averaged.

Interpretation:
  * accuracy roughly flat in K  -> invariance is nearly free; my objection was wrong and
    a rotation-robust head (learned or averaged) costs little.
  * accuracy falls sharply in K -> that drop is the hard ceiling for ANY rotation-robust
    readout at this tap, and rotation-invariant alignment is expensive.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

AA1 = "ARNDCQEGHILKMFPSTWYV"
KS = (1, 4, 8, 24)


def torch_cube_rotate(x: torch.Tensor, R: np.ndarray) -> torch.Tensor:
    """Torch twin of teachers.cryofm_tap.rotate_volume for [B, C, D, H, W]."""
    perm = [int(np.argmax(np.abs(R[a]))) for a in range(3)]
    y = x.permute(0, 1, 2 + perm[0], 2 + perm[1], 2 + perm[2])
    flips = [2 + a for a in range(3) if R[a, perm[a]] < 0]
    return torch.flip(y, dims=flips) if flips else y


@torch.no_grad()
def centre2(tap, boxes, timestep, batch, tapname, noise_level=None,
            noise_seed: int = 0, legacy_noise: bool = False):
    """Forward boxes; return the mean of the CENTRAL 2^3 feature cells (flip-symmetric).

    @torch.no_grad() is LOAD-BEARING: the original `centre_features` carries it, and
    dropping it here made all 24 forwards per chain retain autograd graphs -> 76 GB and
    CUDA OOM. Copying a verified function's body without its decorator is the bug.

    NOISE SEEDING (fixed 2026-08-31). The generator is created ONCE, outside the batch
    loop, matching `centre_features` (local_frame_stability.py:141). It used to be
    re-seeded to 0 *inside* the loop, which made every batch draw the IDENTICAL noise
    field: residues sharing a within-batch index got the same eps, and K repeated
    "draws" were not independent at all -- so noise-averaging removed the same field
    every time and looked like signal. Any coupled-arm number predating this fix is
    suspect. `legacy_noise=True` reproduces the old behaviour byte-for-byte so old
    results can be regenerated deliberately rather than by accident.
    """
    from teachers.cryofm_tap import StopForward

    out = []
    gen = torch.Generator(device="cpu").manual_seed(noise_seed)
    for s in range(0, len(boxes), batch):
        x = boxes[s:s + batch].to(tap.device)
        if noise_level:
            g = torch.Generator(device="cpu").manual_seed(noise_seed) if legacy_noise else gen
            eps = torch.randn(x.shape, generator=g).to(x.device)
            x = (1.0 - noise_level) * x + noise_level * eps
        x = torch.cat([x, torch.zeros_like(x)], dim=1)
        t = torch.full((x.shape[0],), timestep, device=tap.device, dtype=torch.long)
        tap._buf.clear()
        try:
            tap.model(x, timestep=t)
        except StopForward:
            # The tap may carry `stop_after`, which skips the ~half of the network
            # past the deepest tap by RAISING from the capture hook. Any code that
            # calls `tap.model(...)` directly -- rather than through
            # `CryoFM2Tap.forward_taps` -- must catch it. No-op when stop_after is
            # None, since then nothing raises.
            pass
        f = tap._buf[tapname]                     # [b, C, d, d, d]
        d = f.shape[-1]; c = d // 2
        out.append(f[:, :, c - 1:c + 1, c - 1:c + 1, c - 1:c + 1]
                   .mean(dim=(2, 3, 4)).cpu().numpy())
        tap._buf.clear()
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--tap", default="up_blocks[1]")
    ap.add_argument("--timestep", type=int, default=261)
    ap.add_argument("--couple", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--per-chain", type=int, default=20)
    ap.add_argument("--box-chunk", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--feat-dir", type=Path, default=Path("data/o4_fa_parts"))
    ap.add_argument("--cache", type=Path, default=Path("data/o4_frameavg.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/o4_frameavg.json"))
    ap.add_argument("--rotation", choices=["cube", "so3"], default="cube",
                    help="cube = the 24 proper cube rotations, lossless (transpose+flip). "
                         "so3 = generic SO(3). For so3 the box is cut DIRECTLY in the rotated "
                         "frame (R @ fr) rather than cut-then-rotated, so every arm including "
                         "K=1 pays exactly ONE trilinear resample -- otherwise interpolation "
                         "error would be charged to invariance.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-seed", type=int, default=0,
                    help="seed for the coupled-arm noise draw (see centre2)")
    ap.add_argument("--legacy-noise", action="store_true",
                    help="reproduce the pre-2026-08-31 per-batch re-seeding bug, for "
                         "regenerating old numbers byte-for-byte")
    args = ap.parse_args()

    keys = tuple(f"K{k}" for k in KS) + ("raw", "aa", "ss", "split", "cluster")
    if args.cache.exists():
        print(f"loading {args.cache}")
        z = np.load(args.cache, allow_pickle=True); F = {k: z[k] for k in keys}
    else:
        from teachers.cryofm_tap import (MODEL_VOXEL_SIZE, PATCH, CryoFM2Tap,
                                         cube_rotations, preprocess)
        from probes.local_frame_stability import extract_local_boxes
        from probes.stability import load_map
        from probes.homolog_diagnostic_residue import chain_backbone
        from probes.o1_cryofm_benchmark import backbone_with_resnum, ss_labels, to_cubic_even

        rows = list(csv.DictReader(open(args.chains)))[: args.limit]
        rows.sort(key=lambda r: r["emd"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if args.rotation == "cube":
            ROTS = cube_rotations()
        else:
            # A FIXED set shared across all residues (a deployable frame set), with
            # R_0 = identity so the K=1 arm is the canonical local frame -- the same
            # baseline definition as the cube run, which makes the two comparable.
            from probes.local_frame_stability import random_so3
            _r = np.random.default_rng(args.seed)
            ROTS = [np.eye(3)] + [random_so3(_r) for _ in range(max(KS) - 1)]
        args.feat_dir.mkdir(parents=True, exist_ok=True)
        nl = (args.timestep / 1000.0) if args.couple else None
        print(f"{len(rows)} chains | tap={args.tap} t={args.timestep} "
              f"{'coupled' if args.couple else 'decoupled'} | {len(ROTS)} rotations | {dev}",
              flush=True)
        _t = {}
        def tap():
            if not _t:
                _t["m"] = CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                                     batch_size=args.batch_size)
            return _t["m"]

        rng = np.random.default_rng(args.seed)
        cache_key = cache_vol = None; done = skipped = 0
        for i, r in enumerate(rows):
            part = args.feat_dir / f"{r['key']}.npz"
            if part.exists():
                done += 1; continue
            try:
                d = Path(r["pdb"]).parent
                if cache_key != d.name:
                    vol, vs, origin = load_map(str(d / f"{d.name}_raw_emd.map"))
                    cache_vol = (preprocess(to_cubic_even(vol), vs), origin); cache_key = d.name
                norm, origin = cache_vol
                seq, ca, fr, nums = backbone_with_resnum(r["pdb"], r["chain"])
                if seq != chain_backbone(r["pdb"], r["chain"])[0] or seq != r["seq"]:
                    raise ValueError("sequence mismatch")
                coords = (ca - np.asarray(origin)[None]) / MODEL_VOXEL_SIZE
                shape = np.array(norm.shape)
                ok = np.all((coords >= PATCH // 2) & (coords < shape[None] - PATCH // 2), axis=1)
                ssl = ss_labels(d)
                aa = np.array([AA1.index(c) if c in AA1 else -1 for c in seq])
                ss = np.array([ssl.get((r["chain"], int(n)), -1) for n in nums])
                ok &= (aa >= 0) & (ss >= 0)
                idx = np.nonzero(ok)[0]
                if len(idx) < 8:
                    raise ValueError(f"only {len(idx)} usable residues")
                if len(idx) > args.per_chain:
                    idx = rng.choice(idx, args.per_chain, replace=False)
                acc = None; cum = {}
                if args.rotation == "cube":
                    boxes = extract_local_boxes(torch.from_numpy(norm), coords[idx], fr[idx],
                                                device=dev, chunk=args.box_chunk)
                    for k, R in enumerate(ROTS):
                        f = centre2(tap(), torch_cube_rotate(boxes, R), args.timestep,
                                    args.batch_size, args.tap, noise_level=nl,
                                    noise_seed=args.noise_seed,
                                    legacy_noise=args.legacy_noise)
                        acc = f if acc is None else acc + f
                        if (k + 1) in KS:
                            cum[f"K{k+1}"] = (acc / (k + 1)).astype(np.float32)
                else:
                    vt = torch.from_numpy(norm)
                    for k, R in enumerate(ROTS):
                        # ONE resample per arm: cut in the rotated frame, never cut-then-rotate.
                        frk = np.einsum("ij,njk->nik", R, fr[idx])
                        bk = extract_local_boxes(vt, coords[idx], frk, device=dev,
                                                 chunk=args.box_chunk)
                        f = centre2(tap(), bk, args.timestep, args.batch_size,
                                    args.tap, noise_level=nl,
                                    noise_seed=args.noise_seed,
                                    legacy_noise=args.legacy_noise)
                        if k == 0:
                            boxes = bk            # K=1 arm supplies the raw-voxel control
                        else:
                            del bk
                        acc = f if acc is None else acc + f
                        if (k + 1) in KS:
                            cum[f"K{k+1}"] = (acc / (k + 1)).astype(np.float32)
                c = PATCH // 2
                raw = boxes[:, 0, c - 4:c + 4, c - 4:c + 4, c - 4:c + 4] \
                        .reshape(len(idx), -1).float().cpu().numpy()
                del boxes
                if dev == "cuda":
                    torch.cuda.empty_cache()
                np.savez(part, raw=raw.astype(np.float32), aa=aa[idx], ss=ss[idx],
                         split=np.array([r["split"]] * len(idx)),
                         cluster=np.array([r["cluster"]] * len(idx)), **cum)
                done += 1
            except Exception as exc:
                skipped += 1
                if skipped <= 10:
                    print(f"  SKIP {r['key']} {type(exc).__name__}: {exc}", flush=True)
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(rows)}] {done} ok, {skipped} skipped", flush=True)
        print(f"chains: {done} ok, {skipped} skipped", flush=True)
        parts = [args.feat_dir / f"{r['key']}.npz" for r in rows]
        parts = [p for p in parts if p.exists()]
        buf = {k: [] for k in keys}
        for p in parts:
            z = np.load(p, allow_pickle=True)
            for k in keys: buf[k].append(z[k])
        F = {k: np.concatenate(v) for k, v in buf.items()}
        np.savez(args.cache, **F); print(f"cached -> {args.cache}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    sp = F["split"].astype(str); tr, te = sp == "train", sp == "test"
    if tr.sum() == 0 or te.sum() == 0:
        raise SystemExit(f"empty split: train={tr.sum()} test={te.sum()}")
    print(f"\nresidues: train {tr.sum()}  test {te.sum()}  "
          f"({len(np.unique(F['cluster'][tr]))}/{len(np.unique(F['cluster'][te]))} clusters)")
    res = {}
    for task in ("ss", "aa"):
        y = F[task]; prior = float(np.bincount(y[te]).max() / te.sum())
        res[task] = {"prior": prior}
        print(f"\n=== {task} ({int(y.max())+1} classes, prior {prior:.4f}) ===")
        print(f"  {'arm':10s} {'dim':>5s} {'acc':>8s}   {'Δ vs K=1':>9s}")
        base = None
        for arm in [f"K{k}" for k in KS] + ["raw"]:
            X = F[arm]; sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
            a = float(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
            if arm == "K1": base = a
            res[task][arm] = {"acc": a, "dim": int(X.shape[1])}
            dd = "" if base is None or arm == "raw" else f"{a-base:+9.4f}"
            print(f"  {arm:10s} {X.shape[1]:5d} {a:8.4f}   {dd}", flush=True)
    res["_meta"] = {"tap": args.tap, "timestep": args.timestep, "coupled": bool(args.couple),
                    "rotation": args.rotation,
                    "Ks": list(KS), "n_train": int(tr.sum()), "n_test": int(te.sum()),
                    "readout": "mean of central 2^3 feature cells (flip-symmetric)",
                    "question": "how much TASK signal survives imposed octahedral invariance"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
