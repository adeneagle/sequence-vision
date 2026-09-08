"""THE MODEL-FREE ARM: is the backbone FRAME needed, or only the residue's location?

`o4_frame_arms.py` applied `frk = R @ fr[idx]` -- it rotated the CONVENTION of an already
residue-anchored backbone frame. `R @ F_i` is still tied to residue i's N-CA-C geometry, so every
arm there was pose-invariant by construction and every arm still required an atomic model. It
answered "does the axis convention matter" (no), NOT "can the frame be dropped".

This probe removes the frame. `frk` does not depend on the residue at all:

  aligned            frk = fr[i]                the canonical backbone frame (paired baseline,
                                                recomputed here so the comparison is on the SAME
                                                residues, not joined across runs)
  lab                frk = I                    axis-aligned box in the map's own deposited frame
  lab2 / lab3        frk = R_1 / R_2            two fixed global orientations, so the spread over
                                                "which orientation the map happens to be in" shows
  labavg4 / labavg8  mean over R_1..R_k         deployable frame averaging, no atomic model

SCOPE -- read this before quoting the result. The box CENTRE is still the CA coordinate, so these
arms are frame-free, not fully model-free. That is the substantive half: the circularity logged in
CLAUDE.md is specifically about the N-CA-C frame. Localisation is a strictly weaker dependency and
has a measured model-free substitute (blurred-density local maxima, 0.63-0.77 one-to-one
repeatability under generic SO(3)); frame orientation had none that worked.

Reading, pre-committed:
  * lab ~= aligned            -> the frame was never doing anything. Drop it, and with it the
                                 atomic-model dependency on orientation. Strongest outcome.
  * lab < aligned, labavg     -> averaging over orientations recovers it; K~4-8 is the
    recovers it                  inference cost of going model-free.
  * lab < aligned, averaging  -> the frame is load-bearing; the atomic model stays.
    does not recover it
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

AA1 = "ARNDCQEGHILKMFPSTWYV"
AVG_KS = (4, 8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--tap", default="up_blocks[1]")
    ap.add_argument("--timestep", type=int, default=261)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--per-chain", type=int, default=20)
    ap.add_argument("--box-chunk", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-rand", type=int, default=8)
    ap.add_argument("--feat-dir", type=Path, default=Path("data/o4_lab_parts"))
    ap.add_argument("--out", type=Path, default=Path("results/o4_lab_arms.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    N_RAND = args.n_rand
    avg_ks = tuple(k for k in AVG_KS if k <= N_RAND)
    ARMS = ("aligned", "lab", "lab2", "lab3") + tuple(f"labavg{k}" for k in avg_ks)
    keys = ARMS + ("raw", "aa", "ss", "split", "cluster")

    from teachers.cryofm_tap import MODEL_VOXEL_SIZE, PATCH, CryoFM2Tap, preprocess
    from probes.local_frame_stability import extract_local_boxes, random_so3
    from probes.stability import load_map
    from probes.homolog_diagnostic_residue import chain_backbone
    from probes.o1_cryofm_benchmark import backbone_with_resnum, ss_labels, to_cubic_even
    from probes.o4_frameavg_benchmark import centre2

    rows = list(csv.DictReader(open(args.chains)))[: args.limit]
    rows.sort(key=lambda r: r["emd"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Same seed/sequence as o4_frame_arms, so the global orientations are the SAME rotations.
    _r = np.random.default_rng(args.seed)
    RANDS = [random_so3(_r) for _ in range(N_RAND)]
    args.feat_dir.mkdir(parents=True, exist_ok=True)
    nl = args.timestep / 1000.0                       # coupled

    _t: dict = {}

    def tap():
        if not _t:
            _t["m"] = CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                                 batch_size=args.batch_size)
        return _t["m"]

    print(f"{len(rows)} chains | tap={args.tap} t={args.timestep} coupled | "
          f"arms: aligned + lab(I) + {N_RAND} global orientations | {dev}", flush=True)

    rng = np.random.default_rng(args.seed)
    cache_key = cache_vol = None
    done = skipped = 0
    for i, r in enumerate(rows):
        part = args.feat_dir / f"{r['key']}.npz"
        if part.exists():
            done += 1
            continue
        try:
            d = Path(r["pdb"]).parent
            if cache_key != d.name:
                vol, vs, origin = load_map(str(d / f"{d.name}_raw_emd.map"))
                cache_vol = (preprocess(to_cubic_even(vol), vs), origin)
                cache_key = d.name
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
            vt = torch.from_numpy(norm)
            n = len(idx)

            def run(frames):
                bk = extract_local_boxes(vt, coords[idx], frames, device=dev,
                                         chunk=args.box_chunk)
                f = centre2(tap(), bk, args.timestep, args.batch_size, args.tap,
                            noise_level=nl)
                return f.astype(np.float32), bk

            out = {}
            # paired baseline: the residue's own backbone frame
            fa, ba = run(fr[idx])
            out["aligned"] = fa
            # frame-free: a single global orientation, identical for every residue
            fl, _ = run(np.broadcast_to(np.eye(3), (n, 3, 3)).copy())
            out["lab"] = fl
            L = []
            for R in RANDS:
                fk, _ = run(np.broadcast_to(R, (n, 3, 3)).copy())
                L.append(fk)
            out["lab2"], out["lab3"] = L[0], L[1]
            S = np.stack(L)
            for k in avg_ks:
                out[f"labavg{k}"] = S[:k].mean(0).astype(np.float32)
            c = PATCH // 2
            raw = ba[:, 0, c - 4:c + 4, c - 4:c + 4, c - 4:c + 4] \
                    .reshape(n, -1).float().cpu().numpy()
            del ba
            if dev == "cuda":
                torch.cuda.empty_cache()
            np.savez(part, raw=raw.astype(np.float32), aa=aa[idx], ss=ss[idx],
                     split=np.array([r["split"]] * n),
                     cluster=np.array([r["cluster"]] * n), **out)
            done += 1
        except Exception as exc:
            skipped += 1
            if skipped <= 10:
                print(f"  SKIP {r['key']} {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] {done} ok, {skipped} skipped", flush=True)
    print(f"chains: {done} ok, {skipped} skipped", flush=True)

    parts = [args.feat_dir / f"{r['key']}.npz" for r in rows]
    parts = [p for p in parts if p.exists()]
    buf = {k: [] for k in keys}
    for p in parts:
        z = np.load(p, allow_pickle=True)
        for k in keys:
            buf[k].append(z[k])
    F = {k: np.concatenate(v) for k, v in buf.items()}

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    sp = F["split"].astype(str)
    tr, te = sp == "train", sp == "test"
    print(f"\nresidues: train {tr.sum()}  test {te.sum()}", flush=True)
    if te.sum() == 0 or tr.sum() == 0:
        raise SystemExit("empty train or test split -- too few chains to hold out a "
                         "cluster; raise --limit")
    res = {}
    for task in ("ss", "aa"):
        y = F[task]
        prior = float(np.bincount(y[te]).max() / te.sum())
        res[task] = {"prior": prior}
        print(f"\n=== {task} ({int(y.max())+1} classes, prior {prior:.4f}) ===", flush=True)
        base = None
        for arm in ARMS + ("raw",):
            X = F[arm]
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
            a = float(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
            if arm == "aligned":
                base = a
            # binomial SE, so a reader cannot mistake a noise-level gap for a result
            se = float(np.sqrt(a * (1 - a) / te.sum()))
            res[task][arm] = {"acc": a, "se": se, "dim": int(X.shape[1])}
            dd = "" if base is None or arm == "raw" else f"  {a-base:+.4f} vs aligned"
            print(f"  {arm:10s} {a:.4f} +-{se:.4f}  d={X.shape[1]:4d}{dd}", flush=True)
    res["_meta"] = {
        "tap": args.tap, "timestep": args.timestep, "coupled": True,
        "n_train": int(tr.sum()), "n_test": int(te.sum()), "n_chains": len(parts),
        "note": "FRAME-FREE arms: frk does NOT depend on the residue. Box CENTRE is still the CA "
                "coord, so these are frame-free, not fully model-free. Supersedes o4_frame_arms, "
                "which rotated the convention of the backbone frame (R @ fr) and so never removed "
                "the atomic-model dependency.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
