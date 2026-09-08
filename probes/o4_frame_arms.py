"""Does the backbone frame do anything at all? aligned vs single-random vs random-averaged.

The earlier SO(3) run compared K=1 (the canonical N-CA-C frame) against cumulative averages
that INCLUDED that aligned view as one of 24 terms. So it measured "aligned vs averaged", and
never measured the condition the voxel-level design actually deploys: **a single arbitrary
orientation, no frame, no averaging**.

That distinction separates two very different conclusions:
  * single random ~= aligned  -> the backbone frame was never doing anything, and neither frames
    nor averaging are needed. No atomic model anywhere.
  * single random < aligned, recovered by averaging -> averaging does real work, and K~4-8 is the
    right inference cost.

Arms (all boxes cut with ONE trilinear resample, in frame R @ fr; the residue stays box-centred):
  aligned              R = I, the canonical backbone frame
  rand1/rand2/rand3    three individual random SO(3) orientations, NOT averaged. Three of them
                       so the spread across "which orientation you happen to get" is visible.
  randavg4/8/24        means over random orientations, EXCLUDING the identity -- a proper
                       Monte-Carlo Haar estimate, uncontaminated by the aligned view.

Note the random-only averages are approximately, not exactly, invariant: 24 random rotations do
not form a group (unlike the 24 octahedral ones), so this is a Haar estimate with error ~1/sqrt(K).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

AA1 = "ARNDCQEGHILKMFPSTWYV"
# ARMS is derived from --n-rand at runtime; randavg* EXCLUDE the identity so they are a
# proper Monte-Carlo Haar estimate. The rotation set is drawn from a seeded rng, so the first k
# draws are identical for any --n-rand -> parts cached at a larger n-rand stay reusable.
AVG_KS = (4, 8, 24)


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
    ap.add_argument("--feat-dir", type=Path, default=Path("data/o4_arms_parts"))
    ap.add_argument("--out", type=Path, default=Path("results/o4_frame_arms.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rand", type=int, default=8,
                    help="random orientations per residue; averaging arms use k<=n-rand")
    args = ap.parse_args()

    N_RAND = args.n_rand
    avg_ks = tuple(k for k in AVG_KS if k <= N_RAND)
    ARMS = ("aligned", "rand1", "rand2", "rand3") + tuple(f"randavg{k}" for k in avg_ks)
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
    # ONE fixed rotation set shared by every residue (a deployable frame set).
    _r = np.random.default_rng(args.seed)
    RANDS = [random_so3(_r) for _ in range(N_RAND)]
    args.feat_dir.mkdir(parents=True, exist_ok=True)
    nl = args.timestep / 1000.0                       # coupled
    print(f"{len(rows)} chains | tap={args.tap} t={args.timestep} coupled | "
          f"1 aligned + {N_RAND} random | {dev}", flush=True)
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
            vt = torch.from_numpy(norm)

            def feat(R):
                frk = np.einsum("ij,njk->nik", R, fr[idx])
                bk = extract_local_boxes(vt, coords[idx], frk, device=dev,
                                         chunk=args.box_chunk)
                f = centre2(tap(), bk, args.timestep, args.batch_size, args.tap,
                            noise_level=nl)
                return f.astype(np.float32), bk

            out = {}
            f0, b0 = feat(np.eye(3)); out["aligned"] = f0
            R_feats = []
            for k, R in enumerate(RANDS):
                fk, bk = feat(R); R_feats.append(fk); del bk
            for j in (1, 2, 3):
                out[f"rand{j}"] = R_feats[j - 1]
            S = np.stack(R_feats)
            for k in avg_ks:
                out[f"randavg{k}"] = S[:k].mean(0).astype(np.float32)
            c = PATCH // 2
            raw = b0[:, 0, c - 4:c + 4, c - 4:c + 4, c - 4:c + 4] \
                    .reshape(len(idx), -1).float().cpu().numpy()
            del b0
            if dev == "cuda":
                torch.cuda.empty_cache()
            np.savez(part, raw=raw.astype(np.float32), aa=aa[idx], ss=ss[idx],
                     split=np.array([r["split"]] * len(idx)),
                     cluster=np.array([r["cluster"]] * len(idx)), **out)
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
        for k in keys: buf[k].append(z[k])
    F = {k: np.concatenate(v) for k, v in buf.items()}

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    sp = F["split"].astype(str); tr, te = sp == "train", sp == "test"
    print(f"\nresidues: train {tr.sum()}  test {te.sum()}", flush=True)
    res = {}
    for task in ("ss", "aa"):
        y = F[task]; prior = float(np.bincount(y[te]).max() / te.sum())
        res[task] = {"prior": prior}
        print(f"\n=== {task} ({int(y.max())+1} classes, prior {prior:.4f}) ===")
        base = None
        for arm in ARMS + ("raw",):
            X = F[arm]; sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
            a = float(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
            if arm == "aligned": base = a
            res[task][arm] = {"acc": a}
            dd = "" if base is None or arm == "raw" else f"  {a-base:+.4f} vs aligned"
            print(f"  {arm:10s} {a:.4f}{dd}", flush=True)
    res["_meta"] = {"tap": args.tap, "timestep": args.timestep, "coupled": True,
                    "n_train": int(tr.sum()), "n_test": int(te.sum()),
                    "n_chains": len(parts),
                    "note": "randavg* EXCLUDE the identity (proper Haar estimate); "
                            "rand1-3 are single unaveraged orientations"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
