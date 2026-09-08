"""P17 -- the CleanDIFT student on the WHOLE-MAP STITCHED path, which has never run.

Every CleanDIFT number in this project was measured on the **Ca-centred box** path
(`chain_boxes` -> `centre2`): a 64^3 box cut around each residue, one forward per
residue. The DINO.txt voxel objective needs the **stitched whole-map** path
(`feature_volumes`): tile the map into patches, forward each, stitch the
activations back into one feature volume, then sample at arbitrary voxels.

D10 says the student trained on a 50/50 mixture whose second half IS patch-grid
crops with cube rotations, i.e. exactly what `feature_volumes` consumes, so it
*should* be in distribution. "Should be" is not a measurement. If the student
overfit to the box path, every voxel-level number built on it would be quietly
degraded and the box-path result would give no warning.

THREE CHECKS, in increasing order of what they cost to be wrong about:

1. **Timestep is a no-op for the student.** `LearnedTimeEmb` ignores its input
   ("the student's `timestep` argument becomes a no-op"), so two different
   timesteps must give BITWISE-identical features. The teacher must differ. This
   is a paired check: it also proves the harness really is varying something.

2. **Box path vs volume path agree where they should.** Both crop 64^3 and the
   network is convolutional, so a residue whose ~10 A neighbourhood is fully
   inside both crops should get near-identical features -- differing only by
   translation within the crop and by patch-boundary effects. High agreement is
   evidence the stitching is registered correctly (this is the same property the
   patch-alignment fix restored); low agreement means a registration bug, which
   at `up_blocks[1]` (stride 2) should be impossible but is worth pinning.

3. **The substantive one: does the student's ADVANTAGE survive the path change?**
   Linear SS/AA probe on volume-path features, teacher vs student. The box path
   gave student_paperhead +1.5 SS over the best decoupled teacher timestep. If
   that inverts here, the student is path-overfit and the voxel design must use
   the teacher instead. Note the arms are NOT comparable to the published box-path
   table -- different framing, different residues -- only teacher-vs-student
   WITHIN this run is meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

AA1 = "ARNDCQEGHILKMFPSTWYV"


def sample_chain(tap, norm, origin, ca_zyx, tapname):
    """Volume-path per-residue features: stitch the whole map, sample at Ca."""
    from teachers.cryofm_tap import MODEL_VOXEL_SIZE, sample_at

    fv = tap.feature_volumes(np.asarray(norm), timestep=10)[tapname]
    coords = (ca_zyx - np.asarray(origin)[None]) / MODEL_VOXEL_SIZE
    return sample_at(fv, coords).astype(np.float32), fv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--student",
                    default="data/cleandift_runs/distill_t1000_s0_paperhead/best.pt")
    ap.add_argument("--tap", default="up_blocks[1]")
    ap.add_argument("--teacher-t", type=int, default=750,
                    help="dec_best from the CleanDIFT run (decoupled, clean input)")
    ap.add_argument("--vol-dir", type=Path, default=Path("data/cleandift_vols"))
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--per-chain", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("results/o6_volpath_verify.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from probes.o1_cryofm_benchmark import backbone_with_resnum, ss_labels
    from probes.o5_boxes import load_norm_vol
    from teachers.cryofm_tap import CryoFM2Tap

    rows = list(csv.DictReader(open(args.chains)))
    rng = np.random.default_rng(args.seed)
    if args.limit:
        rows = rows[: args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(rows)} chains | tap={args.tap} | teacher t={args.teacher_t} | {dev}",
          flush=True)

    def mk(student: bool):
        return CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                          batch_size=args.batch_size, stop_after=args.tap,
                          student_ckpt=args.student if student else None)

    teacher, stud = mk(False), mk(True)
    res: dict = {"_meta": {"tap": args.tap, "teacher_t": args.teacher_t,
                           "student": args.student, "limit": args.limit,
                           "per_chain": args.per_chain}}

    # ---------------------------------------------------------------- check 0
    # TRUNCATION EQUALITY. `feature_volumes` was patched to catch StopForward so
    # that `stop_after` works on the whole-map path (it previously raised, making
    # the path unusable with the truncation that makes it affordable). That patch
    # touches shared code every other probe imports, so prove it changes nothing:
    # running the full network and stopping after the tap must agree BITWISE.
    r0 = rows[0]
    norm0, _ = load_norm_vol(r0["pdb"], args.vol_dir)
    small = np.asarray(norm0)[:64, :64, :64]
    full = CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                      batch_size=args.batch_size)          # no stop_after
    a = full.feature_volumes(small, timestep=10)[args.tap].data
    b = teacher.feature_volumes(small, timestep=10)[args.tap].data
    trunc_equal = bool(np.array_equal(a, b))
    print(f"\n[0] stop_after truncation is bitwise-equal to the full forward: "
          f"{trunc_equal}")
    res["truncation_equal"] = trunc_equal
    assert trunc_equal, (
        "truncating at the tap changed the features -- the StopForward patch in "
        "feature_volumes is not behaviour-preserving and every probe that imports "
        "cryofm_tap is affected")
    del full

    # ---------------------------------------------------------------- check 1
    # Timestep no-op for the student, and NOT for the teacher.
    def fv_at(tp, t):
        return tp.feature_volumes(small, timestep=t)[args.tap].data
    s10, s900 = fv_at(stud, 10), fv_at(stud, 900)
    t10, t900 = fv_at(teacher, 10), fv_at(teacher, 900)
    same_student = bool(np.array_equal(s10, s900))
    teacher_delta = float(np.abs(t10 - t900).max())
    print(f"\n[1] student t=10 vs t=900 bitwise identical: {same_student}")
    print(f"    teacher  t=10 vs t=900 max|delta|:        {teacher_delta:.4g}")
    res["timestep_noop"] = {"student_identical": same_student,
                            "teacher_max_delta": teacher_delta}
    assert same_student, (
        "student features depend on `timestep` -- LearnedTimeEmb is not attached, so "
        "the checkpoint did not load through the D2 swap and the student is not the "
        "model you think it is")
    assert teacher_delta > 1e-6, (
        "teacher features do NOT depend on timestep -- the harness is not varying "
        "anything and check 1 is vacuous")

    # ------------------------------------------------------- checks 2 and 3
    from probes.o4_frameavg_benchmark import centre2
    from probes.o5_boxes import chain_boxes

    feats: dict = {"vol_teacher": [], "vol_student": [], "box_student": []}
    meta: dict = {"aa": [], "ss": [], "split": [], "cluster": []}
    done = skipped = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        try:
            norm, origin = load_norm_vol(r["pdb"], args.vol_dir)
            seq, ca, fr, nums = backbone_with_resnum(r["pdb"], r["chain"])
            if seq != r["seq"]:
                raise ValueError("sequence mismatch")
            # Same residue filter as every other probe here, so the SS/AA numbers
            # are read on comparable residues.
            from teachers.cryofm_tap import MODEL_VOXEL_SIZE, PATCH
            coords = (ca - np.asarray(origin)[None]) / MODEL_VOXEL_SIZE
            shape = np.array(norm.shape)
            ok = np.all((coords >= PATCH // 2) &
                        (coords < shape[None] - PATCH // 2), axis=1)
            ssl = ss_labels(Path(r["pdb"]).parent)
            aa = np.array([AA1.index(c) if c in AA1 else -1 for c in seq])
            ss = np.array([ssl.get((r["chain"], int(n)), -1) for n in nums])
            ok &= (aa >= 0) & (ss >= 0)
            idx = np.nonzero(ok)[0]
            if len(idx) < 8:
                raise ValueError(f"only {len(idx)} usable residues")
            if len(idx) > args.per_chain:
                idx = rng.choice(idx, args.per_chain, replace=False)
                idx.sort()

            # volume path, both models -- one at a time: a stitched up_blocks[1]
            # volume is ~2 GB fp32 for a large map and holding two is pointless.
            vt, _ = sample_chain(teacher, norm, origin, ca[idx], args.tap)
            vs, _ = sample_chain(stud, norm, origin, ca[idx], args.tap)
            # box path, student only (the published configuration)
            boxes = chain_boxes(r, args.vol_dir, args.per_chain,
                                np.random.default_rng(args.seed), dev, 16)[0]
            bs = centre2(stud, boxes, 10, args.batch_size, args.tap)
            bs = bs.float().cpu().numpy() if torch.is_tensor(bs) else np.asarray(bs)
            n = min(len(vt), len(vs), len(bs))

            feats["vol_teacher"].append(vt[:n])
            feats["vol_student"].append(vs[:n])
            feats["box_student"].append(bs[:n])
            meta["aa"].append(aa[idx][:n])
            meta["ss"].append(ss[idx][:n])
            meta["split"].append(np.array([r["split"]] * n))
            meta["cluster"].append(np.array([r["cluster"]] * n))
            done += 1
        except Exception as exc:
            skipped += 1
            if skipped <= 10:
                print(f"  SKIP {r['key']} {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(rows)}] {done} ok, {skipped} skipped, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    if done == 0:
        raise SystemExit("FAILED: 0 chains processed.")
    F = {k: np.concatenate(v) for k, v in feats.items()}
    M = {k: np.concatenate(v) for k, v in meta.items()}
    n_res = len(M["aa"])
    for k, v in F.items():
        assert len(v) == n_res, f"{k}: {len(v)} rows vs {n_res}"
    print(f"\n{done} chains, {skipped} skipped, {n_res} residues, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

    # -------- check 2: box vs volume agreement (centred cosine, per residue)
    from probes.stability import centred_cosine
    # centred_cosine takes two [N, C] matrices and returns the row-wise cosine
    # after removing the SHARED mean -- it is not a vector-pair function.
    cc = centred_cosine(F["box_student"], F["vol_student"])
    # Floor: the same comparison against a SHUFFLED partner. Without it a high
    # median is uninterpretable -- raw cosine on high-dim activations is ~0.99
    # for any pair, which is why this is centred and why the floor is reported.
    perm = np.random.default_rng(0).permutation(n_res)
    ccf = centred_cosine(F["box_student"], F["vol_student"][perm])
    cc, ccf = cc[~np.isnan(cc)], ccf[~np.isnan(ccf)]
    print(f"[2] box vs volume path, centred cosine: median {np.median(cc):.4f} "
          f"(p10 {np.percentile(cc,10):.4f}) | shuffled floor {np.median(ccf):.4f}")
    res["path_agreement"] = {"median": float(np.median(cc)),
                             "p10": float(np.percentile(cc, 10)),
                             "shuffled_floor": float(np.median(ccf))}

    # -------- check 3: does the student's advantage survive the path change?
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from probes.o5_stats import cluster_bootstrap_diff, mdi

    sp = M["split"].astype(str)
    tr, te = sp == "train", sp == "test"
    if tr.sum() == 0 or te.sum() == 0:
        raise SystemExit(f"empty split (train {tr.sum()}, test {te.sum()}); raise --limit")
    print(f"    residues: train {tr.sum()} test {te.sum()} | "
          f"test clusters {len(np.unique(M['cluster'][te]))}", flush=True)

    res["probe"] = {}
    for task in ("ss", "aa"):
        y = M[task]
        ok, acc = {}, {}
        for a in ("vol_teacher", "vol_student", "box_student"):
            sc = StandardScaler().fit(F[a][tr])
            clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(
                sc.transform(F[a][tr]), y[tr])
            ok[a] = clf.predict(sc.transform(F[a][te])) == y[te]
            acc[a] = float(ok[a].mean())
        d = cluster_bootstrap_diff(ok["vol_student"], ok["vol_teacher"],
                                   M["cluster"].astype(str)[te])
        prior = float(np.bincount(y[te]).max() / te.sum())
        print(f"\n=== {task} (prior {prior:.4f}) ===")
        for a in ("vol_teacher", "vol_student", "box_student"):
            print(f"  {a:14s} {acc[a]:.4f}")
        print(f"  student - teacher (volume path): {d['diff']:+.4f} "
              f"CI [{d['lo95']:+.4f}, {d['hi95']:+.4f}] MDE {mdi(d):.4f}")
        res["probe"][task] = {"acc": acc, "prior": prior,
                              "student_minus_teacher_vol": d}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
