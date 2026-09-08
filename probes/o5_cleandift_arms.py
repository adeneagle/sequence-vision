"""STEP 8 -- the arms evaluation, ALL arms in ONE process on IDENTICAL residues.

Structurally a copy of `probes/o4_lab_arms.py`, which already computes every arm
from one `extract_local_boxes` output into a single per-chain `.npz`, so arms
share residues exactly. That is not a nicety: the ~1.1-point run-to-run wobble
measured on 2026-08-30 (`aligned` read 0.7799 then 0.7685 on the same 395 chains,
because the residue subsample is drawn from a fresh rng) makes cross-process
comparison invalid outright.

Two changes beyond the arm list. The VAL split is carried (`o4_lab_arms` uses only
train/test and drops val), because `dec_best`'s t* and `cou_best`'s t must be
selected on val and test must be read exactly once. And every comparison is a
PAIRED cluster bootstrap (`probes/o5_stats.py`), never two independent binomial
SEs.

`dec_ens` is the arm that matters most for interpretation and it costs a
concatenate: benefit 1 of this whole line is "consolidate features across the
noise schedule", and three decoupled extractions do exactly that with no
training. If the student merely matches `dec_ens`, CleanDIFT bought nothing an
ensemble does not. `dec_ens_pca` is there because `dec_ens` carries 768 dims
against the student's 256 and this project has been burned by unmatched-dimension
comparisons before.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEC_TS = (261, 500, 750, 900)
ENS_TS = (261, 500, 750)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--vol-dir", type=Path, default=Path("data/cleandift_vols"))
    ap.add_argument("--tap", default="up_blocks[1]")
    ap.add_argument("--cou-t", type=int, default=500,
                    help="coupled arm timestep (the COUPLED sweep's own optimum)")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--per-chain", type=int, default=60)
    ap.add_argument("--box-chunk", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--students", default="",
                    help="comma-separated name=path.pt (e.g. "
                         "student=data/cleandift_runs/distill_.../final.pt)")
    ap.add_argument("--feat-dir", type=Path, default=Path("data/o5_arm_parts"))
    ap.add_argument("--out", type=Path, default=Path("results/o5_cleandift_arms.json"))
    ap.add_argument("--bar", type=float, default=0.014)
    ap.add_argument("--ctrl-bar", type=float, default=0.007)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from probes.o4_frameavg_benchmark import centre2
    from probes.o5_boxes import chain_boxes
    from teachers.cryofm_tap import CryoFM2Tap

    students = {}
    for spec in filter(None, args.students.split(",")):
        name, _, path = spec.partition("=")
        students[name] = path
    dec_arms = [f"dec_t{t}" for t in DEC_TS]
    arms = dec_arms + ["cou_best", "cou_seed1", "raw", "rand_student"] \
        + sorted(students)
    keys = arms + ["aa", "ss", "split", "cluster"]

    rows = list(csv.DictReader(open(args.chains)))[: args.limit]
    rows.sort(key=lambda r: r["emd"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.feat_dir.mkdir(parents=True, exist_ok=True)
    _cache: dict = {}

    def get_tap(kind: str):
        if kind not in _cache:
            if kind == "teacher":
                m = CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                               batch_size=args.batch_size, stop_after=args.tap)
            elif kind == "rand":
                m = CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                               batch_size=args.batch_size, stop_after=args.tap,
                               random_weights=True)
            else:
                m = CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                               batch_size=args.batch_size, stop_after=args.tap,
                               student_ckpt=students[kind])
            _cache[kind] = m
        return _cache[kind]

    print(f"{len(rows)} chains | tap={args.tap} | arms {arms} | {dev}", flush=True)
    rng = np.random.default_rng(args.seed)
    done = skipped = 0
    for i, r in enumerate(rows):
        part = args.feat_dir / f"{r['key']}.npz"
        if part.exists():
            z = np.load(part, allow_pickle=True)
            if not set(keys) - set(z.files):
                done += 1
                continue
            # Cache hazard: `if exists: continue` would skip a chain whose .npz
            # predates a newly added arm, and the aggregation loop then raises
            # KeyError. Re-extract instead of skipping.
            part.unlink()
        try:
            boxes, aa, ss, _ = chain_boxes(r, args.vol_dir, args.per_chain, rng,
                                           dev, args.box_chunk)
            n = len(aa)
            out = {}
            for t in DEC_TS:                    # decoupled: clean input, t asserted
                out[f"dec_t{t}"] = centre2(get_tap("teacher"), boxes, t,
                                           args.batch_size, args.tap,
                                           noise_level=None).astype(np.float32)
            for nm, sd in (("cou_best", 0), ("cou_seed1", 1)):
                out[nm] = centre2(get_tap("teacher"), boxes, args.cou_t,
                                  args.batch_size, args.tap,
                                  noise_level=args.cou_t / 1000.0,
                                  noise_seed=sd).astype(np.float32)
            out["rand_student"] = centre2(get_tap("rand"), boxes, 0,
                                          args.batch_size, args.tap,
                                          noise_level=None).astype(np.float32)
            for nm in students:                 # student: clean input, no t
                out[nm] = centre2(get_tap(nm), boxes, 0, args.batch_size,
                                  args.tap, noise_level=None).astype(np.float32)
            from teachers.cryofm_tap import PATCH
            c = PATCH // 2
            out["raw"] = boxes[:, 0, c - 4:c + 4, c - 4:c + 4, c - 4:c + 4] \
                .reshape(n, -1).float().cpu().numpy().astype(np.float32)
            del boxes
            if dev == "cuda":
                torch.cuda.empty_cache()
            np.savez(part, aa=aa, ss=ss, split=np.array([r["split"]] * n),
                     cluster=np.array([r["cluster"]] * n), **out)
            done += 1
        except Exception as exc:
            skipped += 1
            if skipped <= 10:
                print(f"  SKIP {r['key']} {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] {done} ok, {skipped} skipped", flush=True)
    print(f"chains: {done} ok, {skipped} skipped", flush=True)
    # A systematic fault must not look like a successful run. The volume-cache
    # script exited 0 after skipping all 1,147 maps because "never raise out of
    # the loop" swallowed an identical error every iteration. Assert the outcome.
    if done == 0:
        raise SystemExit("FAILED: 0 chains extracted. See the SKIP lines above.")
    if skipped > 0.5 * len(rows):
        raise SystemExit(f"FAILED: {skipped}/{len(rows)} chains skipped -- that is a "
                         f"systematic fault, not bad luck with individual entries.")

    # --- aggregate ---------------------------------------------------------
    parts = sorted(args.feat_dir.glob("*.npz"))
    buf: dict[str, list] = {k: [] for k in keys}
    for p in parts:
        z = np.load(p, allow_pickle=True)
        miss = set(keys) - set(z.files)
        if miss:
            raise KeyError(f"{p} lacks {sorted(miss)}; delete and re-extract")
        for k in keys:
            buf[k].append(z[k])
    F = {k: np.concatenate(v) for k, v in buf.items()}
    n_res = len(F["aa"])
    # Arms parity: identical residues, in identical order, for every arm.
    for a in arms:
        assert len(F[a]) == n_res, f"{a} has {len(F[a])} rows, expected {n_res}"

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from probes.o5_stats import cluster_bootstrap_diff, mcnemar_exact, mdi

    sp = F["split"].astype(str)
    tr, va, te = sp == "train", sp == "val", sp == "test"
    print(f"residues: train {tr.sum()} val {va.sum()} test {te.sum()} | "
          f"test clusters {len(np.unique(F['cluster'][te]))}", flush=True)
    # Fail loudly rather than emitting NaN accuracies: with a small --limit the
    # val or test split can be empty, and `dec_best` is SELECTED on val, so an
    # empty val would silently pick an arbitrary arm.
    for nm, msk in (("train", tr), ("val", va), ("test", te)):
        if msk.sum() == 0:
            raise SystemExit(
                f"empty {nm} split ({len(parts)} chains). `dec_best` is selected on "
                f"val and test is read once, so all three are required. Raise --limit.")
    n_cl_te = len(np.unique(F["cluster"][te]))
    if n_cl_te < 30:
        print(f"  WARNING: only {n_cl_te} test clusters. Step 0b measured MDE ~0.027 "
              f"at 47 clusters, so a null result here is inconclusive.", flush=True)
    F["dec_ens"] = np.concatenate([F[f"dec_t{t}"] for t in ENS_TS], axis=1)
    pca = PCA(n_components=min(256, F["dec_ens"].shape[1]),
              random_state=0).fit(F["dec_ens"][tr])
    F["dec_ens_pca"] = pca.transform(F["dec_ens"]).astype(np.float32)
    all_arms = arms + ["dec_ens", "dec_ens_pca"]

    res: dict = {"_meta": {"tap": args.tap, "cou_t": args.cou_t,
                           "n_chains": len(parts), "n_residues": int(n_res),
                           "per_chain": args.per_chain,
                           "n_test_clusters": int(len(np.unique(F["cluster"][te]))),
                           "students": students, "bar": args.bar}}
    for task in ("ss", "aa"):
        y = F[task]
        ok, acc, accv = {}, {}, {}
        for a in all_arms:
            X = F[a]
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(
                sc.transform(X[tr]), y[tr])
            ok[a] = clf.predict(sc.transform(X[te])) == y[te]
            acc[a] = float(ok[a].mean())
            accv[a] = float((clf.predict(sc.transform(X[va])) == y[va]).mean())
        # t* and the coupled t are selected on VAL; test is read once.
        dec_best = max(dec_arms, key=lambda a: accv[a])
        clusters = F["cluster"].astype(str)[te]
        print(f"\n=== {task} (prior {np.bincount(y[te]).max()/te.sum():.4f}) ===")
        for a in all_arms:
            print(f"  {a:14s} test {acc[a]:.4f}  val {accv[a]:.4f}  "
                  f"d={F[a].shape[1]:4d}"
                  f"{'   <- dec_best (val-selected)' if a == dec_best else ''}")
        # "best t" has three defensible readings and they are NOT interchangeable:
        #   dec_best   - best DECOUPLED arm chosen on VAL. The honest, deployable
        #                baseline, and what the pre-registered gate uses.
        #   dec_oracle - best DECOUPLED arm by TEST accuracy. An ORACLE: it cannot be
        #                selected without peeking, so beating it is a strictly harder
        #                and stricter claim. Reported because "beats the best single
        #                timestep" is the paper's claim in its strongest form.
        #   cou_best   - best COUPLED arm (noised input). This is the analogue of the
        #                paper's actual DIFT baseline; our decoupled arms are an
        #                off-manifold trick that does not appear in the paper at all.
        dec_oracle = max(dec_arms, key=lambda a: acc[a])
        block_extra = {"dec_oracle": dec_oracle}
        pairs = [(s, dec_best) for s in students] \
            + [(s, dec_oracle) for s in students] \
            + [(s, "cou_best") for s in students] \
            + [("cou_best", "cou_seed1"), ("dec_ens_pca", dec_best)] \
            + [(s, "student_ctrl") for s in students if s != "student_ctrl"] \
            + [(s, "raw") for s in students] + [(dec_best, "raw")]
        block = {"acc": acc, "acc_val": accv, "dec_best": dec_best,
                 "dec_oracle": dec_oracle,
                 "dim": {a: int(F[a].shape[1]) for a in all_arms}, "pairs": {}}
        for a, b in pairs:
            if a not in ok or b not in ok:
                continue
            d = cluster_bootstrap_diff(ok[a], ok[b], clusters)
            d["mde_95"] = mdi(d)
            d["mcnemar"] = mcnemar_exact(ok[a], ok[b])
            block["pairs"][f"{a}-{b}"] = d
            print(f"  {a:14s} - {b:14s} {d['diff']:+.4f} "
                  f"CI [{d['lo95']:+.4f},{d['hi95']:+.4f}] MDE {d['mde_95']:.4f}"
                  f"{'  *' if d['excludes_zero'] else ''}")
        res[task] = block

    # --- the pre-registered gate, evaluated in code ------------------------
    ss = res["ss"]
    P = ss["pairs"]
    db = ss["dec_best"]
    g: dict = {}
    if "student" in students:
        c1 = P.get(f"student-{db}")
        g["1_beats_dec_best"] = bool(c1 and c1["diff"] >= args.bar
                                     and c1["excludes_zero"])
        null = P.get("cou_best-cou_seed1")
        g["2_null_pair_valid"] = bool(null and abs(null["diff"]) < 0.005
                                      and not null["excludes_zero"])
        c3 = P.get("student-student_ctrl")
        g["3_beats_ctrl"] = bool(c3 and c3["diff"] >= args.ctrl_bar
                                 and c3["excludes_zero"])
        g["4_beats_raw"] = bool(ss["acc"]["student"] > ss["acc"]["raw"])
        g["PASS"] = all(g.get(k, False) for k in
                        ("1_beats_dec_best", "2_null_pair_valid",
                         "3_beats_ctrl", "4_beats_raw"))
        # The t-ensemble only "matches" the student if it is itself a REAL gain of
        # comparable size. An earlier version tested |student - ens| < bar, which
        # fires whenever the two are merely CLOSE -- including when the ensemble
        # gains nothing at all, i.e. exactly the case where it does NOT match.
        ens = P.get("dec_ens_pca-" + db)
        g["kill_matched_by_t_ensemble"] = bool(
            ens and c1 and ens["excludes_zero"] and ens["diff"] > 0
            and abs(c1["diff"] - ens["diff"]) < 0.5 * abs(c1["diff"]))
        print("\n=== GATE ===")
        for k, v in g.items():
            print(f"  {k}: {v}")
        if c1 and c1["mde_95"] > args.bar:
            print(f"  NOTE MDE {c1['mde_95']:.4f} exceeds the bar {args.bar}: "
                  "a null result here is inconclusive, not negative.")
    res["_gate"] = g
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
