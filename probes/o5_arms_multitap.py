"""All-taps arms evaluation: probe EVERY distilled tap, not just one.

WHY. `cleandift_train` distils three taps (mid_block, up_blocks[0], up_blocks[1])
-- which is exactly the reference rule, "the U-Net's middle block and each of the
decoder blocks except the two final blocks", applied to CryoFM2's 4-decoder-block
UNet. But `o5_cleandift_arms.py` evaluated only `up_blocks[1]`, so two thirds of
what we trained was never measured. `up_blocks[0]` in particular is this project's
best per-residue tap by pose invariance (0.826 vs 0.706) and is the ESM-C
alignment target, so it is a live candidate to beat the published headline.

The cost argument: `CryoFM2Tap` hooks all taps in ONE forward, so capturing three
costs the same GPU time as capturing one. The single-tap restriction bought
nothing.

Adds a `concat` pseudo-tap (all three concatenated) plus a PCA-256 version of it,
because concat carries 1280 dims against a single tap's 256 and this project has
been burned by unmatched-dimension comparisons before.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

DEC_TS = (261, 500, 750, 900)
ENS_TS = (261, 500, 750)


@torch.no_grad()
def centre2_multi(tap, boxes, timestep, batch, tapnames, noise_level=None,
                  noise_seed: int = 0) -> dict:
    """Central 2^3 feature cells for EVERY tap, from a single forward per batch.

    Same reduction as `o4_frameavg_benchmark.centre2` (which this mirrors) so the
    numbers stay comparable with every existing table; the only change is reading
    all taps out of `_buf` instead of one. Generator hoisted out of the loop --
    see the centre2 noise-seeding fix.
    """
    from teachers.cryofm_tap import StopForward
    out: dict[str, list] = {k: [] for k in tapnames}
    gen = torch.Generator(device="cpu").manual_seed(noise_seed)
    for s in range(0, len(boxes), batch):
        x = boxes[s:s + batch].to(tap.device)
        if noise_level:
            eps = torch.randn(x.shape, generator=gen).to(x.device)
            x = (1.0 - noise_level) * x + noise_level * eps
        x = torch.cat([x, torch.zeros_like(x)], dim=1)
        t = torch.full((x.shape[0],), timestep, device=tap.device, dtype=torch.long)
        tap._buf.clear()
        try:
            tap.model(x, timestep=t)
        except StopForward:
            pass
        for k in tapnames:
            f = tap._buf[k]
            d = f.shape[-1]
            c = d // 2
            out[k].append(f[:, :, c - 1:c + 1, c - 1:c + 1, c - 1:c + 1]
                          .mean(dim=(2, 3, 4)).float().cpu().numpy())
        tap._buf.clear()
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--vol-dir", type=Path, default=Path("data/cleandift_vols"))
    ap.add_argument("--taps", default="mid_block,up_blocks[0],up_blocks[1]")
    ap.add_argument("--cou-t", type=int, default=500)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--per-chain", type=int, default=60)
    ap.add_argument("--box-chunk", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--students", default="")
    ap.add_argument("--feat-dir", type=Path, default=Path("data/o5_multitap_parts"))
    ap.add_argument("--out", type=Path, default=Path("results/o5_arms_multitap.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from probes.o5_boxes import chain_boxes
    from teachers.cryofm_tap import TAP_STRIDES, CryoFM2Tap

    taps = tuple(t for t in args.taps.split(",") if t)
    order = ["conv_in"] + [f"down_blocks[{i}]" for i in range(4)] + ["mid_block"] \
        + [f"up_blocks[{i}]" for i in range(4)]
    stop_after = max(taps, key=order.index)

    students = {}
    for spec in filter(None, args.students.split(",")):
        n, _, p = spec.partition("=")
        students[n] = p
    dec_arms = [f"dec_t{t}" for t in DEC_TS]
    arms = dec_arms + ["cou_best", "cou_seed1", "rand_student"] + sorted(students)
    keys = [f"{a}@{t}" for a in arms for t in taps] + ["raw", "aa", "ss", "split",
                                                      "cluster"]

    rows = list(csv.DictReader(open(args.chains)))[: args.limit]
    rows.sort(key=lambda r: r["emd"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.feat_dir.mkdir(parents=True, exist_ok=True)
    _c: dict = {}

    def get_tap(kind):
        if kind not in _c:
            kw = dict(taps=taps, device=dev, batch_size=args.batch_size,
                      stop_after=stop_after)
            if kind == "teacher":
                m = CryoFM2Tap(args.ckpt, **kw)
            elif kind == "rand":
                m = CryoFM2Tap(args.ckpt, random_weights=True, **kw)
            else:
                m = CryoFM2Tap(args.ckpt, student_ckpt=students[kind], **kw)
            _c[kind] = m
        return _c[kind]

    print(f"{len(rows)} chains | taps={taps} stop_after={stop_after} | "
          f"{len(arms)} arms x {len(taps)} taps | {dev}", flush=True)
    rng = np.random.default_rng(args.seed)
    done = skipped = 0
    for i, r in enumerate(rows):
        part = args.feat_dir / f"{r['key']}.npz"
        if part.exists():
            z = np.load(part, allow_pickle=True)
            if not set(keys) - set(z.files):
                done += 1
                continue
            part.unlink()
        try:
            boxes, aa, ss, _ = chain_boxes(r, args.vol_dir, args.per_chain, rng,
                                           dev, args.box_chunk)
            n = len(aa)
            out = {}
            for t in DEC_TS:                       # decoupled: clean input
                for k, v in centre2_multi(get_tap("teacher"), boxes, t,
                                          args.batch_size, taps).items():
                    out[f"dec_t{t}@{k}"] = v
            for nm, sd in (("cou_best", 0), ("cou_seed1", 1)):
                for k, v in centre2_multi(get_tap("teacher"), boxes, args.cou_t,
                                          args.batch_size, taps,
                                          noise_level=args.cou_t / 1000.0,
                                          noise_seed=sd).items():
                    out[f"{nm}@{k}"] = v
            for k, v in centre2_multi(get_tap("rand"), boxes, 0, args.batch_size,
                                      taps).items():
                out[f"rand_student@{k}"] = v
            for nm in students:                    # student: clean input, no t
                for k, v in centre2_multi(get_tap(nm), boxes, 0, args.batch_size,
                                          taps).items():
                    out[f"{nm}@{k}"] = v
            from teachers.cryofm_tap import PATCH
            c = PATCH // 2
            raw = boxes[:, 0, c - 4:c + 4, c - 4:c + 4, c - 4:c + 4] \
                .reshape(n, -1).float().cpu().numpy().astype(np.float32)
            del boxes
            if dev == "cuda":
                torch.cuda.empty_cache()
            np.savez(part, raw=raw, aa=aa, ss=ss,
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
    if done == 0:
        raise SystemExit("FAILED: 0 chains extracted. See the SKIP lines above.")
    if skipped > 0.5 * len(rows):
        raise SystemExit(f"FAILED: {skipped}/{len(rows)} skipped -- systematic fault.")

    # --- aggregate ---------------------------------------------------------
    parts = sorted(args.feat_dir.glob("*.npz"))
    buf: dict[str, list] = {k: [] for k in keys}
    for p in parts:
        z = np.load(p, allow_pickle=True)
        miss = set(keys) - set(z.files)
        if miss:
            raise KeyError(f"{p} lacks {sorted(miss)[:3]}...; delete and re-extract")
        for k in keys:
            buf[k].append(z[k])
    F = {k: np.concatenate(v) for k, v in buf.items()}

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from probes.o5_stats import cluster_bootstrap_diff, mdi

    sp = F["split"].astype(str)
    tr, va, te = sp == "train", sp == "val", sp == "test"
    for nm, m in (("train", tr), ("val", va), ("test", te)):
        if m.sum() == 0:
            raise SystemExit(f"empty {nm} split; raise --limit")
    clusters = F["cluster"].astype(str)[te]
    print(f"residues: train {tr.sum()} val {va.sum()} test {te.sum()} | "
          f"test clusters {len(np.unique(clusters))}", flush=True)

    res: dict = {"_meta": {"taps": list(taps), "cou_t": args.cou_t,
                           "n_chains": len(parts), "per_chain": args.per_chain,
                           "n_test_clusters": int(len(np.unique(clusters))),
                           "students": students,
                           "tap_angstroms": {t: TAP_STRIDES[t] * 1.5 for t in taps}}}

    tap_views = list(taps) + ["concat", "concat_pca"]
    for view in tap_views:
        def feats(arm):
            if view == "concat":
                return np.concatenate([F[f"{arm}@{t}"] for t in taps], axis=1)
            if view == "concat_pca":
                return None            # handled below
            return F[f"{arm}@{view}"]

        X_all = {}
        for a in arms:
            if view == "concat_pca":
                x = np.concatenate([F[f"{a}@{t}"] for t in taps], axis=1)
                X_all[a] = PCA(n_components=256, random_state=0).fit(
                    x[tr]).transform(x).astype(np.float32)
            else:
                X_all[a] = feats(a)
        X_all["raw"] = F["raw"]
        view_arms = arms + ["raw"]

        for task in ("ss", "aa"):
            y = F[task]
            ok, acc, accv = {}, {}, {}
            for a in view_arms:
                X = X_all[a]
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(
                    sc.transform(X[tr]), y[tr])
                ok[a] = clf.predict(sc.transform(X[te])) == y[te]
                acc[a] = float(ok[a].mean())
                accv[a] = float((clf.predict(sc.transform(X[va])) == y[va]).mean())
            dec_best = max(dec_arms, key=lambda a: accv[a])
            dec_oracle = max(dec_arms, key=lambda a: acc[a])
            print(f"\n=== {view} / {task} (dec_best={dec_best} "
                  f"{acc[dec_best]:.4f}, dec_oracle={dec_oracle} "
                  f"{acc[dec_oracle]:.4f}) ===", flush=True)
            for a in view_arms:
                print(f"  {a:22s} test {acc[a]:.4f}  val {accv[a]:.4f}  "
                      f"d={X_all[a].shape[1]:5d}", flush=True)
            blk = {"acc": acc, "acc_val": accv, "dec_best": dec_best,
                   "dec_oracle": dec_oracle,
                   "dim": {a: int(X_all[a].shape[1]) for a in view_arms},
                   "pairs": {}}
            for s in students:
                for base in (dec_best, dec_oracle, "cou_best", "raw"):
                    if base == s:
                        continue
                    d = cluster_bootstrap_diff(ok[s], ok[base], clusters)
                    d["mde_95"] = mdi(d)
                    blk["pairs"][f"{s}-{base}"] = d
                    print(f"  {s:22s} - {base:12s} {d['diff']:+.4f} "
                          f"[{d['lo95']:+.4f},{d['hi95']:+.4f}] "
                          f"MDE {d['mde_95']:.4f}"
                          f"{'  *' if d['excludes_zero'] else ''}", flush=True)
            res.setdefault(view, {})[task] = blk

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
