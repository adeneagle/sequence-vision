"""STEP 2 (D8) -- the K-draw CEILING test. Run before committing to training.

WHAT IT IS FOR. The student trained with a fresh noise draw every step is an
implicit noise-marginaliser: maximising `E_{t,eps}[cos(proj_t(S(x0)), T(x_t,eps))]`
drives it toward `E_eps[T/||T||]`. So K-draw is NOT a prerequisite for building
the student -- but it measures the student's CEILING, and that is worth 3 h in
front of a ~20 h commitment. If `E_eps[f(x_t,t)]` is no better a per-residue
descriptor than the decoupled `f(x0,t)`, the target carries nothing the decoupled
arm already has and the training run is guaranteed to fail.

t = 500, NOT 750. The target here is the COUPLED teacher, so its timestep must be
selected on the COUPLED sweep, where the peak is t~500 (SS 0.7736). t=750 came
from the DECOUPLED peak -- the right choice for the student's `t_init`, the wrong
one here, and `CLAUDE.md` states the rule it broke: "the optimum is ARM-SPECIFIC
and no cross-arm comparison is licensed". At t=750 coupled scores 0.7476 against
decoupled 0.7747, so noise-averaging would have to climb 2.7 points to reach
parity before earning anything.

K draws are CUMULATIVE, so K = 1, 2, 4, 8 all come from the same 8 forwards and
the saturation curve is free. K=8 still leaves ~35% of the single-draw noise std,
so it UNDERSHOOTS the ceiling -- hence the fit of accuracy against 1/K and the
extrapolation to 1/K = 0.

REQUIRES the `centre2` noise-seeding fix (order of work step 0). With that bug
every "draw" used the identical noise field, so K draws removed the SAME field
every time and noise-averaging would have looked like signal.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

KS = (1, 2, 4, 8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--vol-dir", type=Path, default=Path("data/cleandift_vols"))
    ap.add_argument("--tap", default="up_blocks[1]")
    ap.add_argument("--timestep", type=int, default=500)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--per-chain", type=int, default=40)
    ap.add_argument("--box-chunk", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--feat-dir", type=Path, default=Path("data/o5_kdraw_parts"))
    ap.add_argument("--out", type=Path, default=Path("results/o5_kdraw_gate.json"))
    ap.add_argument("--bar", type=float, default=0.010)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from probes.o4_frameavg_benchmark import centre2
    from probes.o5_boxes import chain_boxes
    from teachers.cryofm_tap import CryoFM2Tap

    ks = [k for k in KS if k <= args.k]
    keys = [f"k{k}" for k in ks] + ["dec", "aa", "ss", "split", "cluster"]
    rows = list(csv.DictReader(open(args.chains)))[: args.limit]
    rows.sort(key=lambda r: r["emd"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.feat_dir.mkdir(parents=True, exist_ok=True)
    nl = args.timestep / 1000.0
    _t: dict = {}

    def tap():
        if not _t:
            _t["m"] = CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                                 batch_size=args.batch_size,
                                 stop_after=args.tap)
        return _t["m"]

    print(f"{len(rows)} chains | tap={args.tap} t={args.timestep} | "
          f"K={ks} cumulative | {dev}", flush=True)
    rng = np.random.default_rng(args.seed)
    done = skipped = 0
    for i, r in enumerate(rows):
        part = args.feat_dir / f"{r['key']}.npz"
        if part.exists():
            z = np.load(part, allow_pickle=True)
            if not set(keys) - set(z.files):
                done += 1
                continue
            part.unlink()          # stale cache from before an arm was added
        try:
            boxes, aa, ss, _ = chain_boxes(r, args.vol_dir, args.per_chain, rng,
                                           dev, args.box_chunk)
            n = len(aa)
            acc = None
            out = {}
            for j in range(max(ks)):
                # a DIFFERENT seed per draw -- the whole point of the exercise
                f = centre2(tap(), boxes, args.timestep, args.batch_size, args.tap,
                            noise_level=nl, noise_seed=1000 * (i + 1) + j)
                acc = f.astype(np.float64) if acc is None else acc + f
                if (j + 1) in ks:
                    out[f"k{j+1}"] = (acc / (j + 1)).astype(np.float32)
            out["dec"] = centre2(tap(), boxes, args.timestep, args.batch_size,
                                 args.tap, noise_level=None).astype(np.float32)
            del boxes
            if dev == "cuda":
                torch.cuda.empty_cache()
            np.savez(part, aa=aa, ss=ss,
                     split=np.array([r["split"]] * n),
                     cluster=np.array([r["cluster"]] * n), **out)
            done += 1
        except Exception as exc:
            skipped += 1
            if skipped <= 10:
                print(f"  SKIP {r['key']} {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 25 == 0:
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

    # --- probe + gate ------------------------------------------------------
    from probes.o5_power_check import fit_correctness
    from probes.o5_stats import cluster_bootstrap_diff, mdi

    parts = sorted(args.feat_dir.glob("*.npz"))
    buf: dict[str, list] = {k: [] for k in keys}
    for p in parts:
        z = np.load(p, allow_pickle=True)
        for k in keys:
            buf[k].append(z[k])
    F = {k: np.concatenate(v) for k, v in buf.items()}
    arms = tuple(f"k{k}" for k in ks) + ("dec",)
    res: dict = {"_meta": {"tap": args.tap, "timestep": args.timestep,
                           "ks": ks, "n_chains": len(parts),
                           "n_residues": int(len(F["aa"]))}}
    for task in ("ss", "aa"):
        ok, acc = fit_correctness(F, task, arms=arms)
        te = F["split"].astype(str) == "test"
        clusters = F["cluster"].astype(str)[te]
        # accuracy vs 1/K, extrapolated to 1/K = 0
        x = np.array([1.0 / k for k in ks])
        y = np.array([acc[f"k{k}"] for k in ks])
        slope, intercept = np.polyfit(x, y, 1)
        d = cluster_bootstrap_diff(ok[f"k{max(ks)}"], ok["dec"], clusters)
        gain_extrap = float(intercept - acc["dec"])
        res[task] = {
            "acc": acc, "acc_k_inf_extrapolated": float(intercept),
            "slope_vs_inv_k": float(slope),
            "gain_kmax_over_dec": d, "mde_95": mdi(d),
            "gain_extrapolated_over_dec": gain_extrap,
            "passes_bar": bool(gain_extrap >= args.bar),
            "resolvable": bool(mdi(d) <= args.bar),
        }
        print(f"\n=== {task} ===")
        for a in arms:
            print(f"  {a:5s} {acc[a]:.4f}")
        print(f"  extrapolated K=inf {intercept:.4f}  "
              f"gain over dec {gain_extrap:+.4f}  (bar {args.bar:+.3f})")
        print(f"  K={max(ks)} vs dec: {d['diff']:+.4f} "
              f"CI [{d['lo95']:+.4f},{d['hi95']:+.4f}] MDE {mdi(d):.4f} "
              f"({d['n_clusters']} clusters)")
    v = res["ss"]
    print("\n=== GATE (ss) ===")
    if not v["resolvable"]:
        print(f"  MDE {v['mde_95']:.4f} > bar {args.bar}: this gate cannot resolve "
              f"its own threshold at {v['gain_kmax_over_dec']['n_clusters']} "
              f"clusters. Raise --limit / --per-chain before believing either sign.")
    print(f"  {'PASS' if v['passes_bar'] else 'FAIL'}: extrapolated gain "
          f"{v['gain_extrapolated_over_dec']:+.4f} vs bar {args.bar:+.3f}")
    if not v["passes_bar"]:
        print("  Per PLAN D8: the coupled target carries nothing the decoupled arm "
              "already has -> the training run is expected to fail. STOP HERE.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
