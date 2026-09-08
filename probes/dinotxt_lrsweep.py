"""Discrete LR sweep for the alignment head, selected on VAL E1 -- not an LR finder.

WHY NOT `Tuner.lr_find()` / the Smith LR range test. It selects the learning rate
by where the TRAINING LOSS descends fastest, and this project has twice measured
that the training objective does not predict downstream performance, in opposite
directions: `student_paperhead` had LOWER feature-matching cosine but HIGHER SS
(0.7908 vs 0.7862), while `student_t700` had MUCH higher cosine (0.798 vs 0.706)
and equal SS. Selecting on the loss would have shipped `t700` and skipped
`paperhead`. The gate design exists to prevent exactly that, so the LR must be
chosen on the downstream metric too. (Lightning is not installed in this
subproject in any case, and core torch has no LR finder -- only schedulers.)

This also follows the CleanDIFT stage's own convention: a small discrete probe
(`lrprobe_1e-5`, `3e-5`, `1e-4`, 500 steps each), not a range test. Here it is
even cheaper, because training runs over cached tensors in minutes.

REPORTED, not just the winner: the val-E1 curve against the VOLUME PRIOR, plus
collapse statistics per arm. A learning rate that maximises val E1 while the
representation is collapsing is not a winner, and raw off-diagonal cosine is the
only thing here that detects that (effective rank does NOT -- it read 197.7 on a
provably degenerate set in this project).
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def main() -> None:
    from probes.dinotxt_train import build_parser, run

    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", type=float, nargs="+",
                    default=[1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--steps", type=int, default=800,
                    help="short: this selects an LR, it does not train the model")
    ap.add_argument("--out", type=Path, default=Path("data/dinotxt_runs/lrsweep"))
    ap.add_argument("--result", type=Path,
                    default=Path("results/dinotxt_lrsweep.json"))
    ap.add_argument("--rest", nargs=argparse.REMAINDER, default=[],
                    help="extra args forwarded verbatim to the trainer")
    a = ap.parse_args()

    base = build_parser().parse_args(a.rest)
    base.steps = a.steps
    # Same seed for every arm: the comparison is over LR, so anything else that
    # differs between arms is a confound. The measured run-to-run wobble in this
    # project is ~1.1 points from nothing but a fresh RNG.
    out: dict = {"_meta": {"lrs": a.lrs, "steps": a.steps, "seed": base.seed},
                 "arms": {}}
    for lr in a.lrs:
        args = deepcopy(base)
        args.lr = lr
        args.out = a.out / f"lr{lr:g}"
        print(f"\n{'='*66}\nLR {lr:g}\n{'='*66}", flush=True)
        r = run(args)
        last = r["history"][-1] if r["history"] else {}
        out["arms"][f"{lr:g}"] = {
            "best_val_top1": r["best_val_top1"],
            "final_val_top1": last.get("val_top1"),
            "val_prior": last.get("val_prior"),
            "final_loss": last.get("loss"),
            "offdiag_cos": last.get("offdiag_cos"),
            "rel_variation": last.get("rel_variation"),
            "tau": last.get("tau"),
        }

    print(f"\n{'='*66}")
    print(f"  {'lr':>8s} {'best val E1':>12s} {'prior':>8s} {'loss':>9s} "
          f"{'cos':>7s} {'relvar':>7s}")
    for k, v in out["arms"].items():
        print(f"  {k:>8s} {v['best_val_top1']:12.4f} {v['val_prior'] or 0:8.4f} "
              f"{v['final_loss'] or 0:9.4f} {v['offdiag_cos'] or 0:+7.3f} "
              f"{v['rel_variation'] or 0:7.2f}")

    # Winner on val E1, but a collapsed arm cannot win regardless of its score.
    def ok(v):
        return (v["rel_variation"] or 0) > 0.1 and (v["offdiag_cos"] or 0) < 0.9
    live = {k: v for k, v in out["arms"].items() if ok(v)}
    dead = sorted(set(out["arms"]) - set(live))
    if dead:
        print(f"  excluded as collapsed: {dead}")
    if not live:
        raise SystemExit("every arm collapsed -- the LR range is wrong, or the "
                         "objective is degenerate. Do not pick a winner from this.")
    best = max(live, key=lambda k: live[k]["best_val_top1"])
    out["best_lr"] = float(best)
    print(f"\n  -> best lr {best} (val E1 {live[best]['best_val_top1']:.4f})")

    a.result.parent.mkdir(parents=True, exist_ok=True)
    a.result.write_text(json.dumps(out, indent=2, default=float))
    print(f"  -> {a.result}")


if __name__ == "__main__":
    main()
