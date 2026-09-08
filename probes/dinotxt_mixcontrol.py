"""T0-B verdict: does token mixing beat POST-HOC SPATIAL SMOOTHING?

Both models and every control arm run IN ONE PROCESS on identical voxels, so the
comparison can be a PAIRED cluster bootstrap. Two separate `dinotxt_eval` runs
give the same voxels (selection is a deterministic per-map RNG) but no pairing,
and this project's measured cross-process wobble is ~1.1 points -- enough to
swamp the quantity of interest.

WHY THE CONTROL IS THE WHOLE EXPERIMENT. Voxels near each other usually belong to
the same chain, so ANY mechanism that makes neighbouring predictions agree raises
E1 without improving alignment. Gaussian-smoothing the POINTWISE model's
per-sequence scores is that mechanism in its purest form, at zero training cost.
The honest claim for mixing is not "mixing > pointwise" but "mixing > the best
smoothing of pointwise", and the gap between those two is how much of the gain is
genuinely learned rather than geometric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def build(ckpt: Path, corpus, emd0, dev):
    from probes.dinotxt_model import DinoTxt
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    a, d0 = ck["args"], corpus.get(emd0)
    m = DinoTxt(d_vox=d0["feat"].shape[1], d_seq=d0["res_emb"].shape[1],
                d=a["dim"], hidden=a["hidden"], depth=a["depth"],
                tau_init=a["tau"], pair=a.get("pair", "none"),
                n_tri=a.get("n_tri", 2), d_pair=a.get("d_pair", 32),
                mix_depth=a.get("mix_depth", 0), mix_heads=a.get("mix_heads", 4),
                mix_rmax=a.get("mix_rmax", 30.0)).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/voxel_cache"))
    ap.add_argument("--esmc", type=Path, default=Path("data/esmc_seq32"))
    ap.add_argument("--coords", type=Path, default=Path("data/voxel_coords"))
    ap.add_argument("--mix-ckpt", type=Path,
                    default=Path("data/dinotxt_runs/mix2_12k/best.pt"))
    ap.add_argument("--ptw-ckpt", type=Path,
                    default=Path("data/dinotxt_runs/ptw_rot_12k/best.pt"))
    ap.add_argument("--sigma", type=float, nargs="+", default=[4, 8, 12, 20])
    ap.add_argument("--n-vox", type=int, default=800)
    ap.add_argument("--out", type=Path,
                    default=Path("results/dinotxt_mixcontrol.json"))
    args = ap.parse_args()

    from probes.dinotxt_data import VoxelCorpus
    from probes.dinotxt_eval import eval_arm, macro_chain_metrics
    from probes.o5_stats import cluster_bootstrap_diff, mdi

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = VoxelCorpus(args.cache, args.esmc, coords_dir=args.coords)
    emds = corpus.by_split("test")
    mix, ckm = build(args.mix_ckpt, corpus, emds[0], dev)
    ptw, ckp = build(args.ptw_ckpt, corpus, emds[0], dev)
    assert getattr(mix, "mix", None) is not None, "--mix-ckpt has no mixing"
    assert getattr(ptw, "mix", None) is None, "--ptw-ckpt already mixes"
    print(f"{len(emds)} test maps | mix step {ckm.get('step')} | "
          f"ptw step {ckp.get('step')} | {dev}", flush=True)

    ev = dict(single_rot=True)
    arms = {"mix": eval_arm(mix, corpus, emds, args.n_vox, dev, **ev),
            "pointwise": eval_arm(ptw, corpus, emds, args.n_vox, dev, **ev),
            "volume_prior": eval_arm(ptw, corpus, emds, args.n_vox, dev,
                                     prior_only=True, **ev),
            "mix_shuffled": eval_arm(mix, corpus, emds, args.n_vox, dev,
                                     shuffle=True, **ev)}
    for sg in args.sigma:
        arms[f"smooth{sg:g}"] = eval_arm(ptw, corpus, emds, args.n_vox, dev,
                                         smooth_sigma=float(sg), **ev)

    n = len(arms["mix"][0])
    for k, (ok, cl, *_rest) in arms.items():
        assert len(ok) == n and np.array_equal(cl, arms["mix"][1]), \
            f"arm {k} evaluated different voxels -- comparison invalid"
    cl = arms["mix"][1]
    print(f"voxels {n} | maps {len(np.unique(cl))}\n")

    acc = {k: float(v[0].mean()) for k, v in arms.items()}
    mac = {k: macro_chain_metrics(v[2], v[3], v[1]) for k, v in arms.items()}
    print(f"  {'arm':16s} {'top1':>7s} {'macroIoU':>9s} {'macroRec':>9s}")
    for k in acc:
        print(f"  {k:16s} {acc[k]:7.4f} {mac[k]['macro_iou']:9.4f} "
              f"{mac[k]['macro_recall']:9.4f}")

    best_s = max((k for k in acc if k.startswith("smooth")), key=lambda k: acc[k])
    res = {"_meta": {"n_voxels": int(n), "n_maps": int(len(np.unique(cl))),
                     "mix_ckpt": str(args.mix_ckpt),
                     "ptw_ckpt": str(args.ptw_ckpt),
                     "best_smoothing": best_s},
           "acc": acc,
           "macro": {k: {"macro_iou": v["macro_iou"],
                         "macro_recall": v["macro_recall"]}
                     for k, v in mac.items()},
           "vs": {}}
    for a_, b_ in (("mix", "pointwise"), ("mix", best_s),
                   (best_s, "pointwise"), ("mix", "volume_prior"),
                   ("mix", "mix_shuffled")):
        d = cluster_bootstrap_diff(arms[a_][0], arms[b_][0], cl)
        res["vs"][f"{a_}-{b_}"] = d
        print(f"\n  {a_} - {b_}: {d['diff']:+.4f} "
              f"CI [{d['lo95']:+.4f}, {d['hi95']:+.4f}] MDE {mdi(d):.4f} "
              f"{'SIG' if d['excludes_zero'] else 'ns'}")

    # CIs for the MACRO metrics too. Passing the per-map IoU vector as the
    # "correctness" array with one entry per map makes cluster_bootstrap_diff
    # resample MAPS and average them -- which is exactly a paired bootstrap of
    # a macro-averaged statistic. Without this the macro table would be point
    # estimates while the top-1 table has intervals.
    maps_u = np.unique(cl)
    best_sm = max((k for k in mac if k.startswith("smooth")),
                  key=lambda k: mac[k]["macro_iou"])
    res["_meta"]["best_smoothing_macro"] = best_sm
    print(f"\n  best smoothing by macro IoU: {best_sm} "
          f"(by top-1 it was {best_s})")
    res["vs_macro"] = {}
    for a_, b_ in (("mix", "pointwise"), ("mix", best_sm),
                   (best_sm, "pointwise"), ("mix", "volume_prior")):
        d = cluster_bootstrap_diff(mac[a_]["per_map_iou"],
                                   mac[b_]["per_map_iou"], maps_u)
        res["vs_macro"][f"{a_}-{b_}"] = d
        print(f"  macroIoU {a_} - {b_}: {d['diff']:+.4f} "
              f"CI [{d['lo95']:+.4f}, {d['hi95']:+.4f}] "
              f"{'SIG' if d['excludes_zero'] else 'ns'}")

    g_tot = acc["mix"] - acc["pointwise"]
    g_sm = acc[best_s] - acc["pointwise"]
    res["geometric_fraction"] = float(g_sm / g_tot) if g_tot > 0 else None
    print(f"\n  mixing gain over pointwise: {g_tot:+.4f}; reproducible by "
          f"smoothing alone: {g_sm:+.4f} ({100*g_sm/max(g_tot,1e-9):.0f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=float))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
