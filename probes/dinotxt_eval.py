"""G0 -- does the voxel<->sequence alignment exist at all? All arms, one process.

THE GATE. E1 is per-voxel chain assignment: given a map (no atomic model at
inference) and the set of distinct sequences it contains, say which sequence owns
each voxel. **Stop the whole plan if the trained arm does not clearly beat both
the volume prior and the floors.** It is cheap and it is the kill switch.

ARMS, and none of them is optional (P4 -- an UNTRAINED UNet reached pooled top-1
0.58-0.68 against chance 0.10 in this project, so an absolute number without
floors is uninterpretable):
  trained         the model
  shuffled        map's voxels scored against ANOTHER map's sequences. The
                  pairing floor: anything this arm achieves is available without
                  any real correspondence.
  volume_prior    always predict the sequence owning the most residues. THE size
                  shortcut, and the reason E1 is defined within-map.
  chance          1 / n_sequences, averaged over voxels.
  random_vision   a head TRAINED FROM SCRATCH on features from an UNTRAINED
                  network, evaluated on that same cache. Requires BOTH
                  --random-cache (built with --random-weights) and
                  --random-ckpt (a head trained on it). Reusing the real-feature
                  head here would measure distribution shift, not the floor --
                  it would score near zero for a trivial reason and flatter us.
  untrained_head  the trained model's architecture at init: separates "the
                  frozen features already align" from "training did something".

ALL ARMS IN ONE PROCESS ON IDENTICAL VOXELS. Measured cross-process wobble in
this project is ~1.1 points from nothing but a fresh RNG subsample, so a
comparison assembled from separate runs is not valid. The identical-sample
property is asserted, not assumed.

Statistics are the paired cluster bootstrap over test clusters
(`probes.o5_stats`), never a per-residue SE: voxels within a map are heavily
correlated (adjacent feature cells are ~half-redundant at 4-5 A half-decay).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def eval_arm(model, corpus, emds, n_vox, device, *, shuffle=False,
             prior_only=False, aa_net=None, aa_hard=False, smooth_sigma=0.0,
             single_rot=False):
    """Per-voxel correctness for one arm. Returns (ok [N], cluster [N]).

    `smooth_sigma` > 0 is THE CONTROL FOR TOKEN MIXING (T0-B): it replaces each
    voxel's per-sequence scores with a Gaussian distance-weighted average over
    the map's other voxels, using the POINTWISE model. Neighbouring voxels
    usually share a chain, so this raises E1 purely by making predictions
    locally consistent, with no improvement in alignment. A mixing arm must beat
    this, not merely beat the pointwise model.

    `single_rot` restricts to rotation 0. Required whenever coordinates are used
    -- they are stored in the rotated frame, so mixing or smoothing across
    rotations would combine incompatible geometries.
    """
    ok, cl, PR, YY = [], [], [], []
    with torch.no_grad():
        for j, m in enumerate(emds):
            d = corpus.get(m)
            keep0 = ~d["is_bg"]
            if single_rot:
                keep0 = keep0 & (d["rot"] == 0)
            sel = np.nonzero(keep0)[0]
            if len(sel) == 0:
                continue
            rng = np.random.default_rng(hash(m) % (2 ** 31))   # per-map, fixed
            if len(sel) > n_vox:
                sel = np.sort(rng.choice(sel, n_vox, replace=False))
            ri, w = d["res_idx"][sel], d["weight"][sel]
            keep = (ri >= 0).any(1)
            sel, ri, w = sel[keep], ri[keep], w[keep]
            if len(sel) == 0:
                continue
            best = np.take_along_axis(ri, w.argmax(1)[:, None], 1)[:, 0]
            y = d["seq_of_res"][best]

            if prior_only:
                pred = np.full(len(y), int(np.bincount(d["seq_of_res"]).argmax()))
            elif aa_net is not None:
                # Baseline (b): per-voxel AA distribution scored against each
                # candidate sequence's AA composition. Uses the FULL predicted
                # distribution -- a ~12%-accurate argmax discards most of the
                # signal, and the plan's baseline is composition matching, not
                # hard classification.
                from probes.dinotxt_aa_baseline import composition
                logits = aa_net(torch.from_numpy(
                    d["feat"][sel].astype(np.float32)).to(device))
                p_aa = torch.softmax(logits, dim=1)
                if aa_hard:
                    p_aa = F.one_hot(p_aa.argmax(1), 20).float()
                logf = torch.from_numpy(
                    np.log(composition([str(q) for q in d["seqs"]]))
                ).float().to(device)                       # [n_seq, 20]
                pred = (p_aa @ logf.t()).argmax(1).cpu().numpy()
            else:
                src = emds[(j + 1) % len(emds)] if shuffle else m
                ds = corpus.get(src)
                g = torch.from_numpy(ds["res_emb"]).to(device)
                srn = ds["seq_of_res"].astype(np.int64)
                sr = torch.from_numpy(srn).to(device)
                cs = np.bincount(srn)
                xyz = (torch.from_numpy(d["xyz"][sel]).to(device)
                       if "xyz" in d else None)
                h = model.encode_voxels(
                    torch.from_numpy(d["feat"][sel].astype(np.float32)).to(device),
                    xyz, torch.zeros(len(sel), dtype=torch.long, device=device))
                sim = h @ model.encode_residues(g, chain_sizes=cs).t()
                n_seq = int(sr.max()) + 1
                score = torch.full((len(sel), n_seq), -1e9, device=device)
                score = score.index_reduce_(1, sr, sim, "amax", include_self=True)
                if smooth_sigma > 0:
                    if xyz is None:
                        raise SystemExit("--smooth-sigma needs --coords")
                    dm = torch.cdist(xyz[None], xyz[None])[0]
                    wgt = torch.exp(-0.5 * (dm / smooth_sigma) ** 2)
                    score = (wgt @ score) / wgt.sum(1, keepdim=True)
                pred = score.argmax(1).cpu().numpy()
                if shuffle:
                    # A shuffled arm predicts an index into ANOTHER map's
                    # sequence list, which cannot be correct except by accident
                    # of numbering. That accident is exactly the floor we want.
                    pred = pred % max(int(d["seq_of_res"].max()) + 1, 1)
            ok.append(pred == y)
            cl.append(np.array([m] * len(y)))
            PR.append(pred.astype(np.int64))
            YY.append(y.astype(np.int64))
    if not ok:
        raise SystemExit("no evaluable voxels")
    # pred/y are returned as elements 2 and 3 so existing callers that index
    # [0]/[1] are unaffected; they feed the macro-averaged chain metrics below.
    return (np.concatenate(ok), np.concatenate(cl),
            np.concatenate(PR), np.concatenate(YY))


def macro_chain_metrics(pred, y, clusters) -> dict:
    """Per-chain IoU and recall, macro-averaged over chains then over maps.

    WHY THIS AND NOT ONLY TOP-1. Per-voxel top-1 is dominated by the biggest
    chain -- which is exactly what the volume prior exploits, and why that
    baseline sits at ~0.31 rather than at chance. Macro-averaging weights a
    50-residue chain the same as a 2,000-residue one, so the size shortcut
    collapses toward chance and the metric stops rewarding "get the big chain
    right and ignore the rest".

    It is also much less rewarding of pure SPATIAL SMOOTHING: smoothing makes
    predictions locally coherent, which mostly helps inside already-correct
    large regions; it cannot conjure a small chain that was never predicted.

    IoU is computed over the SAMPLED voxels, so it is an estimate of the
    segmentation overlap, not the exact volumetric IoU against the deposited
    model. Chains absent from the ground truth of a map are skipped; a chain
    present in truth but never predicted scores 0, which is the intended
    penalty for abstention-by-omission.
    """
    out_iou, out_rec = [], []
    for m in np.unique(clusters):
        s = clusters == m
        p, t = pred[s], y[s]
        ious, recs = [], []
        for c in np.unique(t):
            tp = float(((p == c) & (t == c)).sum())
            un = float(((p == c) | (t == c)).sum())
            ious.append(tp / un if un > 0 else 0.0)
            recs.append(tp / float((t == c).sum()))
        out_iou.append(float(np.mean(ious)))
        out_rec.append(float(np.mean(recs)))
    return {"macro_iou": float(np.mean(out_iou)),
            "macro_recall": float(np.mean(out_rec)),
            "per_map_iou": np.array(out_iou),
            "n_maps": len(out_iou)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/voxel_cache"))
    ap.add_argument("--random-cache", type=Path, default=None,
                    help="cache built with --random-weights (the P4 floor)")
    ap.add_argument("--random-ckpt", type=Path, default=None,
                    help="head trained ON --random-cache; required with it")
    ap.add_argument("--aa-ckpt", type=Path, default=None,
                    help="AA classifier from probes.dinotxt_aa_baseline. This is "
                         "baseline (b) and HALF THE G0 STOP CONDITION -- the gate "
                         "is not decided without it.")
    ap.add_argument("--esmc", type=Path, default=Path("data/esmc_seq32"))
    ap.add_argument("--ckpt", type=Path, default=Path("data/dinotxt_runs/v1/best.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--coords", type=Path, default=None,
                    help="recovered voxel coordinates; needed by --smooth-sigma "
                         "and by any checkpoint trained with --mix-depth")
    ap.add_argument("--smooth-sigma", type=float, nargs="*", default=[],
                    help="control arm(s) for T0-B: Gaussian post-hoc smoothing "
                         "of the POINTWISE scores, in Angstrom")
    ap.add_argument("--n-vox", type=int, default=800)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/dinotxt_g0.json"))
    args = ap.parse_args()

    from probes.dinotxt_data import VoxelCorpus
    from probes.dinotxt_model import DinoTxt
    from probes.o5_stats import cluster_bootstrap_diff, mdi

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = VoxelCorpus(args.cache, args.esmc, coords_dir=args.coords)
    emds = corpus.by_split(args.split)
    if args.limit:
        emds = emds[: args.limit]
    if len(emds) < 2:
        raise SystemExit(f"need >=2 {args.split} maps, have {len(emds)}")

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck["args"]
    d0 = corpus.get(emds[0])

    def mk():
        return DinoTxt(d_vox=d0["feat"].shape[1], d_seq=d0["res_emb"].shape[1],
                       d=a["dim"], hidden=a["hidden"], depth=a["depth"],
                       tau_init=a["tau"], pair=a.get("pair", "none"),
                       n_tri=a.get("n_tri", 2), d_pair=a.get("d_pair", 32),
                       mix_depth=a.get("mix_depth", 0),
                       mix_heads=a.get("mix_heads", 4),
                       mix_rmax=a.get("mix_rmax", 30.0),
                       blur_sigma=a.get("blur_sigma", 0.0)).to(dev).eval()

    model = mk()
    model.load_state_dict(ck["model"])
    untrained = mk()

    print(f"{len(emds)} {args.split} maps | ckpt step {ck.get('step')} | {dev}",
          flush=True)

    # One frame for everyone if ANY arm uses coordinates, so all arms stay on
    # identical voxels (the assertion below enforces it).
    one = (bool(args.smooth_sigma) or a.get("mix_depth", 0) > 0
           or a.get("blur_sigma", 0.0) > 0)
    ev = dict(single_rot=one)
    arms: dict = {}
    arms["trained"] = eval_arm(model, corpus, emds, args.n_vox, dev, **ev)
    arms["untrained_head"] = eval_arm(untrained, corpus, emds, args.n_vox, dev,
                                      **ev)
    arms["shuffled"] = eval_arm(model, corpus, emds, args.n_vox, dev,
                                shuffle=True, **ev)
    arms["volume_prior"] = eval_arm(model, corpus, emds, args.n_vox, dev,
                                    prior_only=True, **ev)
    for sg in args.smooth_sigma:
        arms[f"smooth{sg:g}"] = eval_arm(model, corpus, emds, args.n_vox, dev,
                                         smooth_sigma=float(sg), **ev)
    if args.aa_ckpt:
        import torch.nn as nn
        ac = torch.load(args.aa_ckpt, map_location=dev, weights_only=False)
        aa_h = ac["args"]["hidden"]
        aa_net = (nn.Linear(ac["d_in"], 20) if aa_h == 0 else
                  nn.Sequential(nn.LayerNorm(ac["d_in"]),
                                nn.Linear(ac["d_in"], aa_h), nn.GELU(),
                                nn.Linear(aa_h, 20))).to(dev).eval()
        aa_net.load_state_dict(ac["net"])
        arms["aa_classifier"] = eval_arm(model, corpus, emds, args.n_vox, dev,
                                         aa_net=aa_net, **ev)
        arms["aa_classifier_hard"] = eval_arm(model, corpus, emds, args.n_vox,
                                              dev, aa_net=aa_net, aa_hard=True,
                                              **ev)
    if args.random_cache:
        if not args.random_ckpt:
            raise SystemExit(
                "--random-cache needs --random-ckpt: the floor is a head TRAINED "
                "on random features, not the real-feature head fed random input")
        rc = VoxelCorpus(args.random_cache, args.esmc)
        rck = torch.load(args.random_ckpt, map_location=dev, weights_only=False)
        ra, rd0 = rck["args"], rc.get(emds[0])
        rmodel = DinoTxt(
            d_vox=rd0["feat"].shape[1], d_seq=rd0["res_emb"].shape[1],
            d=ra["dim"], hidden=ra["hidden"], depth=ra["depth"],
            tau_init=ra["tau"], pair=ra.get("pair", "none"),
            n_tri=ra.get("n_tri", 2), d_pair=ra.get("d_pair", 32)).to(dev).eval()
        rmodel.load_state_dict(rck["model"])
        arms["random_vision"] = eval_arm(rmodel, rc, emds, args.n_vox, dev, **ev)

    n = len(arms["trained"][0])
    for k, (ok, cl, *_rest) in arms.items():
        assert len(ok) == n, f"arm {k} has {len(ok)} voxels, expected {n}"
        assert np.array_equal(cl, arms["trained"][1]), \
            f"arm {k} evaluated different voxels -- the comparison is invalid"
    clusters = arms["trained"][1]
    print(f"voxels {n} | maps {len(np.unique(clusters))}\n", flush=True)

    res = {"_meta": {"n_voxels": int(n), "n_maps": int(len(np.unique(clusters))),
                     "split": args.split, "ckpt": str(args.ckpt),
                     "step": ck.get("step")},
           "acc": {k: float(v[0].mean()) for k, v in arms.items()}}
    for k, v in res["acc"].items():
        print(f"  {k:16s} {v:.4f}")

    must = ["volume_prior", "shuffled", "untrained_head"]
    if "aa_classifier" in arms:
        must.append("aa_classifier")

    # Bootstrap EVERY arm that is present, not a hardcoded list. An earlier
    # version looped over the three fixed baselines and then gated on `must`,
    # which includes aa_classifier -- so supplying --aa-ckpt crashed with a
    # KeyError after all the accuracies had already been computed.
    res["vs"] = {}
    for base in [k for k in arms if k != "trained"]:
        d = cluster_bootstrap_diff(arms["trained"][0], arms[base][0], clusters)
        res["vs"][base] = d
        print(f"\n  trained - {base}: {d['diff']:+.4f} "
              f"CI [{d['lo95']:+.4f}, {d['hi95']:+.4f}] MDE {mdi(d):.4f} "
              f"{'SIG' if d['excludes_zero'] else 'ns'}")
    passed = all(res["vs"][b]["diff"] > 0 and res["vs"][b]["excludes_zero"]
                 for b in must)
    # G0's stated stop condition names TWO arms: the volume prior AND the AA
    # classifier. Without --aa-ckpt the gate is UNDECIDED, not passed -- reporting
    # a pass on a subset of the stop condition is how the first "G0 PASS" claim in
    # this project got retracted.
    decided = "aa_classifier" in arms
    res["G0_PASS"] = bool(passed and decided)
    res["G0_DECIDED"] = bool(decided)
    res["_meta"]["gated_on"] = must
    if not decided:
        print("\nG0 UNDECIDED -- baseline (b), the AA classifier, was not "
              "supplied (--aa-ckpt). It is half the stated stop condition.")
        print(f"  (would pass on {must}: {passed})")
    else:
        print(f"\nG0 PASS: {passed}  (trained must significantly beat "
              f"{', '.join(must)})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=float))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
