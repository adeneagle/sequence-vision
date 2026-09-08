"""Train the DINO.txt-style alignment head over the cached voxel corpus.

Both foundation models are frozen and their outputs are cached, so this trains
~1-2 M parameters over tensors on disk: minutes per run, which is what makes the
control arms and ablations affordable at all.

MONITORED, and each one exists because of a specific logged failure:
  * `v2r` / `r2v` -- a large asymmetry indicts the SAMPLER, not the representation.
  * `offdiag_cos` / `rel_variation` -- collapse. NOT effective rank, which read
    197.7 on a provably degenerate set in this project.
  * val E1 top-1 against the VOLUME PRIOR -- the loss falling is not evidence the
    task is being solved; the prior is the size shortcut in its purest form.
Selection is on val E1, never on the loss: feature-matching similarity has
mispredicted downstream performance twice here, in opposite directions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def e1_top1(model, corpus, emds, rng, n_vox: int, device: str,
            shuffle_pairing: bool = False) -> dict:
    """E1: per-voxel, which SEQUENCE owns this density. Foreground voxels only.

    Ground truth = the sequence of the highest-weighted target residue. Score for
    a sequence = MAX similarity over its residues (ColBERT-style late
    interaction), so no pooled vector exists for the size shortcut to enter
    through. Reported beside the volume prior, which is that shortcut.
    """
    from probes.dinotxt_data import assemble

    model.eval()
    hits = tot = prior_hits = 0
    with torch.no_grad():
        for m in emds:
            d = corpus.get(m)
            sel = np.arange(len(d["feat"]))
            fg = ~d["is_bg"][sel]
            # Score within ONE rotation frame whenever coordinates are loaded,
            # NOT merely when the model mixes. Keying this on `model.mix` made
            # the mixing arms report val prior 0.2307 while the pointwise
            # control reported 0.2457 -- the prior does not depend on the model,
            # so a different prior means a different voxel set and the two arms
            # were not comparable on val. Test-time comparability was never at
            # risk (dinotxt_eval forces one frame for every arm and asserts
            # equality), but model selection should be on the same voxels too.
            if "xyz" in d:
                fg = fg & (d["rot"][sel] == 0)
            sel = sel[fg]
            if len(sel) == 0:
                continue
            if len(sel) > n_vox:
                sel = rng.choice(sel, n_vox, replace=False)
            ri, w = d["res_idx"][sel], d["weight"][sel]
            keep = (ri >= 0).any(1)
            sel, ri, w = sel[keep], ri[keep], w[keep]
            if len(sel) == 0:
                continue
            best = np.take_along_axis(ri, w.argmax(1)[:, None], 1)[:, 0]
            y = d["seq_of_res"][best]                       # true sequence index

            src = m
            if shuffle_pairing:                             # the P4 floor arm
                src = emds[(emds.index(m) + 1) % len(emds)]
            g = torch.from_numpy(corpus.get(src)["res_emb"]).to(device)
            sr = corpus.get(src)["seq_of_res"]
            cs = np.bincount(sr.astype(np.int64))
            xyz = (torch.from_numpy(d["xyz"][sel]).to(device)
                   if "xyz" in d else None)
            h = model.encode_voxels(
                torch.from_numpy(d["feat"][sel].astype(np.float32)).to(device),
                xyz, torch.zeros(len(sel), dtype=torch.long, device=device))
            sim = h @ model.encode_residues(g, chain_sizes=cs).t()  # [V,R]

            n_seq = int(sr.max()) + 1
            score = torch.full((len(sel), n_seq), -1e9, device=device)
            srt = torch.from_numpy(sr).to(device)
            score = score.index_reduce_(
                1, srt, sim, "amax", include_self=True)
            pred = score.argmax(1).cpu().numpy()
            # Volume prior: the sequence owning the most residues in this map.
            prior = int(np.bincount(d["seq_of_res"]).argmax())
            hits += int((pred == y).sum())
            prior_hits += int((prior == y).sum())
            tot += len(y)
    model.train()
    if tot == 0:
        return {"top1": float("nan"), "volume_prior": float("nan"), "n": 0}
    return {"top1": hits / tot, "volume_prior": prior_hits / tot, "n": tot}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/voxel_cache"))
    ap.add_argument("--esmc", type=Path, default=Path("data/esmc_seq32"))
    ap.add_argument("--out", type=Path, default=Path("data/dinotxt_runs/v1"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-maps", type=int, default=8)
    ap.add_argument("--n-vox", type=int, default=384)
    ap.add_argument("--eval-vox", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--lambda-dual", type=float, default=0.3)
    ap.add_argument("--lambda-within", type=float, default=0.0,
                    help="OLD two-term rebalance; superseded by --n-cross, kept "
                         "for ablation. Requires --n-cross 0.")
    ap.add_argument("--n-cross", type=int, default=2000,
                    help="cross-structure negatives sampled per map. own-map "
                         "share of the denominator is ~1/batch_maps without "
                         "this (12.5%% at 8 maps); this sets it directly. "
                         "0 = full-batch denominator.")
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--pair", default="none",
                    choices=["none", "esm", "relpos", "esm+relpos"],
                    help="relpos is the MANDATORY control: on the old "
                         "target it beat every ESM-pair arm")
    ap.add_argument("--n-tri", type=int, default=2)
    ap.add_argument("--d-pair", type=int, default=32)
    ap.add_argument("--coords", type=Path, default=None,
                    help="recovered voxel coordinates (probes/"
                         "recover_voxel_coords.py). Required by --mix-depth.")
    ap.add_argument("--mix-depth", type=int, default=0,
                    help="T0-B: distance-biased attention blocks over a map's "
                         "voxels. 0 = pointwise (v1).")
    ap.add_argument("--mix-heads", type=int, default=4)
    ap.add_argument("--n-rot", type=int, default=4,
                    help="rotations in the cache; used to pick one per batch")
    ap.add_argument("--rot-per-batch", action="store_true",
                    help="draw each batch from ONE rotation. Forced on by "
                         "--mix-depth (coordinates are frame-specific). Exists "
                         "as a flag so a POINTWISE control can be trained under "
                         "the identical sampling regime -- otherwise the mixing "
                         "arm differs from v1 in two ways at once.")
    ap.add_argument("--mix-rmax", type=float, default=30.0)
    ap.add_argument("--blur-sigma", type=float, default=0.0,
                    help="control for --mix-depth: fixed Gaussian feature blur "
                         "over a map's voxels, no learned attention")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-maps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def run(args) -> dict:
    """Train once. Returns the run summary; used by the LR sweep."""

    from probes.dinotxt_data import VoxelCorpus, assemble
    from probes.dinotxt_loss import collapse_stats, dinotxt_loss
    from probes.dinotxt_model import DinoTxt

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    if (args.mix_depth > 0 or args.blur_sigma > 0) and args.coords is None:
        raise SystemExit(
            "--mix-depth needs --coords; without coordinates the mixing arm "
            "would fall back to pointwise and report a spurious null")
    corpus = VoxelCorpus(args.cache, args.esmc, coords_dir=args.coords)
    tr, va = corpus.by_split("train"), corpus.by_split("val")
    assert tr and va, f"empty split: train {len(tr)} val {len(va)}"
    assert not (set(tr) & set(va)), "train/val map overlap"
    va_eval = va[: args.eval_maps]
    d0 = corpus.get(tr[0])
    print(f"maps: train {len(tr)} val {len(va)} | voxels/map {len(d0['feat'])} | "
          f"feat {d0['feat'].shape[1]}d | {dev}", flush=True)

    model = DinoTxt(d_vox=d0["feat"].shape[1], d_seq=d0["res_emb"].shape[1],
                    d=args.dim, hidden=args.hidden, depth=args.depth,
                    tau_init=args.tau, pair=args.pair, n_tri=args.n_tri,
                    d_pair=args.d_pair, mix_depth=args.mix_depth,
                    mix_heads=args.mix_heads, mix_rmax=args.mix_rmax,
                    blur_sigma=args.blur_sigma).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    print(f"trainable parameters: {n_par/1e6:.2f} M", flush=True)

    hist, best = [], -1.0
    step, t0 = 0, time.time()
    gen = None
    while step < args.steps:
        gen = corpus.batches(tr, args.batch_maps, rng)
        for emds in gen:
            if step >= args.steps:
                break
            # Coordinates are in the ROTATED frame, so a mixing model must see
            # ONE rotation per batch or it attends across incompatible frames.
            one_rot = (args.mix_depth > 0 or args.blur_sigma > 0
                       or args.rot_per_batch)
            rot = int(rng.integers(args.n_rot)) if one_rot else None
            f, g, ri, w, wb, q, vm, rm, sr, cs, xyz = assemble(
                corpus, emds, args.n_vox, rng, rot=rot)
            t = lambda a, dt=torch.float32: torch.from_numpy(a).to(dev, dt)
            h = model.encode_voxels(
                t(f), None if xyz is None else t(xyz), t(vm, torch.long))
            ge = model.encode_residues(t(g), chain_sizes=cs)
            loss, parts = dinotxt_loss(
                model, h, ge, t(ri, torch.long), t(w), t(wb), sample_w=t(q),
                lambda_dual=args.lambda_dual,
                vox_map=t(vm, torch.long), res_map=t(rm, torch.long),
                lambda_within=args.lambda_within, n_cross=args.n_cross, rng=rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                cs = collapse_stats(h)
                ev = e1_top1(model, corpus, va_eval, np.random.default_rng(0),
                             args.eval_vox, dev)
                rec = {"step": step, "loss": float(loss), **parts, **cs,
                       "val_top1": ev["top1"], "val_prior": ev["volume_prior"],
                       "tau": float(1.0 / model.scale()),
                       "min": (time.time() - t0) / 60}
                hist.append(rec)
                print(f"  [{step}/{args.steps}] loss {float(loss):.4f} "
                      f"(v2r {parts['v2r']:.4f} r2v {parts['r2v']:.4f}) | "
                      f"val top1 {ev['top1']:.4f} vs prior {ev['volume_prior']:.4f} | "
                      f"cos {cs['offdiag_cos']:+.3f} relvar {cs['rel_variation']:.2f} | "
                      f"tau {1.0/float(model.scale()):.3f} | "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
                # Selection is on val E1, never on the loss.
                if ev["top1"] > best:
                    best = ev["top1"]
                    tmp = args.out / "best.tmp"
                    torch.save({"model": model.state_dict(), "args": vars(args),
                                "step": step, "val_top1": best}, tmp)
                    tmp.replace(args.out / "best.pt")
        if not gen:
            break

    torch.save({"model": model.state_dict(), "args": vars(args), "step": step},
               args.out / "final.pt")
    (args.out / "history.json").write_text(json.dumps(hist, indent=2, default=float))
    print(f"\nbest val E1 top-1 {best:.4f} | {(time.time()-t0)/60:.1f} min "
          f"-> {args.out}", flush=True)
    return {"best_val_top1": float(best), "steps": step,
            "history": hist, "out": str(args.out)}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
