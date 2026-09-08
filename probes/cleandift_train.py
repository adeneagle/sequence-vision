"""CleanDIFT distillation of CryoFM2 -- the trainer (PLAN_CLEANDIFT.md D1-D16).

Teacher: the frozen model on NOISY x_t at a sampled t (the trained pairing).
Student: a trainable copy on the CLEAN x_0, with `time_embedding` replaced by one
learned constant vector, so `t` enters the loss only through the FiLM-conditioned
projection heads -- which are DISCARDED at inference. Downstream there is then one
feature set and no `t` to choose.

Arms this script produces (select with --mode):

  distill     the real thing: coupled teacher, t stratified over [1, t_max]
  ctrl        D12. IDENTICAL in every respect except the teacher is fed CLEAN
              input at fixed t_init, so it can learn nothing about noise
              marginalisation. Any gain the student shows over THIS is
              attributable to CleanDIFT rather than to corpus adaptation /
              self-distillation regularisation. Not optional.
  t_only      D12. `distill`, but the trunk is frozen so only the conditioning
              vector and the heads move. If this recovers most of the gain, all
              the distillation bought was a better conditioning vector -- which a
              plain t sweep gets for free.

Step 0 must equal the naive decoupled teacher at t_init exactly (D3), so that the
null control is built into the initialisation and any gain is attributable to
training rather than to where training started. `teachers/cryofm_student.py
--self-test` asserts that identity; this script logs the step-0 loss as an
explicit baseline row for the same reason.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cryofm.core.utils.scheduling_fm import FMScheduler
from teachers.cryofm_student import (FiLMHead, PaperHead,
                                     attach_const_time_embedding,
                                     embedding_for_t, freeze_trunk,
                                     nearest_timestep)
from teachers.cryofm_tap import DEFAULT_TAPS, CryoFM2Tap


# ---------------------------------------------------------------------------
# Loss (D5)
# ---------------------------------------------------------------------------

class DistillLoss:
    """Centred cosine on dim=1, mean over spatial cells PLUS a centre-token term.

    Every clause here is load-bearing:

    * **Axis.** This codebase is channels-first `[B, C, D, H, W]`, so the channel
      axis is `dim=1`. The reference implementation uses `dim=-1` on channels-last
      tensors; copying that would silently optimise a SPATIAL-PATTERN similarity
      instead of a feature similarity.
    * **Centring is mandatory.** Raw cosine on high-dim activations here is ~0.99
      for ANY pair, from a shared common component. An uncentred loss would
      descend 0.990 -> 0.995 while learning nothing about the informative
      residual. The uncentred value is logged beside it as the control that proves
      the centring is doing work.
    * **Shared mean**, `0.5*(P_bar + T_bar)`, matching `centred_cosine`
      (`probes/stability.py:90`) and therefore every existing number in the
      project. Consequence worth knowing: part of the loss can be reduced by
      matching means alone.
    * **EMA, not the in-batch mean.** At B=8, `mid_block` offers 8x512 tokens to
      estimate a 512-dim mean -- noisy, and correlated within a batch because all
      tokens come from the same 8 boxes. Detached, so no gradient path through the
      centring constant.
    * **Mean, not sum, over spatial cells**, neutralising the 64x token imbalance
      across taps (512 / 4096 / 32768).
    * **A CENTRE-TOKEN term.** The eval reads the central 2^3 feature cells. At
      `up_blocks[1]` that is 8 of 32,768 tokens, so a spatially uniform mean puts
      ~0.02% of the gradient on the only thing the endpoint measures -- a 4,096:1
      mismatch. The second term matches the eval's reduction exactly.
    * **fp32.** Cosine over 256-512 channels in bf16 carries ~1e-2 relative error
      and the effects being chased are sub-1%.
    """

    def __init__(self, taps, momentum: float = 0.99, centre_weight: float = 0.5):
        self.taps = tuple(taps)
        self.momentum = momentum
        self.centre_weight = centre_weight
        self.ema_tok: dict[str, torch.Tensor] = {}
        self.ema_ctr: dict[str, torch.Tensor] = {}

    @staticmethod
    def _pool_centre(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        c = d // 2
        if d == 1:
            return x.reshape(x.shape[0], x.shape[1])
        return x[..., c - 1:c + 1, c - 1:c + 1, c - 1:c + 1].mean((-3, -2, -1))

    def _update(self, store, key, P, T):
        cur = 0.5 * (P.detach().mean(0) + T.detach().mean(0))
        if key not in store:
            store[key] = cur.clone()
        else:
            store[key].mul_(self.momentum).add_(cur, alpha=1.0 - self.momentum)
        return store[key]

    def terms(self, tap: str, P: torch.Tensor, T: torch.Tensor,
              update: bool = True) -> dict:
        P, T = P.float(), T.float()
        B, C = P.shape[:2]
        Pf, Tf = P.reshape(B, C, -1), T.reshape(B, C, -1)
        # token-level shared mean over [B*spatial, C]
        flatP = Pf.permute(0, 2, 1).reshape(-1, C)
        flatT = Tf.permute(0, 2, 1).reshape(-1, C)
        mu = self._update(self.ema_tok, tap, flatP, flatT) if update \
            else self.ema_tok[tap]
        m = mu.detach().view(1, C, 1)
        cos_tok = F.cosine_similarity(Pf - m, Tf - m, dim=1).mean()
        cos_raw = F.cosine_similarity(Pf, Tf, dim=1).mean()

        Pp, Tp = self._pool_centre(P), self._pool_centre(T)
        muc = self._update(self.ema_ctr, tap, Pp, Tp) if update \
            else self.ema_ctr[tap]
        mc = muc.detach().view(1, C)
        cos_ctr = F.cosine_similarity(Pp - mc, Tp - mc, dim=1).mean()
        return {"tok": cos_tok, "ctr": cos_ctr, "raw": cos_raw}

    def __call__(self, feats_head, feats_bypass, teacher_feats) -> tuple:
        cw = self.centre_weight
        loss = 0.0
        diag: dict[str, float] = {}
        for tap in self.taps:
            t = self.terms(tap, feats_head[tap], teacher_feats[tap], update=True)
            # equal tap weights (1/3 each); do NOT tune them -- multiplicity
            loss = loss - ((1.0 - cw) * t["tok"] + cw * t["ctr"]) / len(self.taps)
            diag[f"cos_tok/{tap}"] = float(t["tok"].detach())
            diag[f"cos_ctr/{tap}"] = float(t["ctr"].detach())
            diag[f"cos_raw/{tap}"] = float(t["raw"].detach())
            if feats_bypass is not None:
                with torch.no_grad():
                    b = self.terms(tap, feats_bypass[tap], teacher_feats[tap],
                                   update=False)
                diag[f"cos_bypass/{tap}"] = float((1.0 - cw) * b["tok"] + cw * b["ctr"])
            diag[f"cos_head/{tap}"] = float(((1.0 - cw) * t["tok"] + cw * t["ctr"]).detach())
        return loss, diag


# ---------------------------------------------------------------------------
# Instrumentation (D15)
# ---------------------------------------------------------------------------

def weight_delta(student, teacher) -> dict:
    """Per-module ||W_s - W_t|| / ||W_t||.

    If this is < 1e-3 everywhere at the end, the student IS the teacher and no
    evaluation number means anything regardless of its confidence interval. That
    is a TRAINING FAILURE, not a scientific negative -- re-run at a higher lr.
    """
    td = dict(teacher.named_parameters())
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for n, p in student.named_parameters():
        if n not in td:                       # the swapped time_embedding
            continue
        mod = n.split(".")[0]
        num[mod] = num.get(mod, 0.0) + float((p.detach() - td[n]).pow(2).sum())
        den[mod] = den.get(mod, 0.0) + float(td[n].pow(2).sum())
    return {k: float(np.sqrt(num[k] / max(den[k], 1e-12))) for k in sorted(num)}


def sample_t(bs: int, t_min: int, t_max: int, bins: int,
             rng: np.random.Generator) -> np.ndarray:
    """Stratified timesteps (D6).

    t_max defaults to 1000 (the full schedule), matching the reference.

    An earlier version defaulted to 600, on the grounds that `o1_thigh.json` puts
    the coupled teacher at SS 0.6396 at t=900 against an untrained-UNet control of
    0.5739 -- i.e. the top third looked nearly empty. **That reasoning was
    withdrawn.** It measures how good a SINGLE timestep is as a standalone
    descriptor under a linear probe; CleanDIFT's mechanism is a MERGE across the
    trajectory, and a timestep that is weak alone can still contribute
    complementary information to the merged representation. Judging the merge by a
    per-timestep proxy is the same error as bounding the distillation by the
    noise-marginalisation gate. DIFT's own optimum in SD is mid-schedule (t=261),
    yet CleanDIFT distils from the full range and beats it, which suggests the
    breadth is load-bearing. `--t-max 600` is retained as an ablation.
    """
    edges = np.linspace(t_min, t_max, bins + 1)
    out = []
    for i in range(bs):
        b = i % bins
        out.append(rng.integers(int(np.ceil(edges[b])), int(edges[b + 1]) + 1))
    return np.array(out, dtype=np.int64)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--vol-dir", type=Path, default=Path("data/cleandift_vols"))
    ap.add_argument("--taps", default=",".join(DEFAULT_TAPS))
    ap.add_argument("--mode", default="distill",
                    choices=["distill", "ctrl", "ctrl_sampled", "t_only"])
    ap.add_argument("--t-init", type=int, default=750)
    ap.add_argument("--t-min", type=int, default=1)
    ap.add_argument("--t-max", type=int, default=1000)
    ap.add_argument("--t-bins", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--centre-weight", type=float, default=0.5)
    ap.add_argument("--head-ratio", type=int, default=4,
                    help="FiLMHead bottleneck: hidden = C // ratio")
    ap.add_argument("--head", default="film", choices=["film", "paper"],
                    help="'paper' = 3 stacked SwiGLU FFN blocks per the reference "
                         "(arXiv 2412.03439); 'film' = our 1-block bottlenecked head")
    ap.add_argument("--head-blocks", type=int, default=3)
    ap.add_argument("--head-expand", type=int, default=4)
    ap.add_argument("--per-map", type=int, default=64)
    ap.add_argument("--buffer-maps", type=int, default=6)
    ap.add_argument("--mix-local", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("data/cleandift_runs"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    taps = tuple(t for t in args.taps.split(",") if t)
    tag = args.tag or f"{args.mode}_t{args.t_init}_lr{args.lr:g}_s{args.seed}"
    run = args.out_dir / tag
    run.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    from probes.cleandift_data import BoxStream, split_map_lists
    sp = split_map_lists(args.chains)
    print(json.dumps(sp["info"], indent=2), flush=True)

    # Deepest tap in execution order -> everything after it is pure waste (D14).
    order = ["conv_in"] + [f"down_blocks[{i}]" for i in range(4)] + ["mid_block"] \
        + [f"up_blocks[{i}]" for i in range(4)]
    stop_after = max(taps, key=order.index)

    teacher = CryoFM2Tap(args.ckpt, taps=taps, device=dev, stop_after=stop_after)
    student = CryoFM2Tap(args.ckpt, taps=taps, device=dev, detach=False,
                         trainable=True, stop_after=stop_after)
    vec = embedding_for_t(teacher.model, args.t_init, dev)
    attach_const_time_embedding(student.model, vec)
    for p in teacher.model.parameters():
        p.requires_grad_(False)
    if args.mode == "t_only":
        freeze_trunk(student.model)
    if args.grad_ckpt and hasattr(student.model, "enable_gradient_checkpointing"):
        student.model.enable_gradient_checkpointing()

    # channel counts, needed to size the heads
    with torch.no_grad():
        probe = torch.zeros(1, 2, 64, 64, 64, device=dev)
        tf = teacher.forward_taps(probe, torch.zeros(1, device=dev, dtype=torch.long))
        chans = {k: tf[k].shape[1] for k in taps}
    def _mk_head(c):
        if args.head == "paper":
            return PaperHead(c, n_blocks=args.head_blocks, expand=args.head_expand)
        return FiLMHead(c, ratio=args.head_ratio)

    heads = torch.nn.ModuleDict(
        {k.replace("[", "_").replace("]", ""): _mk_head(chans[k])
         for k in taps}).to(dev)
    hkey = {k: k.replace("[", "_").replace("]", "") for k in taps}

    trainable = [p for p in student.model.parameters() if p.requires_grad] \
        + list(heads.parameters())
    n_tr = sum(p.numel() for p in trainable)
    n_head = sum(p.numel() for p in heads.parameters())
    print(f"mode={args.mode} taps={taps} chans={chans} stop_after={stop_after}\n"
          f"head={args.head} ({n_head/1e6:.1f} M params) | "
          f"trainable {n_tr/1e6:.1f} M | heads hidden "
          f"{ {k: heads[hkey[k]].hidden for k in taps} } | dev {dev}", flush=True)

    # D7: weight_decay = 0.0. dL/dW[:,1] for the all-zero conditioning channel is
    # exactly 0, so a ch1 freeze is machinery for nothing; the reason to drop
    # decay is that pulling a pretrained 168 M model toward zero is not something
    # a light distillation should do at all.
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    sched = FMScheduler(num_train_timesteps=1000)
    lossfn = DistillLoss(taps, centre_weight=args.centre_weight)
    rng = np.random.default_rng(args.seed)

    stream = iter(BoxStream(sp["train"], args.vol_dir, batch=args.batch,
                            per_map=args.per_map, mix_local=args.mix_local,
                            buffer_maps=args.buffer_maps, seed=args.seed,
                            device=dev))
    vstream = iter(BoxStream(sp["val"], args.vol_dir, batch=args.batch,
                             per_map=args.per_map, mix_local=args.mix_local,
                             buffer_maps=2, seed=args.seed + 1, device=dev))

    def forward_pair(x0: torch.Tensor, ts: np.ndarray):
        """Teacher on x_t at t; student on clean x_0 with its learned constant."""
        x0 = x0.to(dev)
        tt = torch.from_numpy(ts).to(dev)
        if args.mode == "ctrl":
            # CLEAN teacher input at a FIXED t. Identical compute, zero information
            # about noise. NOTE its target is TRIVIAL by construction: the D3
            # identity makes student == teacher at step 0, so the cosine starts
            # near 1.0 (measured 0.993) and there is little to learn. Weights do
            # still drift (wdelta 2.7e-2 at 3.5k steps), so it is a WEAK control on
            # corpus adaptation rather than a vacuous one -- but see ctrl_sampled.
            xt = x0
            tt = torch.full_like(tt, args.t_init)
        elif args.mode == "ctrl_sampled":
            # CLEAN teacher input at the SAMPLED t. A strictly stronger control:
            # the target is NON-trivial (decoupled features genuinely vary with t,
            # so the student must merge across t) while still carrying no NOISE
            # information at all. Separates "gain from t-consolidation" from "gain
            # from learning to read noised inputs" -- which `ctrl` cannot do,
            # because its target is already satisfied at initialisation.
            xt = x0
        else:
            eps = torch.randn(x0.shape, device=dev,
                              generator=torch.Generator(device=dev).manual_seed(
                                  int(rng.integers(2 ** 31))))
            xt = sched.add_noise(x0, eps, tt)
        xin_t = torch.cat([xt, torch.zeros_like(xt)], 1)
        xin_s = torch.cat([x0, torch.zeros_like(x0)], 1)
        with torch.no_grad():
            tfe = {k: v.float() for k, v in
                   teacher.forward_taps(xin_t, tt.long()).items()}
            # Heads are conditioned on the TEACHER's sampled t -- never on the
            # student's learned vector. That asymmetry is the mechanism.
            temb = teacher.model.time_embedding(
                teacher.model.time_proj(tt.long()).to(teacher.model.dtype)).float()
        sfe = student.forward_taps(xin_s, torch.zeros_like(tt).long())
        with_head = {k: heads[hkey[k]](sfe[k].float(), temb) for k in taps}
        return with_head, {k: sfe[k].float() for k in taps}, tfe

    def run_val() -> dict:
        agg: dict[str, list] = {}
        with torch.no_grad():
            for _ in range(args.val_batches):
                try:
                    xb = next(vstream)
                except StopIteration:
                    break
                ts = sample_t(len(xb), args.t_min, args.t_max, args.t_bins, rng)
                wh, byp, tfe = forward_pair(xb, ts)
                _, d = lossfn(wh, byp, tfe)
                for k, v in d.items():
                    agg.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in agg.items()}

    hist: list[dict] = []
    step0: dict | None = None
    best = (-np.inf, -1)
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    step = 0
    while step < args.steps:
        try:
            xb = next(stream)
        except StopIteration:
            print("stream exhausted", flush=True)
            break
        ts = sample_t(len(xb), args.t_min, args.t_max, args.t_bins, rng)
        with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
            wh, byp, tfe = forward_pair(xb, ts)
        loss, diag = lossfn(wh, byp, tfe)
        (loss / args.accum).backward()

        if step == 0:
            # Explicit baseline row: a falling curve means nothing without it.
            step0 = {"loss": float(loss), **diag}
            print("step0 (== naive decoupled at t_init, D3): "
                  + json.dumps({k: round(v, 4) for k, v in step0.items()}), flush=True)

        if (step + 1) % args.accum == 0:
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        step += 1

        if step % args.val_every == 0 or step == args.steps:
            v = run_val()
            nt = nearest_timestep(teacher.model, student.model.time_embedding.p)
            wd = weight_delta(student.model, teacher.model)
            row = {"step": step, "train_loss": float(loss), "lr": opt.param_groups[0]["lr"],
                   "val": v, "nearest_t": nt, "weight_delta": wd,
                   "sec": time.time() - t0}
            hist.append(row)
            # cos_BYPASS, not cos_head: the heads are discarded at inference, so
            # the shipped model's quality is the head-skipped agreement. The lr probe
            # showed these diverge exactly where it matters -- at lr 1e-4, cos_head
            # was the best of the three arms while cos_bypass was the worst.
            key = float(np.mean([v[f"cos_bypass/{t}"] for t in taps]))
            print(f"[{step}/{args.steps}] loss {float(loss):+.4f} "
                  f"val_cos_head {key:+.4f} "
                  f"val_cos_bypass {np.mean([v[f'cos_bypass/{t}'] for t in taps]):+.4f} "
                  f"raw {np.mean([v[f'cos_raw/{t}'] for t in taps]):.4f} "
                  f"nearest_t {nt['nearest_t']} (res {nt['rel_residual']:.3f}) "
                  f"wdelta_max {max(wd.values()):.2e} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
            if key > best[0]:
                best = (key, step)
                save(run / "best.pt", student, heads, args, step, hist, step0)
            (run / "history.json").write_text(json.dumps(
                {"args": vars(args) | {k: str(v) for k, v in vars(args).items()
                                       if isinstance(v, Path)},
                 "step0": step0, "hist": hist, "best_step": best[1],
                 "split_info": sp["info"]}, indent=2, default=str))
        if step % args.ckpt_every == 0:
            save(run / f"last{(step // args.ckpt_every) % 2}.pt",
                 student, heads, args, step, hist, step0)

    # --- final report -----------------------------------------------------
    v = run_val()
    wd = weight_delta(student.model, teacher.model)
    share = {}
    for t in taps:
        d_head = v[f"cos_head/{t}"] - step0[f"cos_head/{t}"]
        d_byp = v[f"cos_bypass/{t}"] - step0[f"cos_bypass/{t}"]
        share[t] = float(d_byp / d_head) if abs(d_head) > 1e-9 else float("nan")
    print("\n=== FINAL ===")
    print("  weight_delta:", {k: f"{x:.2e}" for k, x in wd.items()})
    print("  head_share (D15; <0.5 => the heads absorbed the learning "
          "and the SHIPPED trunk barely moved):",
          {k: round(x, 3) for k, x in share.items()})
    print("  nearest_timestep:", nearest_timestep(
        teacher.model, student.model.time_embedding.p))
    if max(wd.values()) < 1e-3:
        print("  *** WEIGHT DELTA < 1e-3: the student IS the teacher. This is a "
              "TRAINING FAILURE, not a negative result. Re-run at higher lr. ***")
    save(run / "final.pt", student, heads, args, step, hist, step0)
    (run / "history.json").write_text(json.dumps(
        {"args": {k: str(x) for k, x in vars(args).items()}, "step0": step0,
         "hist": hist, "final_val": v, "weight_delta": wd, "head_share": share,
         "best_step": best[1], "split_info": sp["info"]}, indent=2, default=str))
    print(f"wrote {run}")


def save(path: Path, student, heads, args, step, hist, step0) -> None:
    """Atomic snapshot: temp path + os.replace, so a preemption mid-write never
    leaves a truncated checkpoint (these jobs are preempted constantly)."""
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"student": student.model.state_dict(),
                "heads": heads.state_dict(),
                "args": {k: str(v) for k, v in vars(args).items()},
                "step": step, "step0": step0, "hist": hist[-5:]}, tmp)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
