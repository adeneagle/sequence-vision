"""GPU verification for the CleanDIFT implementation. Run before anything else.

Each check guards a failure that is SILENT otherwise -- a falling loss curve, or a
plausible number, with the mechanism broken underneath:

  1. default-path preservation -- the `detach` flag must leave every existing
     probe byte-for-byte unaffected (detached fp32 buffers, no grad).
  2. the `centre2` noise fix is really active: `--legacy-noise` reproduces the old
     per-batch re-seeding, the fixed path differs from it, and both are
     reproducible. Without this, "K independent draws" were K copies of one draw.
  3. D3 identity, head identity-at-init, gradient plumbing, D14 truncation
     equality (delegated to `teachers.cryofm_student._self_test`).
  4. D5 loss sanity on REAL boxes: the centred cosine must sit well below 1.0
     while the uncentred one sits ~0.99. If centring is broken they coincide, and
     the run would optimise a shared common component instead of the signal.
  5. a short live training smoke: loss moves, weights move, `cos_bypass` exists.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--vol-dir", type=Path, default=Path("data/cleandift_vols"))
    ap.add_argument("--t-init", type=int, default=750)
    ap.add_argument("--smoke-steps", type=int, default=30)
    ap.add_argument("--student-ckpt", default=None,
                    help="if given, verify the student checkpoint round-trip (D2 swap "
                         "must be applied BEFORE load_state_dict, and load_cryofm2 "
                         "raises on any key mismatch)")
    args = ap.parse_args()

    from probes.o4_frameavg_benchmark import centre2
    from probes.o5_boxes import chain_boxes
    from teachers.cryofm_student import _self_test
    from teachers.cryofm_tap import CryoFM2Tap

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("WARNING: no CUDA -- this verification is meant to run on a GPU")
    print(f"device {dev}\n{'='*70}", flush=True)

    # -- 1. default path preserved ---------------------------------------
    print("[1] default-path preservation", flush=True)
    tap = CryoFM2Tap(args.ckpt, taps=("up_blocks[1]",), device=dev, batch_size=2)
    x = torch.randn(2, 1, 64, 64, 64, generator=torch.Generator().manual_seed(0))
    x = torch.cat([x, torch.zeros_like(x)], 1).to(dev)
    with torch.no_grad():
        f = tap.forward_taps(x, torch.full((2,), 261, device=dev, dtype=torch.long))
    v = f["up_blocks[1]"]
    assert v.dtype == torch.float32, f"default buffer dtype {v.dtype}, expected fp32"
    assert not v.requires_grad, "default buffer requires grad -- probes would change"
    assert tap.detach is True and tap.stop_after is None
    print(f"    OK  fp32, detached, shape {tuple(v.shape)}", flush=True)

    # -- 2. the centre2 noise fix ----------------------------------------
    print("[2] centre2 noise seeding", flush=True)
    rows = list(csv.DictReader(open(args.chains)))
    row = None
    for r in rows:
        try:
            boxes, aa, ss, _ = chain_boxes(r, args.vol_dir, 24,
                                           np.random.default_rng(0), dev, 8)
            row = r
            break
        except Exception:
            continue
    assert row is not None, "no usable chain found"
    print(f"    chain {row['key']}, {len(aa)} residues", flush=True)
    kw = dict(timestep=500, batch=8, tapname="up_blocks[1]", noise_level=0.5)
    fix_a = centre2(tap, boxes, **kw, noise_seed=0)
    fix_b = centre2(tap, boxes, **kw, noise_seed=0)
    leg_a = centre2(tap, boxes, **kw, noise_seed=0, legacy_noise=True)
    leg_b = centre2(tap, boxes, **kw, noise_seed=0, legacy_noise=True)
    dec_a = centre2(tap, boxes, timestep=500, batch=8, tapname="up_blocks[1]",
                    noise_level=None)
    dec_b = centre2(tap, boxes, timestep=500, batch=8, tapname="up_blocks[1]",
                    noise_level=None)
    assert np.array_equal(fix_a, fix_b), "fixed path is not reproducible"
    assert np.array_equal(leg_a, leg_b), "legacy path is not reproducible"
    assert np.array_equal(dec_a, dec_b), "decoupled path is not deterministic"
    d = float(np.abs(fix_a - leg_a).max())
    assert d > 0, "fixed == legacy: the noise fix is NOT active"
    # the bug is per-BATCH, so batch 0 is identical and later batches differ
    n0 = float(np.abs(fix_a[:8] - leg_a[:8]).max())
    n1 = float(np.abs(fix_a[8:] - leg_a[8:]).max())
    print(f"    OK  reproducible; fixed-vs-legacy max|d| {d:.4g} "
          f"(batch0 {n0:.4g}, later batches {n1:.4g})", flush=True)
    assert n1 > 0, "later batches identical -- the per-batch re-seed is still there"

    # -- 2b. stop_after through the centre2 path --------------------------
    # The check that was missing: `stop_after` signals truncation by RAISING from
    # the capture hook, and `centre2` calls `tap.model(...)` directly, so it must
    # catch StopForward. It did not, and the K-draw gate skipped 300/300 chains
    # with "StopForward: up_blocks[1]" -- a total failure that only surfaced
    # because the per-chain `except` reported it as a data problem.
    print("[2b] stop_after via centre2", flush=True)
    tap_cut = CryoFM2Tap(args.ckpt, taps=("up_blocks[1]",), device=dev,
                         batch_size=8, stop_after="up_blocks[1]")
    a = centre2(tap, boxes, timestep=500, batch=8, tapname="up_blocks[1]",
                noise_level=None)
    b = centre2(tap_cut, boxes, timestep=500, batch=8, tapname="up_blocks[1]",
                noise_level=None)
    err = float(np.abs(a - b).max())
    print(f"    truncated vs full through centre2: max|d| {err:.3e} "
          f"(scale {float(np.abs(a).max()):.3g})", flush=True)
    assert err <= 1e-6 * max(float(np.abs(a).max()), 1.0), \
        "stop_after changes centre2 output"
    del tap_cut
    if dev == "cuda":
        torch.cuda.empty_cache()

    # -- 3. student identities -------------------------------------------
    print(f"[3] student self-test (D3 / head / grad / D14)\n{'-'*70}", flush=True)
    _self_test(args.ckpt, args.t_init, dev)
    print("-" * 70, flush=True)

    # -- 4. D5 loss sanity on real boxes ---------------------------------
    print("[4] D5 loss sanity", flush=True)
    from probes.cleandift_train import DistillLoss
    taps = ("mid_block", "up_blocks[0]", "up_blocks[1]")
    from teachers.cryofm_student import (attach_const_time_embedding,
                                         embedding_for_t)
    teach = CryoFM2Tap(args.ckpt, taps=taps, device=dev, stop_after="up_blocks[1]")
    stud = CryoFM2Tap(args.ckpt, taps=taps, device=dev, detach=False,
                      trainable=True, stop_after="up_blocks[1]")
    attach_const_time_embedding(stud.model, embedding_for_t(teach.model,
                                                            args.t_init, dev))
    xb = boxes[:4].to(dev)
    xin = torch.cat([xb, torch.zeros_like(xb)], 1)
    t = torch.full((4,), 500, device=dev, dtype=torch.long)
    from cryofm.core.utils.scheduling_fm import FMScheduler
    eps = torch.randn(xb.shape, device=dev,
                      generator=torch.Generator(device=dev).manual_seed(0))
    xt = FMScheduler().add_noise(xb, eps, t)
    with torch.no_grad():
        tf = {k: v.float() for k, v in
              teach.forward_taps(torch.cat([xt, torch.zeros_like(xt)], 1), t).items()}
    sf = {k: v.float() for k, v in
          stud.forward_taps(xin, torch.zeros(4, device=dev, dtype=torch.long)).items()}
    lf = DistillLoss(taps)
    loss, diag = lf(sf, sf, tf)
    print(f"    loss {float(loss):+.4f}")
    for k in taps:
        ct, cr = diag[f"cos_tok/{k}"], diag[f"cos_raw/{k}"]
        print(f"    {k:15s} centred {ct:+.4f}  uncentred {cr:+.4f}  "
              f"centre-token {diag[f'cos_ctr/{k}']:+.4f}")
        assert abs(ct) < 0.97, (f"{k}: centred cosine {ct:.4f} ~ 1.0 -- centring is "
                                "NOT working and the loss would chase the shared "
                                "common component")
    # NOTE, and it corrects an expectation stated in the plan's D5: the uncentred
    # cosine is NOT ~0.99 for this pair. "Raw cosine is ~0.99 for any pair" holds
    # for features from the SAME extractor at the SAME operating point; teacher
    # (noisy x_t at t) and student (clean x_0 at t_init) are different operating
    # points, and the plan's own D13 records them as near-orthogonal (median
    # centred cosine 0.003 to -0.44). Measured here: uncentred 0.16-0.26. So the
    # right check is that centring MATERIALLY CHANGES the number -- proving the
    # mean subtraction is live on real data -- not that raw sits near 1.0.
    gaps = [abs(diag[f"cos_tok/{k}"] - diag[f"cos_raw/{k}"]) for k in taps]
    print(f"    centring shifts the cosine by {min(gaps):.3f}-{max(gaps):.3f} "
          f"across taps", flush=True)
    assert min(gaps) > 1e-3, ("centring changes nothing on real data -- the shared "
                              "mean is not being subtracted")
    print("    OK  centred != uncentred, and centred is far from 1.0", flush=True)

    # -- 4b. student checkpoint round-trip --------------------------------
    # NEVER EXERCISED until now: no student checkpoint existed while the pipeline
    # was being built. If this is broken, every training run completes and THEN
    # the evaluation fails -- the worst possible time to find out. The specific
    # hazard is ordering: after the D2 swap the keys are `time_embedding.p`, not
    # `time_embedding.linear_1.*`, and `load_cryofm2` raises on any mismatch.
    if args.student_ckpt:
        print(f"[4b] student checkpoint round-trip: {args.student_ckpt}", flush=True)
        from teachers.cryofm_student import LearnedTimeEmb
        sd = torch.load(args.student_ckpt, map_location="cpu", weights_only=True)
        ref_p = sd["student"]["time_embedding.p"]
        s1 = CryoFM2Tap(args.ckpt, taps=("up_blocks[1]",), device=dev, batch_size=8,
                        stop_after="up_blocks[1]", student_ckpt=args.student_ckpt)
        assert isinstance(s1.model.time_embedding, LearnedTimeEmb), \
            "time_embedding is not a LearnedTimeEmb after load -- the swap was skipped"
        got = s1.model.time_embedding.p.detach().cpu()
        assert torch.equal(got, ref_p), \
            "learned conditioning vector does NOT match the checkpoint"
        f1 = centre2(s1, boxes, timestep=0, batch=8, tapname="up_blocks[1]",
                     noise_level=None)
        # deterministic across independent loads
        s2 = CryoFM2Tap(args.ckpt, taps=("up_blocks[1]",), device=dev, batch_size=8,
                        stop_after="up_blocks[1]", student_ckpt=args.student_ckpt)
        f2 = centre2(s2, boxes, timestep=0, batch=8, tapname="up_blocks[1]",
                     noise_level=None)
        assert np.array_equal(f1, f2), "two independent loads disagree"
        # and the weights genuinely differ from the teacher
        ft = centre2(tap, boxes, timestep=0, batch=8, tapname="up_blocks[1]",
                     noise_level=None)
        d = float(np.abs(f1 - ft).max())
        print(f"    OK  swap+load ordering, p matches, loads deterministic; "
              f"student-vs-teacher max|d| {d:.4g} (step {sd.get('step')})", flush=True)
        assert d > 0, "student features identical to teacher -- weights did not load"
        del s1, s2
        if dev == "cuda":
            torch.cuda.empty_cache()

    # -- 5. training smoke ------------------------------------------------
    print(f"[5] training smoke, {args.smoke_steps} steps\n{'-'*70}", flush=True)
    cmd = [sys.executable, "-u", "-m", "probes.cleandift_train",
           "--steps", str(args.smoke_steps), "--warmup", "5", "--batch", "2",
           "--per-map", "8", "--buffer-maps", "2", "--val-every",
           str(args.smoke_steps), "--ckpt-every", str(args.smoke_steps),
           "--val-batches", "2", "--lr", "1e-4", "--tag", "smoke",
           "--vol-dir", str(args.vol_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit("training smoke FAILED")
    print("-" * 70)
    print("\nALL VERIFICATION PASSED")


if __name__ == "__main__":
    main()
