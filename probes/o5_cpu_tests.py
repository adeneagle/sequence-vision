"""Fast CPU tests for the CleanDIFT pieces. Run BEFORE queueing a GPU job.

Written after a `.view()`-on-a-`chunk()`-output bug in `FiLMHead` cost a full
cache-plus-GPU round trip to discover: the CPU tests at the time covered the loss
and the timestep sampler but not the head, and the failure was in the gap. Anything
that can be checked without a GPU is checked here.

    python -m probes.o5_cpu_tests
"""

from __future__ import annotations

import numpy as np
import torch


def test_stats() -> None:
    from probes.o5_stats import (cluster_bootstrap_diff, mcnemar_exact, mdi,
                                 variance_split)
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(100), 20)
    base = rng.random(100)[cl]
    a = rng.random(2000) < np.clip(base, .05, .95)
    b = rng.random(2000) < np.clip(base - 0.02, .05, .95)
    r = cluster_bootstrap_diff(a, b, cl)
    assert r["n_clusters"] == 100 and r["n_residues"] == 2000
    assert r["lo95"] < r["diff"] < r["hi95"]
    assert mdi(r) > 0
    # identical arms: exactly zero, CI must contain zero
    r0 = cluster_bootstrap_diff(a, a, cl)
    assert r0["diff"] == 0.0 and not r0["excludes_zero"]
    # a p-value must never be reported as 0
    assert r0["p"] > 0
    v = variance_split(a, b, cl)
    assert 0.0 <= v["between_frac"] <= 1.0
    m = mcnemar_exact(a, b)
    assert m["a_only"] >= 0 and m["b_only"] >= 0
    # length mismatch must raise, not broadcast silently
    try:
        cluster_bootstrap_diff(a[:10], b, cl)
        raise AssertionError("length mismatch did not raise")
    except ValueError:
        pass
    print("stats: OK (incl. degenerate input and length-mismatch guard)")


def test_loss() -> None:
    from probes.cleandift_train import DistillLoss
    taps = ("mid_block", "up_blocks[0]", "up_blocks[1]")
    shapes = {"mid_block": (4, 512, 8, 8, 8), "up_blocks[0]": (4, 512, 16, 16, 16),
              "up_blocks[1]": (4, 256, 32, 32, 32)}
    g = torch.Generator().manual_seed(0)
    P = {k: torch.randn(*shapes[k], generator=g) for k in taps}
    loss, d = DistillLoss(taps)(P, P, {k: v.clone() for k, v in P.items()})
    assert abs(float(loss) + 1.0) < 1e-4, f"identical inputs gave {float(loss)}"

    # THE D5 HAZARD: a shared common component gives uncentred ~1 and centred ~0.
    lf = DistillLoss(taps)
    common = torch.randn(1, 256, 1, 1, 1, generator=g) * 20
    A = common + torch.randn(*shapes["up_blocks[1]"], generator=g)
    B = common + torch.randn(*shapes["up_blocks[1]"], generator=g)
    for _ in range(50):
        t = lf.terms("up_blocks[1]", A, B, update=True)
    assert float(t["raw"]) > 0.9 and abs(float(t["tok"])) < 0.2, \
        f"centring not isolating the residual: raw {float(t['raw'])}, " \
        f"centred {float(t['tok'])}"

    # centre pooling must read the central 2^3 cells at every tap resolution
    for D in (8, 16, 32):
        x = torch.zeros(1, 2, D, D, D)
        c = D // 2
        x[0, :, c - 1:c + 1, c - 1:c + 1, c - 1:c + 1] = 7.0
        assert torch.allclose(DistillLoss._pool_centre(x), torch.full((1, 2), 7.0)), D

    # gradient must reach the student side and NOT the teacher side
    Pg = {k: torch.randn(*shapes[k], generator=g, requires_grad=True) for k in taps}
    T = {k: torch.randn(*shapes[k], generator=g) for k in taps}
    l2, _ = DistillLoss(taps)(Pg, None, T)
    l2.backward()
    assert all(Pg[k].grad is not None and Pg[k].grad.norm() > 0 for k in taps)
    print("loss: OK (identity=-1, centring isolates residual, pooling, grads)")


def test_sampler() -> None:
    from probes.cleandift_train import sample_t
    r = np.random.default_rng(0)
    ts = np.concatenate([sample_t(9, 1, 600, 3, r) for _ in range(200)])
    assert ts.min() >= 1 and ts.max() <= 600, (ts.min(), ts.max())
    h = np.histogram(ts, bins=[1, 201, 401, 601])[0]
    assert h.min() > 0 and h.max() / h.min() < 1.15, h
    print(f"sample_t: OK (bins {h.tolist()}, range {ts.min()}-{ts.max()})")


def test_head() -> None:
    from teachers.cryofm_student import FiLMHead, LearnedTimeEmb
    for C, D in ((256, 32), (512, 8), (64, 16)):
        h = FiLMHead(C, ratio=4)
        f = torch.randn(3, C, D, D, D)
        out = h(f, torch.randn(3, 256))
        assert out.shape == f.shape
        # identity at init is what keeps the centred cosine defined at step 0
        assert torch.equal(out, f), f"C={C} head not identity at init"
    h = FiLMHead(256)
    torch.nn.init.normal_(h.conv2.weight, std=0.02)
    f = torch.randn(2, 256, 4, 4, 4, requires_grad=True)
    o = h(f, torch.randn(2, 256))
    assert not torch.equal(o, f)
    o.pow(2).mean().backward()
    assert f.grad is not None and h.film.weight.grad is not None
    # non-contiguous t_emb: the exact shape that raised on GPU
    tnc = torch.randn(2, 512)[:, ::2]
    assert not tnc.is_contiguous()
    FiLMHead(64)(torch.randn(2, 64, 8, 8, 8), tnc)
    e = LearnedTimeEmb(torch.arange(256).float())
    o = e(torch.zeros(5, 64))
    assert o.shape == (5, 256) and torch.equal(o[0], o[4])
    assert o.stride(0) != 0, "shared buffer -- .expand instead of .repeat"
    print("head: OK (identity at init, grads, non-contiguous cond, no shared buffer)")


def test_split() -> None:
    from pathlib import Path

    from probes.cleandift_data import split_map_lists
    p = Path("data/alignment_chains.csv")
    if not p.exists():
        print("split: SKIPPED (no alignment_chains.csv)")
        return
    i = split_map_lists(p)["info"]
    assert i["train_maps_clean"] == 798 and i["train_chains"] == 958, i
    assert i["dropped_shared_with_test"] == 64 and i["dropped_shared_with_val"] == 50, i
    print(f"split: OK ({i['train_chains']} chains / {i['train_maps_clean']} maps / "
          f"{i['train_clusters']} clusters, D11 assertions hold)")


def main() -> None:
    for fn in (test_stats, test_loss, test_sampler, test_head, test_split):
        fn()
    print("\nALL CPU TESTS PASSED")


if __name__ == "__main__":
    main()
