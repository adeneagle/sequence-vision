"""The sibling project's spatial-variability analyses, run on CleanDIFT latents.

The ESM half of this repo concluded that ESM-C per-residue embeddings carry
essentially no COARSE within-protein spatial variance (50 A band 0.1%, 25 A 0.4%,
10 A 3%, local 86%) and that one residue's embedding is ~unpredictable from
another beyond ~10-15 A -- and that the only remaining source of genuine
multi-scale spatial features would be a STRUCTURE-NATIVE encoder. CryoFM2 /
CleanDIFT features are exactly that, so the same three measurements are the
direct test of that prediction.

  A  variance hierarchy   between-protein vs within-protein, and the within part
                          split into Laplacian-pyramid bands at 50/25/10 A + local
                          (`scripts/pyramid_features.compute_pyramid_bands`)
  B  predictability vs distance   ridge R2 predicting residue j's feature from
                          residue i's, binned by 3D distance and by sequence
                          separation, with the |i-j|>8 through-space control
  C  neighbourhood -> residue     ridge R2 from mean+std of all through-space
                          neighbours within radius r
  E  isolated nonlinear gap  fit ridge FIRST, then an MLP to the ridge RESIDUAL.
                          gap = combined R2 - ridge R2 = the nonlinear signal with
                          the linear part removed. The sibling's fairest test: it
                          found the only genuine nonlinearity was sequence-adjacent
                          and that through-space gaps were zero or negative.
  D  spatial correlogram   per-protein-centred cosine similarity vs 3D distance,
                          and the half-decay length L -- the sibling's
                          `scale_segregation.py` statistic. Reported for the raw
                          feature and for each pyramid band, so a band's L can be
                          checked against the scale it is supposed to carry.

READ THE CONTROLS BEFORE THE NUMBERS. A density feature is a function of a box
of density around the residue (measured half-decay 4-5 A at `up_blocks[1]`,
~10 A at `up_blocks[0]`, ~15 A at `mid_block`), so neighbouring residues share
input by construction and BOTH coarse band energy and short-range predictability
are partly built in -- the same coordinate-leakage class that inflated the
pairwise contact probe to AUC 0.997. Two arms exist to price it:

  `rand_student`  identical architecture and receptive field, RANDOM weights.
                  Whatever it scores is reachable with no learned content.
  `raw`           the 8^3 raw density voxels at the Ca. The trivial baseline.

`esmc` is on the SAME residues of the SAME chains, so the cross-modal comparison
is matched. `esmc_all` repeats it at full residue density (every observed
residue, not the 60 sampled ones) and so prices the SUBSAMPLING distortion that
the density arms cannot avoid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Tiny GEMMs (60x60 smoothing kernels) on a 96-core box spend all their time in
# OpenMP barriers: analysis A took >60 s at the default thread count and 7.1 s at
# 8. Set before numpy is imported or it has no effect.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
from numpy.linalg import solve

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pyramid_features import compute_pyramid_bands   # noqa: E402

EDGES3D = [0, 5, 10, 15, 20, 25, 30, 40, 50]
SEDGES = [1, 2, 4, 8, 16, 32, 64, 128, 10 ** 9]
RADII = [10.0, 15.0, 20.0, 30.0, 50.0]
CORR_EDGES = np.arange(0.0, 62.0, 2.0)
SIGMAS = (50.0, 25.0, 10.0)
BANDS = ("50.0Å", "25.0Å", "10.0Å", "private")
# RELATIVE ridge penalties: alpha = a * trace(XtX)/D. Feature RMS across the arms
# spans 0.036 (esmc) to 17.0 (`up_blocks[0]`), a 480x range, so a grid of absolute
# alphas regularises the arms by wildly different amounts -- the density arms came
# out at R2 -0.06..-0.17 in every long-distance bin purely because the largest
# absolute alpha was still far too small for them. Scaling by the data makes the
# grid scale-invariant and the arms comparable.
RELALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

DEFAULT_ARMS = (
    "esmc", "esmc_all", "raw", "rand_student@up_blocks[0]",
    "dec_t500@mid_block", "dec_t500@up_blocks[0]", "dec_t500@up_blocks[1]",
    "student@up_blocks[0]", "student_paperhead@up_blocks[0]",
)


# --------------------------------------------------------------------------- io
class Corpus:
    """Per-chain (features, Ca coords, residue indices, split) for one arm."""

    def __init__(self, coords_npz: Path, feat_dir: Path, esmc_dir: Path, limit: int):
        self.z = np.load(coords_npz, allow_pickle=True)
        self.keys = [str(k) for k in self.z["keys"]][:limit]
        self.feat_dir, self.esmc_dir = feat_dir, esmc_dir
        self.split = {}
        for k in self.keys:
            self.split[k] = str(np.load(feat_dir / f"{k}.npz", allow_pickle=True)["split"][0])

    def chain(self, key: str, arm: str):
        idx = self.z[f"{key}/idx"]
        if arm == "esmc_all":
            E = np.load(self.esmc_dir / f"{key}.npy")
            return E, self.z[f"{key}/ca_all"], np.arange(len(E))
        ca = self.z[f"{key}/ca"]
        if arm == "esmc":
            return np.load(self.esmc_dir / f"{key}.npy")[idx], ca, idx
        return np.load(self.feat_dir / f"{key}.npz", allow_pickle=True)[arm], ca, idx


# ------------------------------------------------------- A: variance hierarchy
def variance_hierarchy(corp: Corpus, arm: str, min_res: int = 20) -> dict:
    """between/within-protein ANOVA + within-protein pyramid band shares."""
    tot_sum, tot_n, chains = None, 0, []
    for k in corp.keys:
        E, ca, _ = corp.chain(k, arm)
        if len(E) < min_res:
            continue
        E = np.asarray(E, np.float64)
        chains.append((k, E, np.asarray(ca, np.float64)))
        tot_sum = E.sum(0) if tot_sum is None else tot_sum + E.sum(0)
        tot_n += len(E)
    gmean = tot_sum / tot_n
    total = sum(float(((E - gmean) ** 2).sum()) for _, E, _ in chains)
    between = sum(len(E) * float(((E.mean(0) - gmean) ** 2).sum()) for _, E, _ in chains)
    within = total - between
    band_ss = {b: 0.0 for b in BANDS}
    for _, E, ca in chains:
        bands = compute_pyramid_bands(E, ca, distances=SIGMAS, include_input=False)
        for b in BANDS:
            bb = bands[b] - bands[b].mean(0)          # within-protein variance only
            band_ss[b] += float((bb ** 2).sum())
    lab = {"50.0Å": "50A", "25.0Å": "25A", "10.0Å": "10A", "private": "local"}
    return {
        "n_chains": len(chains), "n_residues": tot_n, "dim": int(chains[0][1].shape[1]),
        "frac_total": {"global": between / total,
                       **{lab[b]: band_ss[b] / total for b in BANDS}},
        "frac_within": {lab[b]: band_ss[b] / within for b in BANDS},
        "between": between / total, "within": within / total,
    }


# ----------------------------------------------------------- ridge (from sibling)
def pdist(x):
    sq = (x * x).sum(1)
    return np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0, None))


def ridge_r2(Xtr, Ytr, Xva, Yva, Xte, Yte):
    """Ridge R2 on test, penalty chosen on val. Returns (r2, chosen relative alpha)."""
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = (Xtr - xm).astype(np.float64), (Ytr - ym).astype(np.float64)
    XtX, XtY = Xc.T @ Xc, Xc.T @ Yc
    D = XtX.shape[0]
    scale = float(np.trace(XtX)) / D
    eye = np.eye(D)
    best = None
    for a in RELALPHAS:
        W = solve(XtX + a * scale * eye, XtY)
        pv = (Xva - xm) @ W + ym
        r = 1 - ((Yva - pv) ** 2).sum() / ((Yva - ym) ** 2).sum()
        if best is None or r > best[0]:
            best = (r, W, a)
    _, W, a = best
    pt = (Xte - xm) @ W + ym
    return float(1 - ((Yte - pt) ** 2).sum() / ((Yte - ym) ** 2).sum()), a


def _filter(tr, va, te, seq_exclude):
    """Apply the |i-j| > seq_exclude through-space cut to all three splits."""
    def cut(x):
        X, Y, D = x
        if seq_exclude is None:
            return X, Y
        m = D > seq_exclude
        return X[m], Y[m]
    return (*cut(tr), *cut(va), *cut(te))


def _eval(tr, va, te, min_train, seq_exclude=None):
    Xtr, Ytr, Xva, Yva, Xte, Yte = _filter(tr, va, te, seq_exclude)
    if len(Xtr) < min_train or len(Xte) < 200 or len(Xva) < 50:
        return float("nan"), len(Xte), float("nan")
    r2, a = ridge_r2(Xtr, Ytr, Xva, Yva, Xte, Yte)
    return r2, len(Xte), a


# ------------------------------------------ B: predictability vs 3D / seq distance
def gather_pairs(corp, arm, keys, cap, rng):
    nb, nsb = len(EDGES3D) - 1, len(SEDGES) - 1
    B = [[[], [], []] for _ in range(nb)]
    S = [[[], [], []] for _ in range(nsb)]
    D = None
    for k in keys:
        E, ca, idx = corp.chain(k, arm)
        L = len(E)
        if L < 10:
            continue
        E = np.asarray(E, np.float32)
        E = E - E.mean(0)                       # per-protein centring (sibling)
        D = E.shape[1]
        d3 = pdist(np.asarray(ca, np.float64))
        iu, ju = np.triu_indices(L, 1)
        d3v, dsv = d3[iu, ju], np.abs(idx[iu] - idx[ju])
        for b in range(nb):
            ix = np.where((d3v >= EDGES3D[b]) & (d3v < EDGES3D[b + 1]))[0]
            if not len(ix):
                continue
            if len(ix) > cap:
                ix = rng.choice(ix, cap, replace=False)
            B[b][0].append(E[iu[ix]]); B[b][1].append(E[ju[ix]]); B[b][2].append(dsv[ix])
        for b in range(nsb):
            ix = np.where((dsv >= SEDGES[b]) & (dsv < SEDGES[b + 1]))[0]
            if not len(ix):
                continue
            if len(ix) > cap:
                ix = rng.choice(ix, cap, replace=False)
            S[b][0].append(E[iu[ix]]); S[b][1].append(E[ju[ix]]); S[b][2].append(dsv[ix])

    def pack(slot):
        if not slot[0]:
            return (np.zeros((0, D), np.float32), np.zeros((0, D), np.float32),
                    np.zeros(0, int))
        return (np.concatenate(slot[0]), np.concatenate(slot[1]),
                np.concatenate(slot[2]))
    return [pack(b) for b in B], [pack(s) for s in S]


# ------------------------------------------------- C: neighbourhood -> residue
def gather_neigh(corp, arm, keys, seq_exclude, min_nb, cap, rng):
    X = [[] for _ in RADII]; Y = [[] for _ in RADII]
    D = None
    for k in keys:
        E, ca, idx = corp.chain(k, arm)
        L = len(E)
        if L < 20:
            continue
        E = np.asarray(E, np.float32); E = E - E.mean(0); D = E.shape[1]
        d = pdist(np.asarray(ca, np.float64))
        tgt = np.arange(L) if L <= cap else rng.choice(L, cap, replace=False)
        for j in tgt:
            sm = np.abs(idx - idx[j]) > seq_exclude
            dj = d[j]
            for ri, r in enumerate(RADII):
                nb = np.where(sm & (dj > 0) & (dj <= r))[0]
                if len(nb) < min_nb:
                    continue
                En = E[nb]
                X[ri].append(np.concatenate([En.mean(0), En.std(0)])); Y[ri].append(E[j])
    out = []
    for ri in range(len(RADII)):
        if not X[ri]:
            out.append((np.zeros((0, 2 * D), np.float32), np.zeros((0, D), np.float32),
                        np.zeros(0, int)))
        else:
            out.append((np.asarray(X[ri], np.float32), np.asarray(Y[ri], np.float32),
                        np.zeros(len(X[ri]), int)))
    return out


# --------------------------------------------------- E: isolated nonlinear gap
# (hidden, weight_decay, dropout), val-selected per bin -- regularisation-fair, which
# the sibling found mattered: an untuned MLP scored BELOW ridge on restricted bins and
# that dip was a fitting artifact, not masked nonlinearity.
GRID = [(1024, 1e-4, 0.1), (2048, 1e-4, 0.1), (2048, 1e-3, 0.1)]


def mlp_predict(Xtr, Ytr, Xva, Yva, Xte, device, hidden, wd, dropout,
                epochs=150, patience=12, lr=1e-3):
    """Fit MLP X->Y with val early-stopping. Returns (test_pred in target units, val R2)."""
    import torch
    from torch import nn
    xm, xs = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym = Ytr.mean(0, keepdims=True)
    t = lambda A: torch.tensor(A, dtype=torch.float32, device=device)
    Xtr_t, Ytr_t = t((Xtr - xm) / xs), t(Ytr - ym)
    Xva_t, Yva_t = t((Xva - xm) / xs), t(Yva - ym)
    Xte_t = t((Xte - xm) / xs)
    bs = int(np.clip(len(Xtr) // 10, 256, 4096))
    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, Ytr.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    n, best_va, best_state, bad = len(Xtr_t), None, None, 0
    va_denom = (Yva_t ** 2).sum()
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            ((net(Xtr_t[idx]) - Ytr_t[idx]) ** 2).mean().backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            va = (1 - ((Yva_t - net(Xva_t)) ** 2).sum() / va_denom).item()
        if best_va is None or va > best_va:
            best_va, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    import torch as _t
    with _t.no_grad():
        return net(Xte_t).cpu().numpy() + ym, best_va


def mlp_tuned(Xtr, Ytr, Xva, Yva, Xte, device):
    best = None
    for hidden, wd, dropout in GRID:
        pt, va = mlp_predict(Xtr, Ytr, Xva, Yva, Xte, device, hidden, wd, dropout)
        if best is None or va > best[0]:
            best = (va, pt, (hidden, wd, dropout))
    return best[1], best[2]


def ridge_fit(Xtr, Ytr, Xva, Yva):
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = (Xtr - xm).astype(np.float64), (Ytr - ym).astype(np.float64)
    XtX, XtY = Xc.T @ Xc, Xc.T @ Yc
    D = XtX.shape[0]
    scale = float(np.trace(XtX)) / D
    eye = np.eye(D)
    best = None
    for a in RELALPHAS:
        W = solve(XtX + a * scale * eye, XtY)
        pv = (Xva - xm) @ W + ym
        r = 1 - ((Yva - pv) ** 2).sum() / ((Yva - ym) ** 2).sum()
        if best is None or r > best[0]:
            best = (r, W)
    return best[1], xm, ym


def eval_gap(tr, va, te, device, min_train, seq_exclude=None):
    """(ridge R2, combined R2, gap, n_test). MLP is fit to the RIDGE RESIDUAL."""
    Xtr, Ytr, Xva, Yva, Xte, Yte = _filter(tr, va, te, seq_exclude)
    nan = float("nan")
    if len(Xtr) < min_train or len(Xte) < 200 or len(Xva) < 50:
        return nan, nan, nan, len(Xte)
    W, xm, ym = ridge_fit(Xtr, Ytr, Xva, Yva)
    P = lambda Z: (Z - xm) @ W + ym
    Ptr, Pva, Pte = P(Xtr), P(Xva), P(Xte)
    Rte_hat, _ = mlp_tuned(Xtr, Ytr - Ptr, Xva, Yva - Pva, Xte, device)
    denom = ((Yte - ym) ** 2).sum()
    ridge = float(1 - ((Yte - Pte) ** 2).sum() / denom)
    comb = float(1 - ((Yte - (Pte + Rte_hat)) ** 2).sum() / denom)
    return ridge, comb, comb - ridge, len(Xte)


# ------------------------------------------------------- D: spatial correlogram
def _half_decay(centres, curve, counts, min_pairs=200):
    """Distance at which the correlogram falls to half its shortest-range value.

    The reference is the first POPULATED bin, not bin 0: the closest possible
    Ca-Ca pair is ~3.8 A, so the 0-2 A bin is always empty and using it returned
    NaN for every arm.

    Linear interpolation inside the crossing bin, as `scale_segregation.py` does.
    Returns (L, censored, r0): censored=True means the curve never reached half by
    the last bin, so L is a LOWER BOUND -- the sibling logged that silently
    treating a censored L as a point estimate penalises exactly the
    slow-decorrelating (i.e. genuinely coarse) features, so the flag is carried.
    """
    ok = [i for i in range(len(curve))
          if np.isfinite(curve[i]) and counts[i] >= min_pairs]
    if not ok:
        return float("nan"), False, float("nan")
    i0 = ok[0]
    c0 = curve[i0]
    if c0 <= 0:
        return float("nan"), False, float(centres[i0])
    half = c0 / 2.0
    prev = i0
    for i in ok[1:]:
        if curve[i] <= half:
            x0, x1 = centres[prev], centres[i]
            y0, y1 = curve[prev], curve[i]
            t = 0.0 if y0 == y1 else (y0 - half) / (y0 - y1)
            return float(x0 + t * (x1 - x0)), False, float(centres[i0])
        prev = i
    return float(centres[ok[-1]]), True, float(centres[i0])


def correlogram(corp: Corpus, arm: str, keys, min_res: int = 20) -> dict:
    """Centred-cosine similarity vs 3D distance, for the feature and each band.

    CAVEAT carried from the sibling: removing the per-protein mean puts a ceiling
    of roughly diameter/4 (~17 A on typical chains) on the measurable half-decay,
    so two genuinely coarse scales can be indistinguishable here REGARDLESS of the
    representation. Read L as ordinal, not absolute.
    """
    nb = len(CORR_EDGES) - 1
    series = ["feature"] + list(BANDS)
    ssum = {k: np.zeros(nb) for k in series}
    scnt = {k: np.zeros(nb, dtype=np.int64) for k in series}
    n_ch = 0
    for k in keys:
        E, ca, _ = corp.chain(k, arm)
        if len(E) < min_res:
            continue
        n_ch += 1
        E = np.asarray(E, np.float64)
        ca = np.asarray(ca, np.float64)
        d = pdist(ca)
        iu, ju = np.triu_indices(len(E), 1)
        dv = d[iu, ju]
        b = np.digitize(dv, CORR_EDGES) - 1
        keep = (b >= 0) & (b < nb)
        bands = compute_pyramid_bands(E, ca, distances=SIGMAS, include_input=False)
        for name in series:
            V = E if name == "feature" else bands[name]
            V = V - V.mean(0)                                # per-protein centring
            nrm = np.linalg.norm(V, axis=1)
            nrm[nrm == 0] = 1.0
            V = V / nrm[:, None]
            cv = (V[iu] * V[ju]).sum(1)
            np.add.at(ssum[name], b[keep], cv[keep])
            np.add.at(scnt[name], b[keep], 1)
    centres = (CORR_EDGES[:-1] + CORR_EDGES[1:]) / 2.0
    out = {"n_chains": n_ch, "edges": CORR_EDGES.tolist(), "series": {}}
    for name in series:
        with np.errstate(invalid="ignore", divide="ignore"):
            curve = np.where(scnt[name] > 0, ssum[name] / np.maximum(scnt[name], 1),
                             np.nan)
        L, cens, r0 = _half_decay(centres, curve, scnt[name])
        out["series"][name] = {"curve": [None if not np.isfinite(x) else float(x)
                                         for x in curve],
                               "n_pairs": scnt[name].tolist(),
                               "half_decay": L, "censored": bool(cens),
                               "ref_bin_centre": r0,
                               "ref_cos": (float(curve[int(r0 // 2)])
                                           if np.isfinite(r0) else float("nan"))}
    return out


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", type=Path, default=Path("data/o7_coords.npz"))
    ap.add_argument("--feat-dir", type=Path, default=Path("data/o5_multitap_parts"))
    ap.add_argument("--esmc-dir", type=Path, default=Path("data/esmc_chains"))
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--analyses", default="A,B,C")
    ap.add_argument("--limit", type=int, default=10 ** 9)
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--neigh-cap", type=int, default=60)
    ap.add_argument("--seq-exclude", type=int, default=8)
    ap.add_argument("--min-nb", type=int, default=3)
    ap.add_argument("--min-train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/o7_spatial_variability.json"))
    args = ap.parse_args()

    corp = Corpus(args.coords, args.feat_dir, args.esmc_dir, args.limit)
    arms = [a for a in args.arms.split(",") if a]
    todo = set(args.analyses.split(","))
    sp = {s: [k for k in corp.keys if corp.split[k] == s] for s in ("train", "val", "test")}
    print(f"{len(corp.keys)} chains | train {len(sp['train'])} val {len(sp['val'])} "
          f"test {len(sp['test'])} | arms {arms}", flush=True)
    for s, v in sp.items():
        if not v:
            raise SystemExit(f"FAILED: empty {s} split -- ridge needs all three.")
    res: dict = {"config": {k: str(v) for k, v in vars(args).items()},
                 "n_chains": len(corp.keys),
                 "split_chains": {s: len(v) for s, v in sp.items()}, "arms": {}}

    for arm in arms:
        a: dict = {}
        if "A" in todo:
            a["variance"] = variance_hierarchy(corp, arm)
            v = a["variance"]
            print(f"\n[A] {arm:32s} dim={v['dim']:5d} n={v['n_residues']:7d}  "
                  f"global {100*v['between']:5.2f}%  within {100*v['within']:5.2f}%",
                  flush=True)
            print("    % of TOTAL   " + "  ".join(
                f"{k} {100*x:7.3f}" for k, x in v["frac_total"].items()))
            print("    % of WITHIN  " + "  ".join(
                f"{k} {100*x:7.3f}" for k, x in v["frac_within"].items()))
        if "B" in todo:
            rng = np.random.default_rng(args.seed)
            g = {s: gather_pairs(corp, arm, sp[s], args.cap, rng) for s in sp}
            rows3, rowsq = [], []
            print(f"\n[B] {arm}  predict E_j from E_i")
            print(f"{'3D dist (A)':>12} {'n_test':>8} {'R2(all)':>9} "
                  f"{'R2(|i-j|>8)':>12} {'n*':>8}")
            for b in range(len(EDGES3D) - 1):
                r, n, al = _eval(g["train"][0][b], g["val"][0][b], g["test"][0][b],
                                 args.min_train)
                rs, ns, _ = _eval(g["train"][0][b], g["val"][0][b], g["test"][0][b],
                                  args.min_train, seq_exclude=args.seq_exclude)
                lab = f"{EDGES3D[b]}-{EDGES3D[b+1]}"
                rows3.append({"bin": lab, "n_test": n, "r2": r, "r2_seqex": rs,
                              "n_test_seqex": ns, "alpha_rel": al})
                print(f"{lab:>12} {n:>8} {r:>9.3f} {rs:>12.3f} {ns:>8}", flush=True)
            print(f"{'seq sep':>12} {'n_test':>8} {'R2':>9}")
            for b in range(len(SEDGES) - 1):
                r, n, _ = _eval(g["train"][1][b], g["val"][1][b], g["test"][1][b],
                                args.min_train)
                hi = SEDGES[b + 1]
                lab = f"{SEDGES[b]}-{'inf' if hi >= 10**9 else hi}"
                rowsq.append({"bin": lab, "n_test": n, "r2": r})
                print(f"{lab:>12} {n:>8} {r:>9.3f}", flush=True)
            a["dist3d"], a["seqsep"] = rows3, rowsq
            del g
        if "C" in todo:
            rng = np.random.default_rng(args.seed)
            g = {s: gather_neigh(corp, arm, sp[s], args.seq_exclude, args.min_nb,
                                 args.neigh_cap, rng) for s in sp}
            rows = []
            print(f"\n[C] {arm}  neighbourhood(mean+std, |i-j|>{args.seq_exclude}) -> residue")
            print(f"{'radius (A)':>11} {'n_test':>8} {'R2':>9}")
            for ri, r_ in enumerate(RADII):
                r, n, _ = _eval(g["train"][ri], g["val"][ri], g["test"][ri],
                                args.min_train)
                rows.append({"radius": r_, "n_test": n, "r2": r})
                print(f"{r_:>11.0f} {n:>8} {r:>9.3f}", flush=True)
            a["neighbourhood"] = rows
            del g
        if "E" in todo:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            rng = np.random.default_rng(args.seed)
            g = {s_: gather_pairs(corp, arm, sp[s_], args.cap, rng) for s_ in sp}
            rows3, rowsq = [], []
            print(f"\n[E] {arm}  NONLINEAR GAP (ridge -> MLP on residual)  device={dev}")
            print(f"{'3D dist (A)':>12} {'n_test':>8} {'ridge':>8} {'+MLP':>8} {'gap':>8}  | "
                  f"{'ridge*':>8} {'+MLP*':>8} {'gap*':>8} {'n*':>8}   (* = |i-j|>8)")
            for b in range(len(EDGES3D) - 1):
                r, c, gp, n = eval_gap(g["train"][0][b], g["val"][0][b], g["test"][0][b],
                                       dev, args.min_train)
                rs, cs, gs, ns = eval_gap(g["train"][0][b], g["val"][0][b], g["test"][0][b],
                                          dev, args.min_train, seq_exclude=args.seq_exclude)
                lab = f"{EDGES3D[b]}-{EDGES3D[b+1]}"
                rows3.append({"bin": lab, "n_test": n, "ridge": r, "combined": c, "gap": gp,
                              "ridge_seqex": rs, "combined_seqex": cs, "gap_seqex": gs,
                              "n_test_seqex": ns})
                print(f"{lab:>12} {n:>8} {r:>8.3f} {c:>8.3f} {gp:>8.3f}  | "
                      f"{rs:>8.3f} {cs:>8.3f} {gs:>8.3f} {ns:>8}", flush=True)
            print(f"{'seq sep':>12} {'n_test':>8} {'ridge':>8} {'+MLP':>8} {'gap':>8}")
            for b in range(len(SEDGES) - 1):
                r, c, gp, n = eval_gap(g["train"][1][b], g["val"][1][b], g["test"][1][b],
                                       dev, args.min_train)
                hi = SEDGES[b + 1]
                lab = f"{SEDGES[b]}-{'inf' if hi >= 10**9 else hi}"
                rowsq.append({"bin": lab, "n_test": n, "ridge": r, "combined": c, "gap": gp})
                print(f"{lab:>12} {n:>8} {r:>8.3f} {c:>8.3f} {gp:>8.3f}", flush=True)
            a["gap_dist3d"], a["gap_seqsep"] = rows3, rowsq
            del g
            rng = np.random.default_rng(args.seed)
            g = {s_: gather_neigh(corp, arm, sp[s_], args.seq_exclude, args.min_nb,
                                  args.neigh_cap, rng) for s_ in sp}
            rows = []
            print(f"{'radius (A)':>11} {'n_test':>8} {'ridge':>8} {'+MLP':>8} {'gap':>8}")
            for ri, r_ in enumerate(RADII):
                r, c, gp, n = eval_gap(g["train"][ri], g["val"][ri], g["test"][ri],
                                       dev, args.min_train)
                rows.append({"radius": r_, "n_test": n, "ridge": r, "combined": c, "gap": gp})
                print(f"{r_:>11.0f} {n:>8} {r:>8.3f} {c:>8.3f} {gp:>8.3f}", flush=True)
            a["gap_neighbourhood"] = rows
            del g
        if "D" in todo:
            a["correlogram"] = correlogram(corp, arm, corp.keys)
            print(f"\n[D] {arm}  centred-cosine half-decay L (A)")
            for nm, v in a["correlogram"]["series"].items():
                flag = "  (CENSORED: lower bound)" if v["censored"] else ""
                print(f"    {nm:>10s}  L = {v['half_decay']:6.2f}   "
                      f"cos@{v['ref_bin_centre']:.0f}A = {v['ref_cos']:.4f}{flag}")
        res["arms"][arm] = a

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
