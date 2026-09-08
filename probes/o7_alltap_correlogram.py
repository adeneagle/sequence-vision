"""Centred-cosine correlogram for EVERY layer of the CleanDIFT student, in one pass.

Why this exists rather than `o7_spatial_variability --analyses D`: that path calls
`Corpus.chain` per (chain, arm), so a 20-arm sweep re-opens all 1,484 part files 20
times -- 29,680 reads over a network filesystem, which dominated the runtime at
~10 min per arm. Here each chain's npz is opened ONCE and every arm is accumulated
from that single read.

It also computes only the `feature` series, not the pyramid bands. The band columns
are metric-limited -- per-protein mean removal caps measurable L at ~diameter/4, so
50 A and 25 A read the same for every arm INCLUDING random weights -- so they carry
no layer information while costing 5x the compute.

Output uses the same schema `o7_report` and `make_o7_figures` already read.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np

EDGES = np.arange(0.0, 62.0, 2.0)
TAPS = ("conv_in", "down_blocks[0]", "down_blocks[1]", "down_blocks[2]",
        "down_blocks[3]", "mid_block", "up_blocks[0]", "up_blocks[1]",
        "up_blocks[2]", "up_blocks[3]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", type=Path, default=Path("data/o7_coords.npz"))
    ap.add_argument("--feat-dir", type=Path, default=Path("data/o7_alltap_parts"))
    ap.add_argument("--arms", default="student,rand_student")
    ap.add_argument("--min-res", type=int, default=20)
    ap.add_argument("--out", type=Path,
                    default=Path("results/o7_correlogram_alltap.json"))
    args = ap.parse_args()

    z = np.load(args.coords, allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    arms = [f"{a}@{t}" for a in args.arms.split(",") for t in TAPS]
    nb = len(EDGES) - 1
    ssum = {a: np.zeros(nb) for a in arms}
    scnt = {a: np.zeros(nb, dtype=np.int64) for a in arms}
    n_ch = 0
    print(f"{len(keys)} chains | {len(arms)} arms | one npz open per chain", flush=True)

    for i, k in enumerate(keys):
        part = args.feat_dir / f"{k}.npz"
        if not part.exists():
            continue
        ca = np.asarray(z[f"{k}/ca"], np.float64)
        if len(ca) < args.min_res:
            continue
        sq = (ca * ca).sum(1)
        d = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * ca @ ca.T, 0, None))
        iu, ju = np.triu_indices(len(ca), 1)
        b = np.digitize(d[iu, ju], EDGES) - 1
        keep = (b >= 0) & (b < nb)
        bk, iuk, juk = b[keep], iu[keep], ju[keep]
        with np.load(part, allow_pickle=True) as zz:      # ONE open per chain
            n_ch += 1
            for a in arms:
                V = np.asarray(zz[a], np.float64)
                V = V - V.mean(0)                         # per-protein centring
                nrm = np.linalg.norm(V, axis=1)
                nrm[nrm == 0] = 1.0
                V /= nrm[:, None]
                cv = (V[iuk] * V[juk]).sum(1)
                np.add.at(ssum[a], bk, cv)
                np.add.at(scnt[a], bk, 1)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(keys)}] {n_ch} chains", flush=True)

    centres = (EDGES[:-1] + EDGES[1:]) / 2.0
    out = {"n_chains": n_ch, "arms": {}}
    for a in arms:
        with np.errstate(invalid="ignore", divide="ignore"):
            curve = np.where(scnt[a] > 0, ssum[a] / np.maximum(scnt[a], 1), np.nan)
        ok = [j for j in range(nb) if np.isfinite(curve[j]) and scnt[a][j] >= 200]
        # Reference is the first POPULATED bin: the 0-2 A bin never holds a Ca pair
        # (closest Ca-Ca is ~3.8 A), so using bin 0 returns NaN for every arm.
        L, cens, r0, ref = float("nan"), False, float("nan"), float("nan")
        if ok:
            i0 = ok[0]
            r0, ref = float(centres[i0]), float(curve[i0])
            if ref > 0:
                half, prev = ref / 2.0, i0
                L, cens = float(centres[ok[-1]]), True
                for j in ok[1:]:
                    if curve[j] <= half:
                        x0, x1 = centres[prev], centres[j]
                        y0, y1 = curve[prev], curve[j]
                        t = 0.0 if y0 == y1 else (y0 - half) / (y0 - y1)
                        L, cens = float(x0 + t * (x1 - x0)), False
                        break
                    prev = j
        out["arms"][a] = {"correlogram": {
            "n_chains": n_ch, "edges": EDGES.tolist(),
            "series": {"feature": {
                "curve": [None if not np.isfinite(x) else float(x) for x in curve],
                "n_pairs": scnt[a].tolist(), "half_decay": L,
                "censored": bool(cens), "ref_bin_centre": r0, "ref_cos": ref}}}}
        print(f"{a:34s} L = {L:6.2f}   cos@{r0:.0f}A = {ref:.4f}"
              f"{'  (CENSORED)' if cens else ''}", flush=True)

    if n_ch == 0:
        raise SystemExit("FAILED: 0 chains processed.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}  ({n_ch} chains, {len(arms)} arms)")


if __name__ == "__main__":
    main()
