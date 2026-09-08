"""Per-residue local backbone frames: pose-invariance by construction.

Phase 0a found per-residue CryoFM2 features only 31-51% pose-invariant, capping
R^2 at ~0.5, and that no global fix helps -- canonical orientation does nothing
for tiling (0.15-0.29 at FIXED pose), and pose-averaging over 24 rotations washes
out features whose pairwise cosine is 0.3-0.5.

This takes a different route. Instead of canonicalising the MAP, canonicalise each
RESIDUE's own box: cut a 64^3 box centred on the Ca and rotated into the residue's
N-Ca-C backbone frame, then run CryoFM on that box and read the centre feature.

Global rotation of the protein leaves every local box unchanged, so the feature is
pose-invariant BY CONSTRUCTION rather than by augmentation or averaging. It also
kills tiling variance (the residue is always at box centre) and the homo-oligomer
ambiguity (symmetry-equivalent chains have identical local environments).

WHAT THIS TEST ACTUALLY MEASURES. Because invariance is exact by construction, the
residual is *interpolation error*: rotating to an arbitrary frame needs trilinear
sampling, unlike the lossless transpose/flip of cube rotations. So a number below
1.0 here is resampling noise, not conceptual leakage. It could still be large
enough to matter, which is why it needs measuring rather than assuming.

The payoff measurement is RETRIEVAL. Global-frame features gave top-1 of only
3-10% with the top hit 42-46 A away (vs 49.8 A random) -- i.e. not residue-specific
at all. If local framing makes features genuinely residue-specific, retrieval
should jump sharply. That, not the invariance number, decides whether the
per-residue track is revivable.

    pixi run python probes/local_frame_stability.py --limit 4 --n-res 300
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gemmi
import numpy as np
import torch
import torch.nn.functional as F

from probes.stability import centred_cosine, load_map, retrieval
from teachers.cryofm_tap import (StopForward,
    MODEL_VOXEL_SIZE,
    PATCH,
    TAP_STRIDES,
    CryoFM2Tap,
    cube_rotations,
    preprocess,
    rotate_coords,
    rotate_volume,
)


def backbone_frames(cif_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Cα positions [N,3] (z,y,x) and per-residue rotation matrices [N,3,3].

    Gram-Schmidt on (N -> Cα, C -> Cα), the AlphaFold convention. Residues
    missing any backbone atom are dropped.
    """
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    ca, frames = [], []
    for model in st:
        for chain in model:
            for res in chain:
                aN, aCA, aC = (res.find_atom(n, "*") for n in ("N", "CA", "C"))
                if aN is None or aCA is None or aC is None:
                    continue
                p = lambda a: np.array([a.pos.z, a.pos.y, a.pos.x], dtype=np.float64)
                c = p(aCA)
                v1, v2 = p(aN) - c, p(aC) - c
                e1 = v1 / max(np.linalg.norm(v1), 1e-8)
                v2 = v2 - (v2 @ e1) * e1
                e2 = v2 / max(np.linalg.norm(v2), 1e-8)
                e3 = np.cross(e1, e2)
                ca.append(c)
                frames.append(np.stack([e1, e2, e3]))
        break
    return np.array(ca), np.array(frames)


@torch.no_grad()
def extract_local_boxes(vol: torch.Tensor, centres: np.ndarray, frames: np.ndarray,
                        box: int = PATCH, device: str = "cuda",
                        chunk: int = 64) -> torch.Tensor:
    """Cut [M, 1, box, box, box] frame-aligned boxes by trilinear resampling.

    centres are voxel indices into `vol`; frames are [M,3,3] rotations whose rows
    are the residue's local axes.
    """
    D, H, W = vol.shape
    shape = torch.tensor([D, H, W], dtype=torch.float64, device=device)
    lin = torch.arange(box, dtype=torch.float64, device=device) - (box - 1) / 2.0
    gz, gy, gx = torch.meshgrid(lin, lin, lin, indexing="ij")
    local = torch.stack([gz, gy, gx], -1).reshape(-1, 3)          # [box^3, 3]

    vol5 = vol.to(device)[None, None]
    out = []
    for s in range(0, len(centres), chunk):
        R = torch.from_numpy(frames[s:s + chunk]).to(device)      # [m,3,3]
        c = torch.from_numpy(centres[s:s + chunk]).to(device)     # [m,3]
        # world = R^T @ local + centre  (rows of R are the local axes)
        world = torch.einsum("mij,kj->mki", R.transpose(1, 2), local) + c[:, None, :]
        norm = 2.0 * world / (shape - 1) - 1.0                    # [m, box^3, 3]
        grid = norm.flip(-1).float().view(len(c), box, box, box, 3)
        out.append(F.grid_sample(vol5.expand(len(c), -1, -1, -1, -1).float(),
                                 grid, mode="bilinear", align_corners=True,
                                 padding_mode="zeros"))
    return torch.cat(out)                                          # [M,1,box,box,box]


@torch.no_grad()
def centre_features(tap: CryoFM2Tap, boxes: torch.Tensor, timestep: int,
                    batch: int = 16, noise_level: float | None = None,
                    noise_seed: int = 0) -> dict[str, np.ndarray]:
    """Forward each local box; read the feature at the box centre (the residue).

    `noise_level` selects the (x_t, t) PAIRING, which is a real experimental
    variable and not a detail:

      None / 0.0  -- DECOUPLED: feed the clean box while telling the model `t`.
                     Off the training manifold, since CryoFM2 couples them by
                     construction (`FMScheduler.add_noise`:
                     x_t = (1 - t/1000) x_0 + (t/1000) eps). Still a legitimate
                     deterministic extractor, and it is exactly what a
                     CleanDIFT-style student does at inference (clean input,
                     one fixed timestep) -- so it is the right arm for "how good
                     is our clean extractor at conditioning signal t".
      t/1000      -- COUPLED: the trained pairing. This is the arm that answers
                     "how good are the TEACHER's features at true noise level t",
                     i.e. what a distillation would learn from.

    Sweeping t in the decoupled arm alone cannot distinguish a genuinely better
    representation from the network being pushed into its coarse-structure
    regime, which yields smoother -- hence trivially more pose-stable -- features.
    Run both arms and compare at matched t.
    """
    gen = torch.Generator(device="cpu").manual_seed(noise_seed)
    feats: dict[str, list] = {}
    for s in range(0, len(boxes), batch):
        x = boxes[s:s + batch].to(tap.device)
        if noise_level:
            eps = torch.randn(x.shape, generator=gen).to(x.device)
            x = (1.0 - noise_level) * x + noise_level * eps
        x = torch.cat([x, torch.zeros_like(x)], dim=1)             # in_channels=2
        t = torch.full((x.shape[0],), timestep, device=tap.device, dtype=torch.long)
        tap._buf.clear()
        try:
            tap.model(x, timestep=t)
        except StopForward:      # see the note in o4_frameavg_benchmark.centre2
            pass
        for k in tap.taps:
            f = tap._buf[k]                                        # [b,C,d,d,d]
            d = f.shape[-1]
            feats.setdefault(k, []).append(f[:, :, d // 2, d // 2, d // 2].cpu().numpy())
    return {k: np.concatenate(v) for k, v in feats.items()}


def random_so3(rng: np.random.Generator) -> np.ndarray:
    """Uniform SO(3) via QR of a Gaussian matrix (det forced to +1)."""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


@torch.no_grad()
def rotate_volume_so3(vol: np.ndarray, R: np.ndarray, device: str) -> np.ndarray:
    """Resample a volume under an ARBITRARY rotation (trilinear, lossy).

    Unlike the cube-rotation path this genuinely resamples, which is the whole
    point: it introduces the interpolation error that the lossless cube test
    could not expose.
    """
    D, H, W = vol.shape
    shape = torch.tensor([D, H, W], dtype=torch.float64, device=device)
    centre = (shape - 1) / 2.0
    idx = [torch.arange(s, dtype=torch.float64, device=device) for s in (D, H, W)]
    gz, gy, gx = torch.meshgrid(*idx, indexing="ij")
    dst = torch.stack([gz, gy, gx], -1).reshape(-1, 3) - centre
    Rt = torch.from_numpy(R).to(device)
    src = dst @ Rt + centre                      # inverse map: R^T applied as right-mult
    norm = (2.0 * src / (shape - 1) - 1.0).flip(-1).float().view(1, D, H, W, 3)
    out = F.grid_sample(torch.from_numpy(vol).to(device)[None, None].float(),
                        norm, mode="bilinear", align_corners=True, padding_mode="zeros")
    return out[0, 0].cpu().numpy()


def diversity(feats: np.ndarray, max_n: int = 1500) -> dict:
    """Are local-frame features actually different between residues?

    Perfect invariance is worthless if every residue maps to the same vector.
    Reports the median off-diagonal cosine (want LOW) and the effective rank of
    the feature matrix (want HIGH).
    """
    X = feats
    if len(X) > max_n:
        X = X[np.random.default_rng(0).choice(len(X), max_n, replace=False)]

    # RAW (uncentred) cosine. Centring would defeat the purpose: it removes the
    # shared constant, and row-normalising then rescales the leftover noise to
    # unit length, so a collapsed feature set looks exactly like random noise.
    # Verified empirically -- the centred version scored an all-identical input
    # the same as a random one.
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    S = Xn @ Xn.T
    off = S[~np.eye(len(S), dtype=bool)]

    # Scale of between-residue variation relative to the shared mean.
    # ~0 => collapsed; O(1) => residues genuinely differ.
    mu = X.mean(0)
    rel_var = float(np.mean(np.linalg.norm(X - mu, axis=1)) / max(np.linalg.norm(mu), 1e-8))

    Xc = X - mu
    sv = np.linalg.svd(Xc, compute_uv=False)   # svdvals() is numpy>=2.0; we pin <2.0
    p = sv**2 / max(np.sum(sv**2), 1e-12)
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    return {"median_offdiag_cos": float(np.median(off)),   # want LOW (raw)
            "rel_variation": rel_var,                      # want O(1), ~0 = collapsed
            "effective_rank": eff_rank,
            "dim": int(X.shape[1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--n-res", type=int, default=300, help="residues sampled per map")
    ap.add_argument("--n-rot", type=int, default=3)
    ap.add_argument("--timestep", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("results/local_frame.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))[: args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tap = CryoFM2Tap(args.ckpt, device=dev, batch_size=args.batch_size)
    rng = np.random.default_rng(0)
    rots = [random_so3(rng) for _ in range(args.n_rot)]
    print(f"device={dev} maps={len(rows)} residues/map={args.n_res} rotations={args.n_rot}")

    pooled: dict[str, dict[str, list]] = {}
    retr: dict[str, list] = {}
    div: dict = {}
    divstats: dict[str, list] = {}

    for i, row in enumerate(rows):
        try:
            vol, vs, origin_A = load_map(row["map_path"])
            ca, frames = backbone_frames(row["cif_path"])
            if len(ca) < 50:
                continue
            norm = preprocess(vol, vs)
            coords = (ca - origin_A[None]) / MODEL_VOXEL_SIZE
            keep = np.all((coords >= PATCH // 2) &
                          (coords < np.array(norm.shape)[None] - PATCH // 2), axis=1)
            coords, frames = coords[keep], frames[keep]
            if len(coords) < 50:
                print(f"  {row['emdb_id']}: only {len(coords)} residues clear of the border")
                continue
            sel = rng.choice(len(coords), min(args.n_res, len(coords)), replace=False)
            coords, frames = coords[sel], frames[sel]

            vt = torch.from_numpy(norm)
            ref = centre_features(tap, extract_local_boxes(vt, coords, frames, device=dev),
                                  args.timestep, args.batch_size)

            div.setdefault(list(ref)[0], None)
            for k, v in ref.items():
                divstats.setdefault(k, []).append(diversity(v))

            centre_vox = (np.array(norm.shape) - 1) / 2.0
            for R in rots:
                rvol = torch.from_numpy(rotate_volume_so3(norm, R, dev))
                # forward map for points: p' = R (p - c) + c
                rc = (coords - centre_vox) @ R.T + centre_vox
                # local axes rotate with the structure: e' = R e
                rf = np.einsum("ij,mkj->mki", R.astype(np.float64), frames)
                rot = centre_features(tap, extract_local_boxes(rvol, rc, rf, device=dev),
                                      args.timestep, args.batch_size)
                for k in ref:
                    pooled.setdefault(k, {}).setdefault("pose", []).append(
                        centred_cosine(ref[k], rot[k]))
                    retr.setdefault(k, []).append(
                        retrieval(ref[k], rot[k], coords_A=coords))
            print(f"[{i+1}/{len(rows)}] {row['emdb_id']} {len(coords)} residues")
        except Exception as exc:
            print(f"[{i+1}/{len(rows)}] {row['emdb_id']} FAILED {type(exc).__name__}: {exc}")

    summary = {}
    for k, d in pooled.items():
        v = np.concatenate(d["pose"])
        v = v[np.isfinite(v)]
        r = retr[k]
        summary[k] = {
            "pose_median": float(np.median(v)),
            "pose_p10": float(np.percentile(v, 10)),
            "top1": float(np.mean([x["top1"] for x in r])),
            "median_pct_rank": float(np.mean([x["median_pct_rank"] for x in r])),
            "top1_dist_A": float(np.mean([x["top1_dist_A"] for x in r])),
            "random_dist_A": float(np.mean([x["random_dist_A"] for x in r])),
            "median_offdiag_cos": float(np.mean([x["median_offdiag_cos"] for x in divstats[k]])),
            "rel_variation": float(np.mean([x["rel_variation"] for x in divstats[k]])),
            "effective_rank": float(np.mean([x["effective_rank"] for x in divstats[k]])),
            "dim": divstats[k][0]["dim"],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)

    print("\n" + "=" * 76)
    print(f"{'tap':<16}{'pose(SO3)':>11}{'p10':>9}{'top1':>8}{'pctrank':>10}"
          f"{'offdiagcos':>12}{'relvar':>9}{'effrank':>9}")
    print("-" * 76)
    for k, s in summary.items():
        print(f"{k:<16}{s['pose_median']:>11.3f}{s['pose_p10']:>9.3f}{s['top1']:>8.3f}"
              f"{s['median_pct_rank']:>10.4f}{s['median_offdiag_cos']:>12.3f}"
              f"{s['effective_rank']:>9.1f}{s['dim']:>6}")
    print("=" * 76)
    print("GLOBAL-FRAME baseline (Phase 0a): pose 0.33-0.50, top1 0.03-0.10")
    print("pose(SO3) uses ARBITRARY rotations -> real trilinear resampling of the volume,")
    print("so unlike the lossless cube-rotation test this CAN fail. It is the honest number.")
    print("offdiagcos/effrank guard the degenerate case: perfect invariance is worthless")
    print("if every residue maps to the same vector (want LOW offdiag, HIGH effrank).")


if __name__ == "__main__":
    main()
