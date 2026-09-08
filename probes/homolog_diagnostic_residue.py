"""Per-residue homolog diagnostic, on LOCAL-FRAME features.

The pooled diagnostic (`homolog_diagnostic.py`) killed the pooled CryoFM target:
R2 ceiling ~0 once size and composition are partialled out. It says nothing
about the PER-RESIDUE target, which is a different object with a much higher
pose ceiling (0.68-0.83 with local backbone frames, vs 0.21-0.38 pose *noise*
for the pooled descriptor under generic SO(3)). This runs the same three-level
variance decomposition there.

Two things differ from the pooled version.

**Correspondence.** Comparing homologs per residue needs to know which residue
maps to which. Sequence alignment -- not structural superposition -- is the
right tool: the question is whether a feature is predictable *from sequence*, so
the sequence alignment is the correspondence a sequence model would have.
gemmi's aligner over the OBSERVED residues (those with a full N/CA/C backbone),
not the FASTA, since unmodelled residues have no density to describe.

**The trivial control is amino-acid identity, not size.** Corresponding homolog
residues are ~43% identical by construction while random cross-protein pairs are
~5%, so "homologs are closer" is guaranteed by amino-acid identity alone. That
is the per-residue analogue of the `[N, Rg, composition]` confound, and it must
be partialled out for the same reason.

Levels:
    pose       same residue, two independent SO(3) re-simulations
    homolog    aligned residue pairs across 30-60% identity homologs
    unrelated  arbitrary residue pairs across length-matched unrelated chains

    r2_ceiling = (D2_unrel - D2_hom)/D2_unrel

**What this number is, precisely.** It is the fraction of descriptor variance
that TRACKS SEQUENCE SIMILARITY at homolog resolution -- a smoothness measure.
It is NOT a strict upper bound on a trained model, and calling it an "R2 ceiling"
(as an earlier version of this file did) overstates it in one direction and
understates it in another:
  - it can UNDERSTATE a real model, because two 63%-identical sequences are
    still different sequences, so a model given the exact sequence should
    capture some of the within-family variance this charges as unpredictable;
  - it can OVERSTATE generalisation to unseen folds, since homolog pairs share
    a fold by construction.
The genuine hard ceiling is the pose term alone: 1 - pose_fraction (0.84 here),
since orientation noise is irreducible for any pose-free model. The load-bearing
comparison is RELATIVE -- ~0.45 per-residue vs ~0.00 pooled -- and that is
unaffected by the labelling.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np
import torch

from data.simulate_density import model_centre, simulate
from probes.local_frame_stability import (
    centre_features,
    extract_local_boxes,
    random_so3,
)
from probes.pose_invariance_clean import _R_zyx
from teachers.cryofm_tap import MODEL_VOXEL_SIZE, PATCH, CryoFM2Tap, preprocess

AA3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
AA1 = "ARNDCQEGHILKMFPSTWYV"


def chain_backbone(pdb_path: str, chain_id: str):
    """Observed residues of one chain: (one-letter seq, Ca [N,3] zyx, frames [N,3,3]).

    Only residues with a complete N/CA/C backbone -- the frame is undefined
    otherwise, and those residues are exactly the ones with poor density anyway.
    """
    st = gemmi.read_structure(pdb_path)
    st.setup_entities()
    st.remove_alternative_conformations()
    seq, ca, frames = [], [], []
    for model in st:
        for ch in model:
            if ch.name != chain_id:
                continue
            for res in ch:
                aN, aCA, aC = (res.find_atom(n, "*") for n in ("N", "CA", "C"))
                if aN is None or aCA is None or aC is None:
                    continue
                if res.name not in AA3to1:
                    continue
                p = lambda a: np.array([a.pos.z, a.pos.y, a.pos.x], dtype=np.float64)
                c = p(aCA)
                v1, v2 = p(aN) - c, p(aC) - c
                e1 = v1 / max(np.linalg.norm(v1), 1e-8)
                v2 = v2 - (v2 @ e1) * e1
                e2 = v2 / max(np.linalg.norm(v2), 1e-8)
                ca.append(c)
                frames.append(np.stack([e1, e2, np.cross(e1, e2)]))
                seq.append(AA3to1[res.name])
        break
    return "".join(seq), np.array(ca), np.array(frames)


def correspondence(sa: str, sb: str) -> tuple[np.ndarray, np.ndarray]:
    """Aligned index pairs between two observed sequences.

    gemmi CIGAR convention, verified empirically: M consumes both, I consumes
    the query only, D the target only.
    """
    r = gemmi.align_string_sequences(list(sa), list(sb), [],
                                     gemmi.AlignmentScoring("b"))
    ia, ib, i, j = [], [], 0, 0
    num = ""
    for chpos in r.cigar_str():
        if chpos.isdigit():
            num += chpos
            continue
        n = int(num or 1)
        num = ""
        if chpos == "M":
            ia.extend(range(i, i + n))
            ib.extend(range(j, j + n))
            i += n
            j += n
        elif chpos == "I":
            i += n
        else:                                   # 'D'
            j += n
    return np.array(ia, dtype=int), np.array(ib, dtype=int)


def seq_window(seq: str, idx: np.ndarray, w: int) -> np.ndarray:
    """[len(idx), 2w+1] amino-acid indices around each position; -1 off the end."""
    out = np.full((len(idx), 2 * w + 1), -1, dtype=int)
    for r, i in enumerate(idx):
        for c, j in enumerate(range(i - w, i + w + 1)):
            if 0 <= j < len(seq):
                out[r, c] = AA1.index(seq[j])
    return out


def residue_features(tap, pdb, chain, ca, frames, sel, R, args, dev):
    """Per-residue descriptors for residues `sel` at orientation R.

    Returns ``({name: [M, C]}, keep_mask)``. Two rungs, differing in what they
    need at INFERENCE time -- which is the whole point of running them together:

      local     box cut in the residue's N-CA-C frame. Pose-invariant by
                construction and the strongest measured (0.68-0.83), but the
                frame comes from an atomic model, so it can never be computed
                for an unknown map. Fine for defining a training TARGET, fatal
                for map-side inference.
      frameavg  no local frame at all: whole-map features averaged over the 24
                cube rotations. Exactly invariant to that group by construction,
                lossless, and MODEL-FREE. If this retains the local rung's
                homolog signal, the approach becomes deployable.
    """
    vol, origin, spacing = simulate(
        pdb, d_min=args.d_min, voxel=MODEL_VOXEL_SIZE, rotation=R,
        chain=chain, margin=args.margin)
    c_ca, fr = ca[sel], frames[sel]
    if R is not None:
        ctr = model_centre(pdb, chain)[::-1]              # xyz -> zyx
        Rz = _R_zyx(R)
        c_ca = (c_ca - ctr) @ Rz.T + ctr
        fr = np.einsum("ij,mkj->mki", Rz, fr)
    coords = (c_ca - np.asarray(origin)[::-1][None]) / np.asarray(spacing)[None]
    shape = np.array(vol.shape)
    keep = np.all((coords >= PATCH // 2) & (coords < shape[None] - PATCH // 2), axis=1)
    if keep.sum() < args.min_res:
        raise ValueError(f"only {int(keep.sum())} residues have box clearance")
    norm = preprocess(vol, MODEL_VOXEL_SIZE)
    out = {}

    if "local" in args.rungs:
        boxes = extract_local_boxes(torch.from_numpy(norm), coords[keep], fr[keep],
                                    device=dev)
        # centre_features returns {tap_name: [M, C]}, not an array.
        nl = (args.timestep / 1000.0) if args.couple else None
        out["local"] = centre_features(
            tap, boxes, args.timestep, args.batch_size, noise_level=nl)[args.tap]

    if "frameavg" in args.rungs:
        out["frameavg"] = tap.sample_frame_averaged(
            norm, coords[keep], n_frames=args.n_frames,
            timestep=args.timestep,
            noise_level=(args.timestep / 1000.0) if args.couple else 0.0)[args.tap]

    return out, keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/homolog_pairs.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--limit", type=int, default=60, help="pairs per kind")
    ap.add_argument("--n-res", type=int, default=40, help="residues per chain")
    ap.add_argument("--min-res", type=int, default=10)
    ap.add_argument("--d-min", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=PATCH // 2 * MODEL_VOXEL_SIZE)
    ap.add_argument("--timestep", type=int, default=10)
    ap.add_argument("--couple", action="store_true",
                    help="tie input noise to the timestep (noise_level = t/1000), the\n                          TRAINED pairing. Default off = clean input, which is what every\n                          ceiling number logged before 2026-08-27 used.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-window", type=int, default=3,
                    help="+-w residues of local sequence to partial out; the "
                         "stronger trivial control, since a residue's local "
                         "density is set by its neighbours as well as itself")
    ap.add_argument("--tap", default="up_blocks[0]",
                    help="best per-residue tap (0.826 vs 0.677 for up_blocks[1])")
    ap.add_argument("--rungs", nargs="*", default=["local", "frameavg"],
                    help="which per-residue descriptors to compute; they differ "
                         "in what they need at INFERENCE (see residue_features)")
    ap.add_argument("--n-frames", type=int, default=24,
                    help="octahedral frames to average for the frameavg rung; "
                         "must be 24 for exact group invariance")
    ap.add_argument("--out", type=Path,
                    default=Path("results/homolog_residue.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.pairs)))
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    pairs = [r for k in ("homolog", "unrelated") for r in by_kind[k][: args.limit]]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tap = CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                     batch_size=args.batch_size)
    rng = np.random.default_rng(0)
    print(f"{len(pairs)} pairs, device={dev}, tap={args.tap}, rungs={args.rungs}")

    A0, A1, B0, AAa, AAb, kinds = [], [], [], [], [], []
    for i, r in enumerate(pairs):
        try:
            sa, caa, fra = chain_backbone(r["pdb_a"], r["chain_a"])
            sb, cab, frb = chain_backbone(r["pdb_b"], r["chain_b"])
            if min(len(sa), len(sb)) < args.min_res:
                continue
            if r["kind"] == "homolog":
                ia, ib = correspondence(sa, sb)
                # Take whatever aligns. Observed sequences are gappy (unmodelled
                # residues are simply absent), so a global alignment often keeps
                # only a shorter high-identity core -- 17-25 residues on some
                # pairs. Requiring the full n_res would silently drop exactly the
                # more divergent homologs and bias the sample toward easy ones.
                if len(ia) < args.min_res:
                    continue
            else:
                # No correspondence exists between unrelated chains, and none is
                # needed: the null is "two arbitrary residues from different
                # proteins". Pair them in order after an independent shuffle.
                n = min(len(sa), len(sb))
                ia = rng.permutation(len(sa))[:n]
                ib = rng.permutation(len(sb))[:n]
            pick = rng.choice(len(ia), min(args.n_res, len(ia)), replace=False)
            ia, ib = ia[pick], ib[pick]

            fa0, ka = residue_features(tap, r["pdb_a"], r["chain_a"], caa, fra,
                                       ia, None, args, dev)
            fa1, ka1 = residue_features(tap, r["pdb_a"], r["chain_a"], caa, fra,
                                        ia, random_so3(rng), args, dev)
            fb0, kb = residue_features(tap, r["pdb_b"], r["chain_b"], cab, frb,
                                       ib, None, args, dev)
            both = ka & ka1 & kb
            if both.sum() < args.min_res:
                continue
            # Re-index each feature block onto the surviving residues.
            sa_i = np.cumsum(ka) - 1
            sa1_i = np.cumsum(ka1) - 1
            sb_i = np.cumsum(kb) - 1
            A0.append({k: v[sa_i[both]] for k, v in fa0.items()})
            A1.append({k: v[sa1_i[both]] for k, v in fa1.items()})
            B0.append({k: v[sb_i[both]] for k, v in fb0.items()})
            AAa.append(seq_window(sa, ia[both], args.seq_window))
            AAb.append(seq_window(sb, ib[both], args.seq_window))
            kinds.append(r["kind"])
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(pairs)}] {len(A0)} pairs ok")
        except Exception as exc:
            print(f"  [{i+1}/{len(pairs)}] SKIP {Path(r['pdb_a']).stem}:{r['chain_a']} "
                  f"{type(exc).__name__}: {exc}")

    if not A0:
        raise SystemExit("no pair produced features -- check the skip messages above")

    kinds = np.array(kinds)
    # `rungs` first: each A0 entry is now a DICT of descriptors, so len() on it
    # counts rungs, not residues. Build the pair index from a rung's array.
    rungs = [k for k in A0[0] if all(k in d for d in A0 + A1 + B0)]
    idxa = np.concatenate([np.full(len(d[rungs[0]]), i) for i, d in enumerate(A0)])
    XA = {k: np.concatenate([d[k] for d in A0]) for k in rungs}
    XB = {k: np.concatenate([d[k] for d in B0]) for k in rungs}
    XA1 = {k: np.concatenate([d[k] for d in A1]) for k in rungs}
    Wa_idx, Wb_idx = np.concatenate(AAa), np.concatenate(AAb)   # [N, 2w+1]
    w = args.seq_window
    aa_a, aa_b = Wa_idx[:, w], Wb_idx[:, w]                      # central residue
    ident = float((aa_a[np.isin(idxa, np.where(kinds == "homolog")[0])]
                   == aa_b[np.isin(idxa, np.where(kinds == "homolog")[0])]).mean())
    ident_u = float((aa_a[np.isin(idxa, np.where(kinds == "unrelated")[0])]
                     == aa_b[np.isin(idxa, np.where(kinds == "unrelated")[0])]).mean())
    print(f"\naligned-residue identity: homolog {ident:.1%}  unrelated {ident_u:.1%}")

    def onehot(v):
        """[N] AA indices -> [N, 20]; index -1 (out of chain) stays all-zero."""
        M = np.zeros((len(v), 20))
        ok = v >= 0
        M[np.arange(len(v))[ok], v[ok]] = 1.0
        return M

    def window_onehot(_unused, __unused, w, W=None):
        """[N, 2w+1] AA indices -> [N, 20*(2w+1)] flattened one-hot."""
        return np.concatenate([onehot(W[:, k]) for k in range(W.shape[1])], 1)

    def decompose(Xa, Xb, Xa1):
        sd = np.where(Xa.std(0) < 1e-9, 1.0, Xa.std(0))
        mu = Xa.mean(0)
        Za, Zb, Za1 = (Xa - mu) / sd, (Xb - mu) / sd, (Xa1 - mu) / sd
        hp = np.isin(idxa, np.where(kinds == "homolog")[0])
        up = np.isin(idxa, np.where(kinds == "unrelated")[0])
        d2p = ((Za - Za1) ** 2).sum(1)
        d2c = ((Za - Zb) ** 2).sum(1)
        if not (hp.any() and up.any()):
            return None
        m = {"pose": float(np.median(d2p)), "homolog": float(np.median(d2c[hp])),
             "unrelated": float(np.median(d2c[up]))}
        denom = m["unrelated"] - m["pose"]
        return {
            "n_residues": {"homolog": int(hp.sum()), "unrelated": int(up.sum())},
            **{f"D2_{k}": v for k, v in m.items()},
            "r2_ceiling": float((m["unrelated"] - m["homolog"]) / m["unrelated"]),
            "homolog_signal": float((m["unrelated"] - m["homolog"]) / denom)
            if denom > 0 else None,
            "pose_fraction": float(m["pose"] / m["unrelated"]),
        }

    Ta, Tb = onehot(aa_a), onehot(aa_b)
    Wa_full = window_onehot(None, None, w, Wa_idx)
    Wb_full = window_onehot(None, None, w, Wb_idx)

    results = {}
    # Trivial control: amino-acid identity alone. NB its r2_ceiling is 1.000 by a
    # MEDIAN degeneracy, not by being a perfect descriptor -- aligned homolog
    # residues are >50% identical so the MEDIAN one-hot distance is exactly 0.
    # Reported for completeness; the meaningful controls are the partialled rows.
    results["trivial_aa (median-degenerate)"] = decompose(Ta, Tb, Ta)

    for rung in rungs:
        Xa, Xb, Xa1 = XA[rung], XB[rung], XA1[rung]
        results[f"{rung}:{args.tap}"] = decompose(Xa, Xb, Xa1)
        # Remove the central amino acid, then a whole local sequence WINDOW. The
        # window is the stronger test: a residue's local density is largely set
        # by its neighbours too, so partialling only the central identity leaves
        # an obvious route by which the "signal" could still be plain sequence.
        for label, (Wa, Wb) in (("-aa", (Ta, Tb)),
                                (f"-seqwin{w}", (Wa_full, Wb_full))):
            Xw = np.concatenate([Wa, np.ones((len(Wa), 1))], 1)
            beta, *_ = np.linalg.lstsq(Xw, Xa, rcond=None)
            Xwb = np.concatenate([Wb, np.ones((len(Wb), 1))], 1)
            results[f"{rung} [{label}]"] = decompose(
                Xa - Xw @ beta, Xb - Xwb @ beta, Xa1 - Xw @ beta)

    results["_meta"] = {"n_pairs": len(A0), "aligned_identity_homolog": ident,
                        "aligned_identity_unrelated": ident_u, "tap": args.tap,
                        "n_res_per_chain": args.n_res, "rungs": rungs,
                        "n_frames": args.n_frames}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print(f"{'descriptor':<34}{'R2ceil':>9}{'signal':>9}{'posefrac':>10}")
    print("-" * 78)
    for k, v in results.items():
        if k.startswith("_") or v is None:
            continue
        sig = "n/a" if v["homolog_signal"] is None else f"{v['homolog_signal']:.3f}"
        print(f"{k:<34}{v['r2_ceiling']:>9.3f}{sig:>9}{v['pose_fraction']:>10.3f}")
    print("=" * 78)
    print("local    = needs an atomic model (training target only)")
    print("frameavg = MODEL-FREE, 24-frame octahedral averaging. If this keeps")
    print("           local's signal, the approach is deployable on unknown maps.")
    print("Pooled reference: CryoFM 0.145 -> -0.022 after partialling out")
    print("[n_res, Rg, composition] -- pooling destroyed the predictable part.")


if __name__ == "__main__":
    main()
