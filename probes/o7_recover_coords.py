"""Recover the Ca coordinates + residue indices behind the o5 CleanDIFT latents.

`probes/o5_cleandift_arms.py` / `o5_arms_multitap.py` save per-residue features but
NOT the residue index or the Ca position, so none of the sibling project's spatial
analyses can be run on them as stored. Re-extracting on GPU would cost hours; the
selection is instead RECOVERABLE, because `chain_boxes` draws it from a single
`np.random.default_rng(seed)` advanced over the chains in `emd`-sorted order and
every failure path in `chain_boxes` raises BEFORE `rng.choice` is reached.

Recovery is therefore a replay, and it is verified rather than assumed: the
recovered index set must reproduce the `aa` AND `ss` label arrays actually stored
in each part file, exactly, for every chain. Anything less aborts -- a partial
match would silently mis-assign coordinates to features, which is precisely the
class of bug this project keeps catching late.

Output: `data/o7_coords.npz`, one `<key>/idx` and `<key>/ca` per chain, plus the
full-chain `<key>/ca_all` needed by the subsampling control.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

AA1 = "ARNDCQEGHILKMFPSTWYV"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--vol-dir", type=Path, default=Path("data/cleandift_vols"))
    ap.add_argument("--feat-dir", type=Path, default=Path("data/o5_multitap_parts"))
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--per-chain", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/o7_coords.npz"))
    args = ap.parse_args()

    from probes.homolog_diagnostic_residue import chain_backbone
    from probes.o1_cryofm_benchmark import backbone_with_resnum, ss_labels
    from teachers.cryofm_tap import MODEL_VOXEL_SIZE, PATCH

    rows = list(csv.DictReader(open(args.chains)))[: args.limit]
    rows.sort(key=lambda r: r["emd"])
    rng = np.random.default_rng(args.seed)

    bb_cache: dict[str, tuple] = {}
    ss_cache: dict[str, dict] = {}
    out: dict[str, np.ndarray] = {}
    keys, n_match, n_nopart, n_skip = [], 0, 0, 0
    bad: list[str] = []

    for r in rows:
        d = Path(r["pdb"]).parent
        part = args.feat_dir / f"{r['key']}.npz"
        meta = args.vol_dir / f"{d.name}.json"
        # Mirror chain_boxes' ordering exactly: every early-out below sits before
        # the rng.choice call, so it consumes no randomness.
        if not meta.exists():
            n_skip += 1
            continue
        m = json.loads(meta.read_text())
        if abs(m["voxel_size"] - MODEL_VOXEL_SIZE) > 1e-9:
            raise ValueError(f"{d.name}: cached voxel_size {m['voxel_size']}")
        try:
            if r["pdb"] + "|" + r["chain"] not in bb_cache:
                bb_cache[r["pdb"] + "|" + r["chain"]] = backbone_with_resnum(
                    r["pdb"], r["chain"])
            seq, ca, _fr, nums = bb_cache[r["pdb"] + "|" + r["chain"]]
            if seq != chain_backbone(r["pdb"], r["chain"])[0] or seq != r["seq"]:
                raise ValueError("sequence mismatch")
        except Exception:
            n_skip += 1
            continue
        coords = (ca - np.asarray(m["origin_zyx"])[None]) / MODEL_VOXEL_SIZE
        shape = np.array(m["shape"])
        ok = np.all((coords >= PATCH // 2) & (coords < shape[None] - PATCH // 2), axis=1)
        if str(d) not in ss_cache:
            ss_cache[str(d)] = ss_labels(d)
        ssl = ss_cache[str(d)]
        aa = np.array([AA1.index(c) if c in AA1 else -1 for c in seq])
        ss = np.array([ssl.get((r["chain"], int(n)), -1) for n in nums])
        ok &= (aa >= 0) & (ss >= 0)
        idx = np.nonzero(ok)[0]
        if len(idx) < 8:
            n_skip += 1
            continue
        if len(idx) > args.per_chain:                     # the only rng draw
            idx = rng.choice(idx, args.per_chain, replace=False)
            idx.sort()
        if not part.exists():
            n_nopart += 1
            continue
        z = np.load(part, allow_pickle=True)
        if not (len(z["aa"]) == len(idx) and np.array_equal(z["aa"], aa[idx])
                and np.array_equal(z["ss"], ss[idx])):
            bad.append(r["key"])
            continue
        n_match += 1
        keys.append(r["key"])
        out[f"{r['key']}/idx"] = idx.astype(np.int32)
        out[f"{r['key']}/ca"] = ca[idx].astype(np.float32)      # Angstrom
        out[f"{r['key']}/ca_all"] = ca.astype(np.float32)
        out[f"{r['key']}/idx_all"] = np.nonzero(ok)[0].astype(np.int32)

    n_parts = len(list(args.feat_dir.glob("*.npz")))
    print(f"parts on disk {n_parts} | verified {n_match} | no part {n_nopart} | "
          f"pre-rng skip {n_skip} | MISMATCH {len(bad)}", flush=True)
    # A partial replay is worse than none: it silently pairs the wrong coordinates
    # with the right features. Demand a clean sweep.
    if bad:
        raise SystemExit(f"FAILED: {len(bad)} chains failed aa/ss verification "
                         f"(first 10: {bad[:10]}). The rng replay is not valid.")
    if n_match != n_parts:
        raise SystemExit(f"FAILED: verified {n_match} of {n_parts} part files. "
                         f"Every stored chain must be recoverable.")
    out["keys"] = np.array(keys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    ns = np.array([len(out[f"{k}/idx"]) for k in keys])
    na = np.array([len(out[f"{k}/ca_all"]) for k in keys])
    print(f"wrote {args.out}  chains={len(keys)}  residues={ns.sum()} "
          f"(median {np.median(ns):.0f}/chain of {np.median(na):.0f} observed)")


if __name__ == "__main__":
    main()
