"""Shared per-chain box extraction for the o5 (CleanDIFT) probes.

Factored out so the K-draw gate and the arms evaluation cut IDENTICAL boxes from
IDENTICAL residues -- the arms-parity requirement is only meaningful if the
extraction is literally the same code.

Reads the `build_vol_cache` volumes when present (preprocessing is 1.2-2.9 s per
map and would otherwise dominate) and falls back to `load_map` + `preprocess`,
which is byte-identical by construction since the cache stores exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

AA1 = "ARNDCQEGHILKMFPSTWYV"


def load_norm_vol(pdb: str, vol_dir: Path | None):
    """(preprocessed volume, origin_zyx). Cache hit or honest recomputation."""
    from probes.o1_cryofm_benchmark import to_cubic_even
    from probes.stability import load_map
    from teachers.cryofm_tap import MODEL_VOXEL_SIZE, preprocess

    d = Path(pdb).parent
    if vol_dir is not None:
        npy, meta = Path(vol_dir) / f"{d.name}.npy", Path(vol_dir) / f"{d.name}.json"
        if npy.exists() and meta.exists():
            m = json.loads(meta.read_text())
            if abs(m["voxel_size"] - MODEL_VOXEL_SIZE) > 1e-9:
                raise ValueError(f"cached voxel_size {m['voxel_size']} != model grid")
            return np.load(npy, mmap_mode="r"), np.asarray(m["origin_zyx"])
    vol, vs, origin = load_map(str(d / f"{d.name}_raw_emd.map"))
    return preprocess(to_cubic_even(vol), vs), origin


def chain_boxes(row, vol_dir, per_chain: int, rng, device: str, box_chunk: int):
    """Ca-centred, backbone-frame boxes plus the per-residue labels.

    Filters exactly as `o4_lab_arms.py` does -- Ca at least PATCH//2 from every
    edge, known amino acid, known secondary structure -- so numbers stay
    comparable with the existing tables.
    """
    from probes.homolog_diagnostic_residue import chain_backbone
    from probes.local_frame_stability import extract_local_boxes
    from probes.o1_cryofm_benchmark import backbone_with_resnum, ss_labels
    from teachers.cryofm_tap import MODEL_VOXEL_SIZE, PATCH

    d = Path(row["pdb"]).parent
    norm, origin = load_norm_vol(row["pdb"], vol_dir)
    seq, ca, fr, nums = backbone_with_resnum(row["pdb"], row["chain"])
    if seq != chain_backbone(row["pdb"], row["chain"])[0] or seq != row["seq"]:
        raise ValueError("sequence mismatch")
    coords = (ca - np.asarray(origin)[None]) / MODEL_VOXEL_SIZE
    shape = np.array(norm.shape)
    ok = np.all((coords >= PATCH // 2) & (coords < shape[None] - PATCH // 2), axis=1)
    ssl = ss_labels(d)
    aa = np.array([AA1.index(c) if c in AA1 else -1 for c in seq])
    ss = np.array([ssl.get((row["chain"], int(n)), -1) for n in nums])
    ok &= (aa >= 0) & (ss >= 0)
    idx = np.nonzero(ok)[0]
    if len(idx) < 8:
        raise ValueError(f"only {len(idx)} usable residues")
    if len(idx) > per_chain:
        idx = rng.choice(idx, per_chain, replace=False)
        idx.sort()
    vt = torch.from_numpy(np.ascontiguousarray(norm))
    boxes = extract_local_boxes(vt, coords[idx], fr[idx], device=device,
                                chunk=box_chunk)
    del vt
    return boxes, aa[idx], ss[idx], idx
