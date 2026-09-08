"""Batch assembly for DINO.txt training: cached voxels + ESM-C residue vectors.

CLUSTER-DISJOINT BATCHES. In-batch residues from other maps are the negatives, so
putting two homologous maps in one batch asserts that genuinely-similar residues
are negatives. Aligned homologs here share 63% identity, and false negatives in
this setting are structural rather than incidental (every alpha-helix resembles
every other). Batches are therefore built so no two maps share a 30%-identity
cluster.

THE SPLIT IS THE ONE THE STUDENT WAS TRAINED AGAINST. `map_chains.csv` carries
the map-level split resolved from `alignment_chains.csv` with test>val>train
precedence, matching `split_map_lists`' `clean_train`. Do not re-derive a split:
the encoder was trained on these train maps, so a new partition would show it
density containing val/test residues.
"""

from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np


class VoxelCorpus:
    """Lazy per-map access to (voxel features, targets, residue ESM-C vectors)."""

    def __init__(self, cache_dir: Path, esmc_dir: Path,
                 chains_csv: Path = Path("data/alignment_chains.csv"),
                 max_cached: int = 96, strict: bool = False,
                 coords_dir: Path | None = None):
        self.cache_dir = Path(cache_dir)
        self.esmc_dir = Path(esmc_dir)
        # Voxel coordinates live in a SEPARATE directory because they are
        # recovered post hoc on CPU (probes/recover_voxel_coords.py) rather than
        # written by the GPU cache build. They are shared across every low-pass
        # cache: `sample_voxels(pool="model")` never reads the volume's values,
        # only its shape, so the geometry is identical at every resolution.
        self.coords_dir = Path(coords_dir) if coords_dir else None
        self.maps: list[str] = sorted(
            (p.stem for p in self.cache_dir.glob("*.npz")), key=int)
        if not self.maps:
            raise SystemExit(f"no voxel cache in {cache_dir}")

        # cluster sets per map, for cluster-disjoint batching
        self.clusters: dict[str, set] = {}
        for r in csv.DictReader(open(chains_csv)):
            self.clusters.setdefault(r["emd"], set()).add(r["cluster"])

        self.split: dict[str, str] = {}
        for m in self.maps:
            self.split[m] = str(np.load(self.cache_dir / f"{m}.npz",
                                        allow_pickle=True)["split"])
        self._get = lru_cache(maxsize=max_cached)(self._load)

        # Drop maps whose sequences are not all embedded, UP FRONT. Discovering
        # this mid-training kills a run at an arbitrary step (measured: 8 maps,
        # 0.7%, all carrying a chain longer than the extraction's --max-len).
        # `strict=True` raises instead, for when a silent drop would be worse.
        bad = []
        for m in list(self.maps):
            try:
                self._check(m)
            except (KeyError, ValueError) as exc:
                bad.append((m, str(exc).split(" -- ")[0]))
        if bad:
            msg = (f"dropping {len(bad)}/{len(self.maps)} maps with unusable "
                   f"sequences, e.g. {bad[0][1]}")
            if strict:
                raise SystemExit(msg)
            print(f"  VoxelCorpus: {msg}", flush=True)
            drop = {m for m, _ in bad}
            self.maps = [m for m in self.maps if m not in drop]
        self.dropped = [m for m, _ in bad]

    def _check(self, emd: str) -> None:
        """Cheap validation: sequence files exist and have the right length.

        Deliberately NOT `_load`: that builds the full [n_res, 1152] residue
        matrix and would populate the LRU cache for every map in the corpus at
        construction time. `mmap_mode` reads the header only, so this costs a
        stat plus a few bytes per sequence.
        """
        d = np.load(self.cache_dir / f"{emd}.npz", allow_pickle=True)
        for s in d["seqs"]:
            s = str(s)
            f = self.esmc_dir / f"{hashlib.sha1(s.encode()).hexdigest()[:12]}.npy"
            if not f.exists():
                raise KeyError(f"{emd}: no ESM-C for a {len(s)}-residue sequence")
            n = np.load(f, mmap_mode="r").shape[0]
            if n != len(s):
                raise ValueError(
                    f"{emd}: ESM-C has {n} rows for a {len(s)}-residue sequence "
                    "-- the two modalities would be misaligned")

    def by_split(self, name: str) -> list[str]:
        return [m for m in self.maps if self.split[m] == name]

    def _load(self, emd: str) -> dict:
        d = dict(np.load(self.cache_dir / f"{emd}.npz", allow_pickle=True))
        seqs = [str(s) for s in d["seqs"]]
        # Residue-level ESM-C: row r of the map's residue table is position
        # res_pos[r] of sequence res_seq[r]. Assembled here so index alignment
        # between the two modalities is done in exactly one place.
        per_seq = []
        for s in seqs:
            k = hashlib.sha1(s.encode()).hexdigest()[:12]
            f = self.esmc_dir / f"{k}.npy"
            per_seq.append(np.load(f) if f.exists() else None)
        if any(p is None for p in per_seq):
            raise KeyError(f"{emd}: missing ESM-C for {sum(p is None for p in per_seq)}"
                           f"/{len(seqs)} sequences")
        for s, p in zip(seqs, per_seq):
            if len(p) != len(s):
                raise ValueError(
                    f"{emd}: ESM-C has {len(p)} rows for a {len(s)}-residue "
                    "sequence -- the two modalities would be misaligned")
        if self.coords_dir is not None:
            f = self.coords_dir / f"{emd}.npz"
            if not f.exists():
                raise KeyError(f"{emd}: no recovered coordinates in "
                               f"{self.coords_dir}")
            c = np.load(f)
            if len(c["xyz"]) != len(d["feat"]):
                raise ValueError(
                    f"{emd}: {len(c['xyz'])} coordinates for {len(d['feat'])} "
                    "voxels -- coordinates and features are misaligned")
            assert np.array_equal(c["rot"], d["rot"]), \
                f"{emd}: coordinate rotation labels disagree with the cache"
            d["xyz"] = c["xyz"]
        res_seq, res_pos = d["res_seq"], d["res_pos"]
        g = np.concatenate([per_seq[si][pi][None] for si, pi
                            in zip(res_seq, res_pos)]).astype(np.float32)
        d["res_emb"] = g
        d["seq_of_res"] = res_seq
        return d

    def get(self, emd: str) -> dict:
        return self._get(emd)

    def batches(self, maps: list[str], batch_maps: int, rng,
                shuffle: bool = True):
        """Yield lists of maps, cluster-disjoint within each batch."""
        order = list(maps)
        if shuffle:
            rng.shuffle(order)
        cur: list[str] = []
        used: set = set()
        for m in order:
            cl = self.clusters.get(m, set())
            if cur and (cl & used):
                continue                      # would collide; try it next epoch
            cur.append(m)
            used |= cl
            if len(cur) == batch_maps:
                yield cur
                cur, used = [], set()
        if len(cur) > 1:
            yield cur


def assemble(corpus: VoxelCorpus, emds: list[str], n_vox: int, rng,
             rot: int | None = None):
    """Concatenate a batch into flat arrays with GLOBAL residue indices.

    Returns feat [V,256], res_emb [R,1152], res_idx [V,K] (global, -1 pad),
    weight [V,K], w_bg [V], quota [V], vox_map [V], res_map [R], seq_of_res [R],
    chain_sizes [n_seq_total], xyz [V,3] or None.

    `xyz` is present only if the corpus was built with `coords_dir`. It is in the
    ROTATED frame, so a caller that mixes across voxels MUST pass `rot=` to keep
    a single frame per batch -- otherwise voxels from different rotations of the
    same map carry incompatible coordinates.

    `chain_sizes` is the residue count of each distinct SEQUENCE, in the order
    they appear in `res_emb`. The residue table is built grouped by sequence
    (`build_map_chains` emits positions 0..L-1 per sequence in turn), so those
    blocks are already contiguous -- which is what lets the pair channel run per
    chain without a gather.
    """
    F, RI, W, WB, Q, VM, G, RM, SR, CS = [], [], [], [], [], [], [], [], [], []
    XYZ = []
    off = 0
    for j, m in enumerate(emds):
        d = corpus.get(m)
        n_all = len(d["feat"])
        sel = np.arange(n_all)
        if rot is not None:
            sel = sel[d["rot"] == rot]
        if len(sel) > n_vox:
            sel = rng.choice(sel, n_vox, replace=False)
        ri = d["res_idx"][sel].astype(np.int64)
        F.append(d["feat"][sel].astype(np.float32))
        if "xyz" in d:
            XYZ.append(d["xyz"][sel])
        RI.append(np.where(ri >= 0, ri + off, -1))
        W.append(d["weight"][sel])
        WB.append(d["w_bg"][sel])
        Q.append(d["quota"][sel])
        VM.append(np.full(len(sel), j, dtype=np.int64))
        G.append(d["res_emb"])
        RM.append(np.full(len(d["res_emb"]), j, dtype=np.int64))
        sr = d["seq_of_res"].astype(np.int64)
        SR.append(sr)
        sizes = np.bincount(sr)
        assert (sizes > 0).all() and sizes.sum() == len(sr), (
            f"{m}: residue table is not grouped contiguously by sequence")
        CS.append(sizes)
        off += len(d["res_emb"])
    return (np.concatenate(F), np.concatenate(G), np.concatenate(RI),
            np.concatenate(W), np.concatenate(WB), np.concatenate(Q),
            np.concatenate(VM), np.concatenate(RM), np.concatenate(SR),
            np.concatenate(CS),
            np.concatenate(XYZ) if XYZ else None)
