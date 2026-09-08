"""Box sampling for CleanDIFT training, cut from the volume cache (D10, D11).

D10 -- a 50/50 MIXTURE, not one distribution:

  * 50% Ca-centred 64^3 boxes at uniform random SO(3) frames. The eval feeds
    Ca-centred, frame-rotated, trilinearly resampled boxes, so a student trained
    only on patch-grid crops is off-distribution at eval time (`RotCube24` is a
    lossless transpose+flip; `grid_sample` boxes are interpolation-blurred).
    Random SO(3) rather than backbone frames deliberately: it covers both the
    `aligned` and the `lab*` eval arms and avoids committing the student to the
    atomic-model-dependent frame distribution.
  * 50% patch-grid crops with a random cube rotation, which is what CryoFM2's own
    pretraining saw and what the `feature_volumes` tiling use case consumes.

D11 -- the split is enforced at MAP level, not cluster level. An EMDB entry can
host chains in different splits (measured: 64 entries host both train and test
chains, 50 host train and val), so training on a train chain from a shared entry
would show the student, unsupervised, the very density that contains test
residues -- exposure no teacher arm ever had.

PADDING SUBTLETY, and it matters for training as much as for eval:
`extract_local_boxes` uses `padding_mode="zeros"`, and 0 in preprocessed units is
raw density 0.04, which is ABOVE background (raw 0 maps to -0.44). An
out-of-bounds box therefore gets a slab of mean-density rather than vacuum. The
`ok` filter below (Ca at least PATCH//2 from every edge) is the same one the eval
applies, so the student never trains on artifacts the eval will not show it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from teachers.cryofm_tap import PATCH, cube_rotations


def split_map_lists(chains_csv: Path) -> dict:
    """Map-level split lists plus the assertions D11 demands."""
    rows = list(csv.DictReader(open(chains_csv)))
    by_split: dict[str, set] = {}
    for r in rows:
        by_split.setdefault(r["split"], set()).add(Path(r["pdb"]).parent.name)
    train, val, test = (by_split.get(k, set()) for k in ("train", "val", "test"))
    clean_train = train - val - test
    rows_train = [r for r in rows
                  if r["split"] == "train" and Path(r["pdb"]).parent.name in clean_train]
    rows_val = [r for r in rows if r["split"] == "val"]

    assert not (clean_train & test), "train maps leak into test"
    assert not (clean_train & val), "train maps leak into val"
    clusters_train = {r["cluster"] for r in rows_train}
    clusters_test = {r["cluster"] for r in rows if r["split"] == "test"}
    assert not (clusters_train & clusters_test), "train clusters leak into test"

    info = {
        "n_rows_all": len(rows),
        "train_maps_raw": len(train), "train_maps_clean": len(clean_train),
        "dropped_shared_with_test": len(train & test),
        "dropped_shared_with_val": len(train & val),
        "train_chains": len(rows_train), "train_clusters": len(clusters_train),
        "val_chains": len(rows_val),
    }
    return {"train": rows_train, "val": rows_val, "info": info}


class BoxStream:
    """Infinite stream of [B, 1, 64, 64, 64] preprocessed boxes.

    A shuffle buffer over several maps is not optional: without it every batch of
    8 comes from a single map, which correlates the gradients AND the centring EMA
    the loss depends on.
    """

    def __init__(self, rows, vol_dir: Path, batch: int = 8, per_map: int = 64,
                 mix_local: float = 0.5, buffer_maps: int = 6, seed: int = 0,
                 device: str = "cuda", box_chunk: int = 16):
        self.by_map: dict[str, list] = {}
        for r in rows:
            self.by_map.setdefault(Path(r["pdb"]).parent.name, []).append(r)
        self.keys = sorted(self.by_map)
        self.vol_dir = Path(vol_dir)
        self.batch, self.per_map = batch, per_map
        self.mix_local, self.buffer_maps = mix_local, buffer_maps
        self.rng = np.random.default_rng(seed)
        self.device, self.box_chunk = device, box_chunk
        self.CUBES = cube_rotations()
        self.skipped: list[str] = []

    # -- one map's worth of boxes ------------------------------------------
    def _boxes_for_map(self, key: str) -> torch.Tensor | None:
        from probes.local_frame_stability import extract_local_boxes, random_so3
        from probes.o1_cryofm_benchmark import backbone_with_resnum
        from probes.o4_frameavg_benchmark import torch_cube_rotate

        npy, meta = self.vol_dir / f"{key}.npy", self.vol_dir / f"{key}.json"
        if not (npy.exists() and meta.exists()):
            return None
        m = json.loads(meta.read_text())
        vol = np.load(npy, mmap_mode="r")
        origin = np.asarray(m["origin_zyx"])
        vs = m["voxel_size"]
        shape = np.array(vol.shape)

        n_local = int(round(self.per_map * self.mix_local))
        n_crop = self.per_map - n_local
        out = []

        if n_local > 0:
            cs = []
            for r in self.by_map[key]:
                try:
                    _seq, ca, _fr, _nums = backbone_with_resnum(r["pdb"], r["chain"])
                except Exception:
                    continue
                c = (ca - origin[None]) / vs
                ok = np.all((c >= PATCH // 2) & (c < shape[None] - PATCH // 2), axis=1)
                if ok.any():
                    cs.append(c[ok])
            if cs:
                c = np.concatenate(cs)
                idx = self.rng.choice(len(c), min(n_local, len(c)),
                                      replace=len(c) < n_local)
                frames = np.stack([random_so3(self.rng) for _ in idx])
                vt = torch.from_numpy(np.ascontiguousarray(vol))
                out.append(extract_local_boxes(vt, c[idx], frames, device=self.device,
                                               chunk=self.box_chunk).cpu())
                del vt

        if n_crop > 0 and np.all(shape >= PATCH):
            hi = shape - PATCH
            for _ in range(n_crop):
                # Patch-GRID starts: multiples of PATCH where possible, which is
                # what `feature_volumes` actually tiles with.
                s = [int(self.rng.integers(0, h // PATCH + 1)) * PATCH for h in hi]
                s = [min(v, int(h)) for v, h in zip(s, hi)]
                blk = np.asarray(vol[s[0]:s[0] + PATCH, s[1]:s[1] + PATCH,
                                     s[2]:s[2] + PATCH], dtype=np.float32)
                x = torch.from_numpy(blk)[None, None]
                R = self.CUBES[int(self.rng.integers(len(self.CUBES)))]
                out.append(torch_cube_rotate(x, R).contiguous())

        del vol
        if not out:
            return None
        return torch.cat([o.float() for o in out])

    def __iter__(self):
        buf: list[torch.Tensor] = []
        while True:
            order = self.rng.permutation(len(self.keys))
            for oi in order:
                k = self.keys[int(oi)]
                try:
                    b = self._boxes_for_map(k)
                except Exception as exc:
                    self.skipped.append(f"{k}: {type(exc).__name__}: {exc}")
                    b = None
                if b is not None:
                    buf.append(b)
                if len(buf) >= self.buffer_maps:
                    pool = torch.cat(buf)
                    buf = []
                    perm = torch.from_numpy(self.rng.permutation(len(pool)))
                    pool = pool[perm]
                    for s in range(0, len(pool) - self.batch + 1, self.batch):
                        yield pool[s:s + self.batch]
