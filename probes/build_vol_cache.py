"""STEP 1 -- cache WHOLE preprocessed volumes, so boxes can be cut later for free.

Caching volumes rather than boxes is deliberate. `preprocess` measured at 1.2-2.9 s
per map against `load_map`'s 0.0-0.4 s, and it is paid once per map here instead of
once per (map, centre, frame) configuration later. Boxes then come from
`extract_local_boxes` at whatever centres and frames an experiment wants -- random
SO(3) frames, backbone frames, patch-grid crops -- without re-paying it.

TWO THINGS THAT WOULD SILENTLY CORRUPT EVERY DOWNSTREAM NUMBER:

* **Preprocess the whole map, never per crop.** `preprocess` divides by the
  99.999th percentile of the volume it is given, so normalising a crop applies a
  DIFFERENT transform from the one every existing result in this project used.
* **fp32, not fp16.** fp16 costs ~5e-4 relative error on preprocessed values. The
  effects being chased here are sub-1%, so a 5e-4 floor is a confound that cannot
  be falsified after the fact. 78 GB is cheap insurance.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/cleandift_vols"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from probes.o1_cryofm_benchmark import to_cubic_even
    from probes.stability import load_map
    from teachers.cryofm_tap import MODEL_VOXEL_SIZE, preprocess

    rows = list(csv.DictReader(open(args.chains)))
    # One volume per EMDB entry, not per chain: 1,500 chains share ~1,150 maps.
    entries: dict[str, str] = {}
    for r in rows:
        entries.setdefault(Path(r["pdb"]).parent.name, r["pdb"])
    keys = sorted(entries)
    if args.limit:
        keys = keys[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} chains -> {len(keys)} unique maps -> {args.out_dir}", flush=True)

    done = skipped = cached = 0
    t0 = time.time()
    for i, k in enumerate(keys):
        npy = args.out_dir / f"{k}.npy"
        meta = args.out_dir / f"{k}.json"
        if npy.exists() and meta.exists():     # requeue-safe: jobs get preempted
            cached += 1
            continue
        try:
            d = Path(entries[k]).parent
            vol, vs, origin = load_map(str(d / f"{d.name}_raw_emd.map"))
            norm = preprocess(to_cubic_even(vol), vs).astype(np.float32)
            # Write through an open handle: `np.save(path, ...)` APPENDS ".npy"
            # to any path that does not already end in it, so a ".npy.tmp" temp
            # name is silently written as ".npy.tmp.npy" and the rename below then
            # fails on a path that never existed. A file object bypasses that
            # renaming entirely.
            tmp = npy.with_suffix(".tmp.npy")
            with open(tmp, "wb") as fh:
                np.save(fh, norm)
            tmp.replace(npy)                   # atomic: a preempted job never
            meta.write_text(json.dumps({       # leaves a half-written volume
                "emd": k,
                "origin_zyx": list(map(float, origin)),
                "voxel_size": MODEL_VOXEL_SIZE,   # post-resample, by construction
                "src_voxel_size": float(vs),
                "shape": list(map(int, norm.shape)),
                "dtype": "float32",
            }, indent=2))
            done += 1
        except Exception as exc:               # never raise out of the loop
            skipped += 1
            if skipped <= 10:
                print(f"  SKIP {k} {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(keys)}] {done} new, {cached} cached, "
                  f"{skipped} skipped, {el/60:.1f} min", flush=True)
    tot = sum(p.stat().st_size for p in args.out_dir.glob("*.npy"))
    print(f"maps: {done} new, {cached} cached, {skipped} skipped | "
          f"{tot/2**30:.1f} GiB | {(time.time()-t0)/60:.1f} min", flush=True)

    # Never let a 100%-failure run exit 0. The first version of this script wrote
    # 69.8 GiB of orphaned temp files, skipped all 1,147 maps, and reported
    # "done" -- the never-raise-out-of-the-loop pattern turned a total failure
    # into a successful-looking job. Assert the outcome, not just the attempt.
    ok = done + cached
    if ok == 0:
        raise SystemExit("FAILED: 0 maps cached. See the SKIP lines above.")
    if skipped > 0.5 * len(keys):
        raise SystemExit(f"FAILED: {skipped}/{len(keys)} maps skipped -- that is a "
                         f"systematic fault, not bad luck with individual entries.")
    stray = list(args.out_dir.glob("*.tmp.npy")) + list(args.out_dir.glob("*.npy.tmp*"))
    if stray:
        print(f"WARNING: {len(stray)} stray temp files left behind, e.g. "
              f"{stray[0].name}", flush=True)


if __name__ == "__main__":
    main()
