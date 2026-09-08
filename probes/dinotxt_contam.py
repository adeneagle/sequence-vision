"""E1 split by CryoFM2 pretrain contamination -- is the gate inflated by seen maps?

15.1% of Cryo2StructData is in CryoFM2's published pretrain TRAIN list
(Zenodo 18013604). Those maps' density features come from a model that saw them,
so E1 on them is not a clean generalisation number. This splits the SAME voxels
the gate scored into seen/unseen and reports both, plus the paired cluster
bootstrap of trained-minus-volume-prior WITHIN each subset.

The unseen subset is the number to quote externally. The comparison between
subsets is descriptive, not a test: the two are different maps, so a difference
confounds contamination with whatever else differs between the two populations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def pretrain_ids(d: Path) -> set[int]:
    """EMDB ids in the CryoFM2 pretrain TRAIN list. Rows are 2x entries (each
    half-map pair appears twice with map_path1/2 swapped), so dedupe."""
    ids = set()
    for r in csv.DictReader(open(d / "train.csv")):
        ids.add(int(r["emdb_id"].split("-")[1]))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/voxel_cache"))
    ap.add_argument("--esmc", type=Path, default=Path("data/esmc_seq32"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("data/dinotxt_runs/v2_nc500_long/best.pt"))
    ap.add_argument("--lists", type=Path,
                    default=Path("data/cryofm2_pretrain_lists"))
    ap.add_argument("--n-vox", type=int, default=800)
    ap.add_argument("--out", type=Path,
                    default=Path("results/dinotxt_contam.json"))
    args = ap.parse_args()

    from probes.dinotxt_data import VoxelCorpus
    from probes.dinotxt_model import DinoTxt
    from probes.dinotxt_eval import eval_arm
    from probes.o5_stats import cluster_bootstrap_diff, mdi

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = VoxelCorpus(args.cache, args.esmc)
    emds = corpus.by_split("test")
    seen = pretrain_ids(args.lists)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a, d0 = ck["args"], corpus.get(emds[0])
    model = DinoTxt(d_vox=d0["feat"].shape[1], d_seq=d0["res_emb"].shape[1],
                    d=a["dim"], hidden=a["hidden"], depth=a["depth"],
                    tau_init=a["tau"], pair=a.get("pair", "none"),
                    n_tri=a.get("n_tri", 2),
                    d_pair=a.get("d_pair", 32)).to(dev).eval()
    model.load_state_dict(ck["model"])

    ok_t, cl, _, _ = eval_arm(model, corpus, emds, args.n_vox, dev)
    ok_p, _, _, _ = eval_arm(model, corpus, emds, args.n_vox, dev,
                             prior_only=True)

    is_seen = np.array([int(m) in seen for m in cl])
    n_seen_maps = len({m for m in cl if int(m) in seen})
    print(f"test maps {len(set(cl))} | in CryoFM2 pretrain TRAIN: {n_seen_maps} "
          f"({n_seen_maps / len(set(cl)):.1%})")
    print(f"voxels {len(cl)} | seen {is_seen.sum()} ({is_seen.mean():.1%})\n")

    res: dict = {"_meta": {"ckpt": str(args.ckpt), "step": ck.get("step"),
                           "n_maps": len(set(cl)), "n_seen_maps": n_seen_maps}}
    for name, m in (("seen", is_seen), ("unseen", ~is_seen)):
        if m.sum() == 0:
            continue
        d = cluster_bootstrap_diff(ok_t[m], ok_p[m], cl[m])
        res[name] = {"n_voxels": int(m.sum()),
                     "n_maps": int(len(np.unique(cl[m]))),
                     "trained": float(ok_t[m].mean()),
                     "volume_prior": float(ok_p[m].mean()), "vs_prior": d}
        print(f"  {name:7s} n={m.sum():>6d} maps={len(np.unique(cl[m])):>3d} | "
              f"trained {ok_t[m].mean():.4f} prior {ok_p[m].mean():.4f} | "
              f"diff {d['diff']:+.4f} CI [{d['lo95']:+.4f}, {d['hi95']:+.4f}] "
              f"MDE {mdi(d):.4f} {'SIG' if d['excludes_zero'] else 'ns'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
