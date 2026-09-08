"""E2 -- map -> sequence retrieval, the zero-shot-classification analogue.

Query is a map's foreground voxels with NO atomic model; the answer is which of
K candidate sequences is actually in it. Scored by LATE INTERACTION with 1/|V|
normalisation: score(map, s) = mean_v max_{r in s} sim(v, r). The max picks each
voxel's best-matching residue, the mean stops long sequences winning by having
more chances -- which is the same size shortcut E1 avoids by being within-map.

TWO CONTROLS THAT DECIDE WHETHER THE NUMBER MEANS ANYTHING:
  * LENGTH-MATCHED POOLS, by NEAREST NEIGHBOUR not by tolerance band. Map size
    correlates with sequence length, so an unmatched pool is largely solvable by
    size alone -- the same shape as the coordinate leakage that inflated this
    project's contact probe to AUC 0.997. A first version drew distractors
    randomly from a +-0.25 log-length band and scored the control against the
    TRUE sequence's length, where the answer sits at distance exactly 0: the
    control read 1.0000 and the run was void. Distractors are now the K nearest
    in length, and the control is scored against a MAP-SIDE size proxy (the
    map's total observed residues), never against the answer. Realised pool
    length spread is reported so the matching can be checked directly.
  * CLUSTER-DISJOINT DISTRACTORS. A distractor homologous to the answer is not a
    distractor. Candidates are drawn only from maps sharing no 30%-identity
    cluster with the query.

Chance is 1/pool_size, and pool_size is capped by how many length-matched
cluster-disjoint candidates exist, so it is reported as the realised mean 1/K
rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/voxel_cache"))
    ap.add_argument("--esmc", type=Path, default=Path("data/esmc_seq32"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("data/dinotxt_runs/v2_nc500_long/best.pt"))
    ap.add_argument("--aa-ckpt", type=Path,
                    default=Path("data/dinotxt_runs/aa_baseline.pt"))
    ap.add_argument("--pool", type=int, default=11,
                    help="candidates incl. the answer; chance = 1/pool")
    ap.add_argument("--tol", type=float, default=0.25,
                    help="max |log(L_cand / L_true)| for a length match")
    ap.add_argument("--coords", type=Path, default=None,
                    help="required if --ckpt was trained with --mix-depth")
    ap.add_argument("--n-vox", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/dinotxt_e2.json"))
    args = ap.parse_args()

    from probes.dinotxt_data import VoxelCorpus
    from probes.dinotxt_model import DinoTxt
    from probes.o5_stats import cluster_bootstrap_diff, mdi

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = VoxelCorpus(args.cache, args.esmc, coords_dir=args.coords)
    emds = corpus.by_split("test")
    rng = np.random.default_rng(args.seed)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a, d0 = ck["args"], corpus.get(emds[0])
    model = DinoTxt(d_vox=d0["feat"].shape[1], d_seq=d0["res_emb"].shape[1],
                    d=a["dim"], hidden=a["hidden"], depth=a["depth"],
                    tau_init=a["tau"], pair=a.get("pair", "none"),
                    n_tri=a.get("n_tri", 2), d_pair=a.get("d_pair", 32),
                    mix_depth=a.get("mix_depth", 0),
                    mix_heads=a.get("mix_heads", 4),
                    mix_rmax=a.get("mix_rmax", 30.0)).to(dev).eval()
    model.load_state_dict(ck["model"])
    mixing = getattr(model, "mix", None) is not None
    if mixing and args.coords is None:
        raise SystemExit("--ckpt mixes but no --coords given")

    aa_net = None
    if args.aa_ckpt and args.aa_ckpt.exists():
        import torch.nn as nn
        ac = torch.load(args.aa_ckpt, map_location=dev, weights_only=False)
        h = ac["args"]["hidden"]
        aa_net = (nn.Linear(ac["d_in"], 20) if h == 0 else
                  nn.Sequential(nn.LayerNorm(ac["d_in"]),
                                nn.Linear(ac["d_in"], h), nn.GELU(),
                                nn.Linear(h, 20))).to(dev).eval()
        aa_net.load_state_dict(ac["net"])

    # Catalogue every distinct sequence in the test split, with the map it came
    # from (for cluster exclusion) and its length (for matching).
    cat = []
    for m in emds:
        d = corpus.get(m)
        sr = d["seq_of_res"].astype(np.int64)
        for s in range(int(sr.max()) + 1):
            cat.append((m, s, int((sr == s).sum()), str(d["seqs"][s])))
    lens = np.array([c[2] for c in cat], dtype=np.float64)
    print(f"{len(emds)} test maps | {len(cat)} distinct sequences | {dev}",
          flush=True)

    AAS = "ACDEFGHIKLMNPQRSTVWY"
    aa_i = {c: i for i, c in enumerate(AAS)}

    def comp(seq: str) -> np.ndarray:
        v = np.ones(20)                                     # Laplace
        for ch in seq:
            if ch in aa_i:
                v[aa_i[ch]] += 1
        return v / v.sum()

    clen = np.array([len(c[3]) for c in cat], dtype=np.float64)
    hit_r, hit_a, hit_l, chance, ks, cl, spread = [], [], [], [], [], [], []
    with torch.no_grad():
        for m in emds:
            d = corpus.get(m)
            sr = d["seq_of_res"].astype(np.int64)
            true_s = int(np.bincount(sr).argmax())          # the map's largest
            L = float((sr == true_s).sum())
            mine = corpus.clusters.get(m, set())

            Ltrue = len(str(d["seqs"][true_s]))
            ok = np.array([
                (c[0] != m) and not (corpus.clusters.get(c[0], set()) & mine)
                for c in cat])
            idx = np.nonzero(ok)[0]
            if len(idx) < 2:
                continue                                    # no honest pool
            # K NEAREST in length, so the pool is as tightly matched as the
            # corpus allows and length carries as little signal as possible.
            dist = np.abs(np.log(np.maximum(clen[idx], 1) / max(Ltrue, 1)))
            k = min(args.pool - 1, len(idx))
            pick = idx[np.argsort(dist)[:k]]
            pool = [(m, true_s)] + [(cat[i][0], cat[i][1]) for i in pick]
            pool_seq = [str(d["seqs"][true_s])] + [cat[i][3] for i in pick]
            cl_pool = [len(q) for q in pool_seq]

            # A mixing model must see ONE rotation frame (coordinates are
            # stored rotated), so restrict to rotation 0.
            keep0 = ~d["is_bg"]
            if mixing:
                keep0 = keep0 & (d["rot"] == 0)
            sel = np.nonzero(keep0)[0]
            if len(sel) > args.n_vox:
                sel = np.sort(rng.choice(sel, args.n_vox, replace=False))
            fv = torch.from_numpy(d["feat"][sel].astype(np.float32)).to(dev)
            xyz = (torch.from_numpy(d["xyz"][sel]).to(dev) if mixing else None)
            h = model.encode_voxels(
                fv, xyz, torch.zeros(len(sel), dtype=torch.long, device=dev))

            # real arm: late interaction, 1/|V| normalised
            sc = []
            for (em, si) in pool:
                dd = corpus.get(em)
                rows = dd["seq_of_res"].astype(np.int64) == si
                g = torch.from_numpy(dd["res_emb"][rows]).to(dev)
                e = model.encode_residues(g, chain_sizes=np.array([rows.sum()]))
                sc.append(float((h @ e.t()).max(1).values.mean()))
            hit_r.append(int(np.argmax(sc) == 0))

            # baseline: predicted AA composition vs each candidate's composition
            if aa_net is not None:
                p = torch.softmax(aa_net(fv), 1).mean(0).cpu().numpy()
                lc = [float(p @ np.log(comp(s))) for s in pool_seq]
                hit_a.append(int(np.argmax(lc) == 0))

            # control: rank by length against a MAP-SIDE size proxy (total
            # observed residues), NOT against the answer's own length. This is
            # the honest "can size alone do it" arm; with nearest-K pools it
            # should sit near chance, and if it does not the corpus simply does
            # not contain tight enough matches to make E2 interpretable.
            Lmap = float(len(sr))
            dl = [abs(np.log(max(len(s), 1) / max(Lmap, 1))) for s in pool_seq]
            hit_l.append(int(np.argmin(dl) == 0))
            spread.append(float(np.log(max(cl_pool) / max(min(cl_pool), 1))))

            chance.append(1.0 / len(pool))
            ks.append(len(pool))
            cl.append(m)

    hit_r = np.array(hit_r); hit_l = np.array(hit_l)
    chance = np.array(chance); cl = np.array(cl); ks = np.array(ks)
    sp = np.array(spread)
    print(f"scored {len(hit_r)} maps | mean pool {ks.mean():.1f} "
          f"(min {ks.min()}, max {ks.max()}) | chance {chance.mean():.4f}")
    print(f"pool length spread |log(Lmax/Lmin)|: median {np.median(sp):.3f} "
          f"p90 {np.percentile(sp, 90):.3f}  (0 = perfectly matched)\n")

    res = {"_meta": {"n_maps": int(len(hit_r)), "mean_pool": float(ks.mean()),
                     "tol": args.tol, "ckpt": str(args.ckpt),
                     "pool_len_spread_median": float(np.median(sp)),
                     "pool_len_spread_p90": float(np.percentile(sp, 90)),
                     "step": ck.get("step")},
           "acc": {"retrieval": float(hit_r.mean()),
                   "length_only": float(hit_l.mean()),
                   "chance": float(chance.mean())}}
    if len(hit_a):
        res["acc"]["aa_composition"] = float(np.array(hit_a).mean())
    for k, v in res["acc"].items():
        print(f"  {k:16s} {v:.4f}")

    res["vs"] = {}
    for name, arm in [("chance", chance), ("length_only", hit_l)] + (
            [("aa_composition", np.array(hit_a))] if len(hit_a) else []):
        dd = cluster_bootstrap_diff(hit_r, arm, cl)
        res["vs"][name] = dd
        print(f"\n  retrieval - {name}: {dd['diff']:+.4f} "
              f"CI [{dd['lo95']:+.4f}, {dd['hi95']:+.4f}] MDE {mdi(dd):.4f} "
              f"{'SIG' if dd['excludes_zero'] else 'ns'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
