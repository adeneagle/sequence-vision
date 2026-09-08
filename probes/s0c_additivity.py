"""Stage 0c -- the additivity read. Does the density channel add anything
OVER ESM-C on a per-residue interface label?

All arms are fit and scored on IDENTICAL residues in one process (asserted),
with a cluster-level train/val/test split inherited from the cached parts.
The decisive quantity is `fused - esmc`.

Two arms from the plan are NOT here and their absence is load-bearing:

* `esmc_seqctx` (ESM-C + sequence-context attention, no density). The sibling
  measured raw-ESM interface MCC 0.236 -> 0.375 with context alone, while LieRE
  WITH coordinates gave 0.318. Since seqctx can only RAISE the sequence
  baseline, this run supports a ONE-SIDED inference only: a null here is
  decisive, a positive here is NOT.
* `docked_geometry`. Degenerate at Stage 0 -- with the deposited complex in hand
  it is the label generator. `own_geom` (monomer-only geometry) is the honest
  Stage-0 substitute and IS included.

Statistics: paired cluster bootstrap over test clusters. NOTE `o5_stats`
documents `mcnemar_exact` as secondary and anticonservative (it assumes
independent residues), so it is reported but never gated on.

Output: results/s0c_additivity.json
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AA1 = "ARNDCQEGHILKMFPSTWYV"
TAPS = ("mid_block", "up_blocks[0]", "up_blocks[1]")


def seq_window(seq, idx, w=3):
    L = len(seq)
    out = np.zeros((len(idx), 20 * (2 * w + 1)), np.float32)
    for r, i in enumerate(idx):
        for k, off in enumerate(range(-w, w + 1)):
            j = i + off
            if 0 <= j < L and seq[j] in AA1:
                out[r, k * 20 + AA1.index(seq[j])] = 1.0
    return out


def cluster_boot(y, sa, sb, clusters, metric, n_boot=1000, seed=0):
    """Paired metric difference (a - b) with a cluster-bootstrap 95% CI."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.nonzero(inv == k)[0] for k in range(len(uniq))]
    obs = metric(y, sa) - metric(y, sb)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([members[p] for p in pick])
        yy = y[ii]
        if yy.min() == yy.max():
            draws[b] = np.nan
            continue
        draws[b] = metric(yy, sa[ii]) - metric(yy, sb[ii])
    d = draws[~np.isnan(draws)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    se = float(d.std(ddof=1))
    return dict(diff=float(obs), ci_lo=float(lo), ci_hi=float(hi),
                se_cluster=se, mde=float(1.96 * se),
                excludes_zero=bool(lo > 0 or hi < 0), n_boot=int(len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/s0b_labels.npz")
    ap.add_argument("--feats", default="data/s0c_features.npz")
    ap.add_argument("--parts", default="data/o5_multitap_parts")
    ap.add_argument("--esmc", default="data/esmc_chains")
    ap.add_argument("--chains", default="data/alignment_chains.csv")
    ap.add_argument("--label", default="iface5_heavy")
    ap.add_argument("--arm", default="student_paperhead")
    ap.add_argument("--rand-arm", default="rand_student")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="results/s0c_additivity.json")
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 matthews_corrcoef, roc_auc_score)

    lab = np.load(a.labels, allow_pickle=True)
    ft = np.load(a.feats, allow_pickle=True)
    rows = {r["key"]: r for r in csv.DictReader(open(a.chains))}
    keys = sorted(set(k.split("/")[0] for k in lab.files if k.endswith("/idx")))
    keys = [k for k in keys
            if (Path(a.parts) / f"{k}.npz").exists()
            and Path(f"{a.esmc}/{k}.npy").exists()
            and f"{k}/own_geom" in ft.files
            and f"{k}/dens_stats" in ft.files]
    if a.limit:
        keys = keys[:a.limit]

    B = {n: [] for n in ["esmc", "aa", "seqwin3", "own_geom", "dens_stats", "raw"]}
    for t in TAPS:
        B[f"dens@{t}"] = []
        B[f"rand@{t}"] = []
    y, split, clus = [], [], []

    for key in keys:
        p = np.load(Path(a.parts) / f"{key}.npz", allow_pickle=True)
        idx = lab[f"{key}/idx"].astype(int)
        yy = lab[f"{key}/{a.label}"]
        e = np.load(f"{a.esmc}/{key}.npy", mmap_mode="r")
        if len(yy) != len(idx) or len(p["aa"]) != len(idx):
            raise SystemExit(f"{key}: row misalignment")
        B["esmc"].append(np.asarray(e[idx], np.float32))
        oh = np.zeros((len(idx), 20), np.float32)
        good = p["aa"] >= 0
        oh[np.arange(len(idx))[good], p["aa"][good]] = 1.0
        B["aa"].append(oh)
        B["seqwin3"].append(seq_window(rows[key]["seq"], idx))
        B["own_geom"].append(ft[f"{key}/own_geom"])
        B["dens_stats"].append(np.nan_to_num(ft[f"{key}/dens_stats"]))
        B["raw"].append(p["raw"].astype(np.float32))
        for t in TAPS:
            B[f"dens@{t}"].append(p[f"{a.arm}@{t}"].astype(np.float32))
            B[f"rand@{t}"].append(p[f"{a.rand_arm}@{t}"].astype(np.float32))
        y.append(yy.astype(int))
        split.append(p["split"])
        clus.append(p["cluster"])

    B = {k: np.concatenate(v) for k, v in B.items()}
    y = np.concatenate(y); split = np.concatenate(split); clus = np.concatenate(clus)
    n = len(y)
    assert all(len(v) == n for v in B.values()), "arm length mismatch"
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    print(f"{len(keys)} chains | {n} residues | train {tr.sum()}"
          f" val {va.sum()} test {te.sum()} | test clusters"
          f" {len(set(clus[te]))} | pos rate {y.mean():.4f}"
          f" (test {y[te].mean():.4f}, {int(y[te].sum())} positives)")

    arms = {}
    arms["shuffled"] = B["esmc"]
    for k in ["aa", "seqwin3", "own_geom", "dens_stats", "raw", "esmc"]:
        arms[k] = B[k]
    for t in TAPS:
        arms[f"dens@{t}"] = B[f"dens@{t}"]
        arms[f"rand@{t}"] = B[f"rand@{t}"]
        arms[f"fused@{t}"] = np.hstack([B["esmc"], B[f"dens@{t}"]])
    arms["fused_geom"] = np.hstack([B["esmc"], B["own_geom"], B["dens_stats"]])
    for t in TAPS:
        # ESM-C + cheap geometry + CryoFM: isolates the foundation model's
        # MARGINAL contribution over features that cost nothing to compute.
        arms[f"fused_all@{t}"] = np.hstack(
            [B["esmc"], B["own_geom"], B["dens_stats"], B[f"dens@{t}"]])

    rng = np.random.default_rng(0)
    perm = rng.permutation(int(te.sum()))
    res, scores = {}, {}
    for name, X in arms.items():
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-8] = 1.0
        Z = (X - mu) / sd
        best = None
        for C in (0.01, 0.1, 1.0):
            m = LogisticRegression(C=C, max_iter=3000)
            m.fit(Z[tr], y[tr])
            sv = m.decision_function(Z[va])
            ap_v = average_precision_score(y[va], sv)
            if best is None or ap_v > best[0]:
                best = (ap_v, C, m)
        ap_v, C, m = best
        st_ = m.decision_function(Z[te])
        if name == "shuffled":
            st_ = st_[perm]
        # MCC threshold tuned on val
        sv = m.decision_function(Z[va])
        qs = np.quantile(sv, np.linspace(0.02, 0.98, 49))
        thr = max(qs, key=lambda q: matthews_corrcoef(y[va], sv > q))
        scores[name] = st_
        res[name] = dict(
            dim=int(X.shape[1]), C=float(C), val_ap=float(ap_v),
            ap=float(average_precision_score(y[te], st_)),
            auroc=float(roc_auc_score(y[te], st_)),
            mcc=float(matthews_corrcoef(y[te], st_ > thr)),
            bal_acc=float(balanced_accuracy_score(y[te], st_ > thr)))
        print(f"  {name:18s} dim {X.shape[1]:5d} AP {res[name]['ap']:.4f}"
              f"  AUROC {res[name]['auroc']:.4f}  MCC {res[name]['mcc']:.4f}")

    # tap selected ON VAL, as the o5 harness does
    best_tap = max(TAPS, key=lambda t: res[f"fused@{t}"]["val_ap"])
    apm = lambda yy, ss: average_precision_score(yy, ss)
    comps = {
        "fused-esmc": (f"fused@{best_tap}", "esmc"),
        "fused-own_geom": (f"fused@{best_tap}", "own_geom"),
        "fused-dens_stats": (f"fused@{best_tap}", "dens_stats"),
        "dens-rand": (f"dens@{best_tap}", f"rand@{best_tap}"),
        "dens-dens_stats": (f"dens@{best_tap}", "dens_stats"),
        "dens-raw": (f"dens@{best_tap}", "raw"),
        "esmc-seqwin3": ("esmc", "seqwin3"),
        "fused_geom-esmc": ("fused_geom", "esmc"),
        # THE decisive pair: does CryoFM earn its place over cheap geometry?
        "fused-fused_geom": (f"fused@{best_tap}", "fused_geom"),
        "fused_all-fused_geom": (f"fused_all@{best_tap}", "fused_geom"),
        "own_geom-dens_stats": ("own_geom", "dens_stats"),
    }
    boot = {}
    for nm, (x, z) in comps.items():
        boot[nm] = cluster_boot(y[te], scores[x], scores[z], clus[te], apm,
                                n_boot=a.n_boot)
        b = boot[nm]
        print(f"  dAP {nm:20s} {b['diff']:+.4f}"
              f"  CI [{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]"
              f"  MDE {b['mde']:.4f}"
              f"  {'SIG' if b['excludes_zero'] else 'ns'}")

    out = dict(_meta=dict(label=a.label, arm=a.arm, chains=len(keys),
                          residues=int(n), test_residues=int(te.sum()),
                          test_positives=int(y[te].sum()),
                          test_clusters=len(set(clus[te])),
                          pos_rate_test=float(y[te].mean()),
                          best_tap_by_val=best_tap, n_boot=a.n_boot,
                          missing_arms=["esmc_seqctx", "docked_geometry"],
                          inference="one-sided: null is decisive, positive is not"),
               arms=res, deltas=boot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nbest tap (val) {best_tap}\n-> {a.out}")


if __name__ == "__main__":
    main()
