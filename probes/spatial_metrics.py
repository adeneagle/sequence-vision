"""Spatial fidelity as a PRIMARY metric, not a degeneracy guard.

`spatial` entered the relational analysis as a guard: conv_in scored 0.06 while
being highly pose-stable, proving a feature can be perfectly invariant and carry
no geometry. Under the geometric reading of the relational objective -- we want
residue-scale features whose similarity structure tracks 3D arrangement -- it
stops being a guard and becomes the thing being optimised. That promotion
demands what any primary metric needs and the guard version never had:

  a FLOOR      permute the residue<->feature assignment. Destroys the geometry
               link while preserving every marginal, so it is the honest zero.
               An absolute correlation without its floor is uninterpretable --
               the lesson this project has now paid for three times.
  a SECOND     `spatial_knn` asks a local, rank-free question (do a residue's
  ESTIMATOR    top-k neighbours in FEATURE space coincide with its top-k in 3D?)
               where `spatial_spearman` asks a global monotone one. They fail
               differently: a feature can rank all pairs roughly right while
               getting no neighbourhood exactly right, and vice versa.
  ANNOTATION   both depend on n_res and on the structure population, exactly as
               held-out Procrustes does (the same n_res=120 gave 0.820 on chains
               80-400 and 0.659 on chains >=400). Callers must record both.

Self-test at the bottom runs a KNOWN-GOOD and a KNOWN-BAD input, because a
metric that cannot fail its own smoke test should not be trusted on real data.

PRE-REGISTERED TIE-BREAK, fixed 2026-08-27 BEFORE seeing which answer it favours,
because deciding after the fact is how a metric gets chosen to fit a conclusion:

    kNN GOVERNS. Spearman is the sanity check.

Reason, and it is an asymmetry rather than a preference. A feature with high
Spearman and floor-level kNN knows coarse near-from-far but cannot identify any
actual neighbour -- useless for residue-scale work, which is the stated goal. The
converse (high kNN, low Spearman) is strange but still locally usable. So kNN is
both the more demanding and the more relevant test.

TWO HONEST CAVEATS ON THAT CHOICE:
  * It creates an objective/metric mismatch. The relational objective as designed
    predicts the FULL similarity matrix, which is a global quantity, while kNN is
    local. If kNN governs and the objective is global, either the objective
    should grow a local term or the mismatch should be stated in the write-up.
  * A CONFLICT BETWEEN THE TWO IS A FINDING, not just something the rule
    resolves. Report both numbers whenever they disagree and say so explicitly;
    silently applying the tie-break would hide the more interesting result.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _sim(X: np.ndarray) -> np.ndarray:
    """Centred cosine similarity. Centring is mandatory -- raw cosines on
    high-dim CNN activations sit near 0.99 for any pair."""
    Xc = X.astype(np.float64)
    Xc = Xc - Xc.mean(0, keepdims=True)
    Xc /= np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
    return Xc @ Xc.T


def spatial_spearman(X: np.ndarray, xyz: np.ndarray) -> float:
    """Global: Spearman(-feature similarity, 3D distance) over residue pairs."""
    S, D = _sim(X), np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    iu = np.triu_indices(len(S), k=1)
    return float(spearmanr(-S[iu], D[iu]).statistic)


def knn_chance(n: int, k: int) -> float:
    """Expected top-k overlap under a random correspondence: exactly k/(n-1).

    ANALYTIC, not empirical: under a permutation each residue's k feature
    neighbours are a uniform k-subset of the other n-1, so the expected
    intersection with a fixed k-set is k^2/(n-1), i.e. k/(n-1) as a fraction.
    Measuring it is an implementation check worth doing once, not a per-run cost.
    """
    return k / (n - 1)


def spatial_knn(X: np.ndarray, xyz: np.ndarray, k: int = 10,
                corrected: bool = True) -> float:
    """Local: overlap between each residue's top-k FEATURE and top-k 3D neighbours.

    Returned CHANCE-CORRECTED by default: (overlap - chance)/(1 - chance), which
    is 0 at chance and 1 at perfect FOR ANY n. This is not cosmetic -- raw
    overlap is n-dependent by construction (chance is 0.127 at n=80, 0.042 at
    n=240, 0.025 at n=400), so two runs at different n_res are NOT comparable
    even if both annotate n_res. The corrected form is.
    """
    S = _sim(X)
    D = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    np.fill_diagonal(S, -np.inf)
    np.fill_diagonal(D, np.inf)
    kf = np.argsort(-S, axis=1)[:, :k]
    kd = np.argsort(D, axis=1)[:, :k]
    raw = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(kf, kd)]))
    if not corrected:
        return raw
    c = knn_chance(len(X), k)
    return (raw - c) / (1.0 - c)


def spatial_all(X: np.ndarray, xyz: np.ndarray, k: int = 10,
                n_floor: int = 8, seed: int = 0) -> dict:
    """Both estimators plus their permutation floors.

    The floor permutes residue<->feature assignment: every marginal of X and of
    xyz is preserved, only the correspondence is broken. That isolates "this
    feature encodes THIS residue's position" from "these features and these
    positions are individually structured".
    """
    rng = np.random.default_rng(seed)
    out = {"spearman": spatial_spearman(X, xyz),
           "knn": spatial_knn(X, xyz, k),                       # chance-corrected
           "knn_raw": spatial_knn(X, xyz, k, corrected=False)}
    fs, fk = [], []
    for _ in range(n_floor):
        p = rng.permutation(len(X))
        fs.append(spatial_spearman(X[p], xyz))
        fk.append(spatial_knn(X[p], xyz, k))
    out["spearman_floor"] = float(np.median(fs))
    out["knn_floor"] = float(np.median(fk))
    out["knn_chance"] = knn_chance(len(X), k)
    out["spearman_sep"] = out["spearman"] - out["spearman_floor"]
    out["knn_sep"] = out["knn"] - out["knn_floor"]
    out["n_res"] = int(len(X))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, C = 80, 64
    xyz = rng.normal(size=(n, 3)) * 20.0

    # KNOWN-GOOD: the feature IS a smooth function of position, so similarity
    # must track 3D distance.
    W = rng.normal(size=(3, C))
    good = xyz @ W + 0.05 * rng.normal(size=(n, C))
    # KNOWN-BAD 1: pure noise, no relation to position.
    bad = rng.normal(size=(n, C))
    # KNOWN-BAD 2: degenerate -- every residue nearly identical. Highly
    # "stable", carries nothing. This is the conv_in failure mode.
    degen = np.ones((n, C)) + 1e-3 * rng.normal(size=(n, C))

    print(f"{'input':<12}{'spearman':>10}{'floor':>8}{'sep':>8}"
          f"{'knn':>8}{'floor':>8}{'chance':>8}{'sep':>8}")
    for name, X in (("geometric", good), ("noise", bad), ("degenerate", degen)):
        r = spatial_all(X, xyz)
        print(f"{name:<12}{r['spearman']:>10.3f}{r['spearman_floor']:>8.3f}"
              f"{r['spearman_sep']:>8.3f}{r['knn']:>8.3f}{r['knn_floor']:>8.3f}"
              f"{r['knn_chance']:>8.3f}{r['knn_sep']:>8.3f}")

    g, b, d = (spatial_all(X, xyz) for X in (good, bad, degen))
    assert g["spearman_sep"] > 0.5, "known-good input failed to register as geometric"
    assert abs(b["spearman_sep"]) < 0.1, "noise should sit at the floor"
    assert abs(d["spearman_sep"]) < 0.1, "degenerate input should sit at the floor"
    assert g["knn_sep"] > 0.3 and abs(b["knn_sep"]) < 0.1
    print("\nOK: separates geometric from noise AND from degenerate")
