"""
select_probe.py -- how many features are actually worth keeping.

Answers two questions hull_probe.py cannot:
  * does the model benefit from all ~1100 columns, or does the curve flatten
    long before that;
  * is the importance ranking itself stable enough to select on.

The second question comes first. If the top-k set turns over completely
between adjacent folds, the ranking is fitting noise and no value of k
matters -- and a feature's "negative importance" is noise too, which is the
most likely explanation for pruning experiments that backfire.

Wiring into hull_probe.py:
    from select_probe import rank_features_in_window
    def _select_in_window(X, y, n_keep, seed):
        return rank_features_in_window(X, y, seed)[:n_keep]
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from wfo import Fold, paired_compare, run_wfo, summarize

# importance_type must be explicit. The sklearn wrapper defaults to "split",
# which counts how often a feature was used for a cut -- that rewards
# high-cardinality continuous columns for being convenient to partition on,
# regardless of whether the splits helped. "gain" sums the actual loss
# reduction. The thresholds the original notebook selected on were split
# counts; they do not transfer to gain and would need re-tuning.
RANKER = dict(
    objective="regression",
    importance_type="gain",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=60,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_lambda=5.0,
    n_jobs=-1,
    verbose=-1,
)

TREE = dict(
    objective="regression",
    n_estimators=250,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    n_jobs=-1,
    verbose=-1,
)

_RANK_CACHE: Dict[Tuple[int, int, int, int], List[str]] = {}


# ============================================================
# Ranking, cached
# ============================================================

def rank_features_in_window(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 0,
    n_repeat: int = 2,
    cache_key: Optional[Tuple[int, int, int, int]] = None,
) -> List[str]:
    """
    Full importance ranking computed on one training window.

    n_repeat averages over bagging seeds before ranking. A single fit gives a
    ranking that reshuffles noticeably just from the subsample draw, which
    would show up later as selection instability that has nothing to do with
    the market.

    cache_key lets the sweep rank once per (fold, seed) and slice every k
    from the same ranking, instead of refitting for each k.
    """
    if cache_key is not None and cache_key in _RANK_CACHE:
        return _RANK_CACHE[cache_key]

    ok = np.isfinite(y)
    total = pd.Series(0.0, index=X.columns)
    for r in range(n_repeat):
        m = LGBMRegressor(**{**RANKER, "random_state": seed * 1000 + r})
        m.fit(X[ok], y[ok])
        imp = pd.Series(m.feature_importances_, index=X.columns, dtype=float)
        s = imp.sum()
        total += imp / s if s > 0 else 0.0

    ranking = total.sort_values(ascending=False).index.tolist()
    if cache_key is not None:
        _RANK_CACHE[cache_key] = ranking
    return ranking


def rankings_by_fold(
    df: pd.DataFrame,
    folds: Sequence[Fold],
    feature_cols: Sequence[str],
    target_col: str,
    seeds: Sequence[int] = (0,),
) -> Dict[Tuple[int, int], List[str]]:
    """Rank once per (fold, seed) and fill the cache the sweep reads from."""
    cols = list(feature_cols)
    y = df[target_col].to_numpy(float)
    out: Dict[Tuple[int, int], List[str]] = {}
    for f in folds:
        for s in seeds:
            key = (f.k, s, int(f.train[0]), int(f.train[-1]))
            out[(f.k, s)] = rank_features_in_window(
                df.iloc[f.train][cols], y[f.train], seed=s, cache_key=key
            )
        print(f"  ranked fold {f.k}", end="\r")
    print()
    return out


# ============================================================
# Stability
# ============================================================

def selection_stability(
    rankings: Dict[Tuple[int, int], List[str]],
    n_total: int,
    ks: Sequence[int] = (30, 60, 120, 300),
    seed: int = 0,
) -> pd.DataFrame:
    """
    Jaccard overlap of the top-k sets between adjacent folds.

    `random_baseline` is what two independent draws of k out of n_total would
    give: k / (2*n_total - k). Overlap near that line means the ranking
    carries no reusable information about which features matter, so any
    pruning decision taken from it -- including dropping features with
    negative measured importance -- is a coin flip.
    """
    fold_ids = sorted({k for k, s in rankings if s == seed})
    rows = []
    for k in ks:
        sets = [set(rankings[(fk, seed)][:k]) for fk in fold_ids]
        adj = [
            len(a & b) / len(a | b)
            for a, b in zip(sets[:-1], sets[1:])
            if len(a | b) > 0
        ]
        first_last = (
            len(sets[0] & sets[-1]) / len(sets[0] | sets[-1]) if len(sets) > 1 else np.nan
        )
        counts = pd.Series(0, index=sorted({c for s_ in sets for c in s_}), dtype=int)
        for s_ in sets:
            counts[list(s_)] += 1
        rows.append(
            {
                "k": k,
                "adjacent_jaccard": float(np.mean(adj)) if adj else np.nan,
                "first_vs_last_jaccard": first_last,
                "random_baseline": k / (2.0 * n_total - k),
                "features_ever_selected": int(len(counts)),
                "selected_in_all_folds": int((counts == len(sets)).sum()),
                "selected_in_half_or_more": int((counts >= len(sets) / 2).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def core_features(
    rankings: Dict[Tuple[int, int], List[str]], k: int = 60, min_frac: float = 0.5, seed: int = 0
) -> pd.Series:
    """Features that make the top-k in at least `min_frac` of folds, with their rate."""
    fold_ids = sorted({fk for fk, s in rankings if s == seed})
    counts: Dict[str, int] = {}
    for fk in fold_ids:
        for c in rankings[(fk, seed)][:k]:
            counts[c] = counts.get(c, 0) + 1
    s = pd.Series(counts, dtype=float) / len(fold_ids)
    return s[s >= min_frac].sort_values(ascending=False)


# ============================================================
# n_select sweep
# ============================================================

def make_tree_signal(
    feature_cols: Sequence[str],
    target_col: str,
    n_select: Optional[int],
    rankings: Optional[Dict[Tuple[int, int], List[str]]] = None,
):
    """
    Tree-only signal at a given selection width.

    No linear stage on purpose: a 756-row training window against ~1100
    columns has p > n, so SGD degenerates and the sweep would be measuring
    that rather than the features. LightGBM takes p > n and NaN natively, so
    the curve it traces is about the features alone.

    n_select=None keeps every column.
    """
    cols = list(feature_cols)

    def f(df, train_idx, test_idx, seed):
        y = df[target_col].to_numpy(float)
        if n_select is None:
            sel = cols
        elif rankings is not None:
            sel = rankings[(_fold_of(train_idx, rankings, seed), seed)][:n_select]
        else:
            sel = rank_features_in_window(df.iloc[train_idx][cols], y[train_idx], seed)[:n_select]

        m = LGBMRegressor(**{**TREE, "random_state": seed})
        ok = np.isfinite(y[train_idx])
        m.fit(df.iloc[train_idx[ok]][sel], y[train_idx[ok]])
        return m.predict(df.iloc[test_idx][sel])

    return f


_FOLD_LOOKUP: Dict[Tuple[int, int], int] = {}


def register_folds(folds: Sequence[Fold]) -> None:
    """make_tree_signal only receives index arrays, so map them back to fold ids."""
    _FOLD_LOOKUP.clear()
    for f in folds:
        _FOLD_LOOKUP[(int(f.train[0]), int(f.train[-1]))] = f.k


def _fold_of(train_idx, rankings, seed) -> int:
    key = (int(train_idx[0]), int(train_idx[-1]))
    if key not in _FOLD_LOOKUP:
        raise KeyError("call register_folds(folds) before running the sweep")
    return _FOLD_LOOKUP[key]


def n_select_sweep(
    df: pd.DataFrame,
    folds: Sequence[Fold],
    feature_cols: Sequence[str],
    target_col: str,
    metric_fn,
    wrap,                                   # signal -> allocation, from hull_probe
    rankings: Dict[Tuple[int, int], List[str]],
    levels: Sequence[Optional[int]] = (None, 300, 120, 60, 30, 10),
    seeds: Sequence[int] = (0, 1, 2),
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Same folds, same seeds, only the selection width changes.

    Read the paired table, not the summary table: the level-to-level
    difference is what says whether the extra columns paid, and its
    detectable_at column says whether the fold count can resolve a
    difference that size at all.
    """
    register_folds(folds)
    runs: Dict[str, pd.DataFrame] = {}
    for lv in levels:
        tag = f"k={'all' if lv is None else lv}"
        print(f"  {tag}")
        sig = make_tree_signal(feature_cols, target_col, lv, rankings)
        runs[tag] = run_wfo(df, folds, wrap(sig), metric_fn, seeds=seeds, tag=tag)

    summary = pd.DataFrame({k: summarize(v) for k, v in runs.items()}).T
    tags = list(runs)
    paired = pd.DataFrame(
        {f"{b} - {a}": paired_compare(runs[a], runs[b]) for a, b in zip(tags[:-1], tags[1:])}
    ).T
    return summary, paired, runs


def save_runs(runs: Dict[str, pd.DataFrame], path: str) -> None:
    """
    Per-fold-per-seed scores, written so a run from this script can be paired
    against one from hull_probe.py without refitting either.

    The first pass only kept summaries, which made the obvious comparison --
    the k=all tree at 1.084 against the linear+tree pipeline at 0.762, both
    under the same mapping -- impossible to test without rerunning both.
    """
    pd.concat(runs.values(), ignore_index=True).to_csv(path, index=False)


# ============================================================
# Driver
# ============================================================

def main() -> None:
    from hull_probe import (
        RESULT_DIR, TARGET_COL, EMBARGO, HOLDOUT_FRAC, HOLDOUT_GAP,
        START_AT, TEST_WINDOW, TRAIN_WINDOW,
        feature_cols, load, mapper_original, metric_fn,
    )
    from wfo import make_folds

    os.makedirs(RESULT_DIR, exist_ok=True)
    df = load()
    feats = feature_cols(df)
    n = len(df)
    dev_end = n - int(round(n * HOLDOUT_FRAC)) - HOLDOUT_GAP
    folds = make_folds(dev_end, TRAIN_WINDOW, TEST_WINDOW, embargo=EMBARGO, start_at=START_AT)
    print(f"{len(folds)} folds, {len(feats)} features")

    # The sweep is about feature width, so the mapping is held fixed at the
    # original rule -- the same one hull_probe's model stage uses, which keeps
    # the two scripts' numbers on one scale.
    orig = mapper_original(df)

    def mapper(signal_fn):
        def fit_predict(d, tr, te, seed):
            return orig(d, tr, te, None, signal_fn(d, tr, te, seed))
        return fit_predict

    print("\nranking features per fold")
    ranks = rankings_by_fold(df, folds, feats, TARGET_COL, seeds=(0, 1, 2))

    print("\n[stability]")
    stab = selection_stability(ranks, n_total=len(feats))
    stab.to_csv(os.path.join(RESULT_DIR, "selection_stability.csv"))
    print(stab.round(4).to_string())

    core = core_features(ranks, k=60, min_frac=0.5)
    core.to_csv(os.path.join(RESULT_DIR, "core_features.csv"))
    print(f"\n{len(core)} features reach the top-60 in at least half the folds")
    print(core.head(25).round(3).to_string())

    print("\n[sweep]")
    summary, paired, runs = n_select_sweep(
        df, folds, feats, TARGET_COL, metric_fn, mapper, ranks
    )
    save_runs(runs, os.path.join(RESULT_DIR, "raw_nselect.csv"))
    summary.to_csv(os.path.join(RESULT_DIR, "nselect_summary.csv"))
    paired.to_csv(os.path.join(RESULT_DIR, "nselect_paired.csv"))
    print(summary[["n_folds", "mean", "std", "stability", "q10"]].round(3).to_string())
    print()
    print(paired[["mean_diff", "t_stat", "win_rate", "detectable_at"]].round(3).to_string())


if __name__ == "__main__":
    main()
