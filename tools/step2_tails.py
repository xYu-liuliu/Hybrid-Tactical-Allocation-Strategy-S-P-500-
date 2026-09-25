"""
step2_tails.py (v3) -- does the tail separation survive causal ranking?

v1 -> v2 (kept, this is why the file looks the way it does):

  FIXED TWO-SAMPLE t. hi_vs_mid_t and lo_vs_mid_t in v1 were computed as the
  mean t of concatenate([v, -ref]), which only equals the difference in means
  when the two groups are the same size.

  BAGGED SIGNAL. Across seeds the v1 spread_t was 1.34 / 2.06 / 0.64 -- the
  seed moved the verdict further than the effect did.

  MONOTONICITY AND BIN OCCUPANCY. A positive spread does not imply an ordered
  signal, and (arr < v).mean() saturates at 0 and 1 whenever the signal sets a
  new extreme, so a trending signal piles up in the end bins.

v2 -> v3. Three changes, all forced by what the v2 output turned out to mean:

  THE POOLED STATISTICS ARE DEMOTED. tail=0.10 should put 8.4 of a fold's 84
  days in each tail. Under p_trail it put between 0 and 51 in the low tail
  (sd 12.5), and the per-fold count correlates +0.68 to +0.72 with the fold's
  mean signal level. The mask is a drift detector, not a decile, so the pooled
  spread_t / hi_vs_mid_t / lo_vs_mid_t are dominated by whichever folds the
  signal happened to drift through. They disagree with the cross-fold test on
  the same data: v2's lo_vs_mid_t was -1.86 pooled and -0.25 across folds.
  spread_t also applies HAC to concatenate([hi, -lo]), whose autocorrelation
  structure is an artefact of the concatenation. Folds are the independent
  replicates; everything primary is now computed across them, with the seeds
  averaged INSIDE each fold first.

  THE SEED-RANGE CRITERION IS REPLACED. v2's override said that a spread_t
  range which bagging does not shrink means the dispersion is in the data. On
  the rebuilt features the range went 1.36 -> 1.07 (-21%) and the override
  fired -- but the bagged signals across seeds correlate 0.9898 and their
  day-level dispersion is 8.7% of the signal's own scale, identically to the
  run where the override did not fire. Bagging worked both times. The range of
  three numbers was never a usable statistic. Seed agreement is now measured on
  the signal itself, which is what the criterion was always trying to ask.

  THE RANKING IS SPLIT. p_trail confounds two questions: whether the signal's
  LEVEL predicts returns (drift) and whether its ORDERING within a period does.
  p_window answers the second but needs the whole window. p_trail_dm ranks the
  signal net of its own trailing mean, and p_drift ranks that trailing mean --
  causal, and between them they separate the two. Pre-registered readings:
    * if drift is the confound and the ranking is real, p_trail_dm has a much
      higher usable-fold fraction than p_trail and a comparable t, and the
      look-ahead cost against p_window collapses;
    * if the edge IS the drift, p_drift carries the spread and p_trail_dm does
      not;
    * if neither carries it, the v2 result was the mask selecting regimes.

Nothing else moved. No tail width was added, no threshold was tuned on a
result. The thresholds below are for v3's primary statistic and are fixed
before this version has been run once.

    python step2_tails.py                # fit and evaluate
    python step2_tails.py --from-cache   # re-read the last run's signal files
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from wfo import Fold, make_folds
from diagnostics import hac_ols
from hull_probe import (
    EMBARGO, HOLDOUT_FRAC, HOLDOUT_GAP, INNER_FRAC, RANK_MAP, RESULT_DIR,
    SEEDS, START_AT, TARGET_COL, TEST_WINDOW, TRAIN_WINDOW, TREE,
    CachedSignal, feature_cols, load, trailing_percentile,
)

TAILS = (0.10, 0.20, 0.30)
RANKINGS = ("p_window", "p_trail", "p_trail_mid", "p_trail_dm", "p_drift")
N_PERM = 2000
PERM_SEED = 7
BAG_SIZES = (1, 15)

# Window for the trailing mean that p_trail_dm removes and p_drift ranks.
# Fixed at one quarter on purpose, not swept: shorter than the 252-day ranking
# window so it removes drift rather than the ranking itself, longer than the
# 84-day test window so it is not just re-centring inside the fold.
MU_WINDOW = 63

# Folds needing at least this many days in EACH tail to enter the cross-fold
# test. 5 of 84 is already permissive; the point of reporting the fraction that
# qualifies is that under p_trail it was only 47 of 93 (fold, seed) cells.
MIN_TAIL_N = 5

# ---- pre-registered thresholds for v3's primary statistic ----
# The primary statistic changed scale (pooled HAC t -> cross-fold t), so v2's
# numbers do not carry over. These are set on principle: |t| > 2 is the
# conventional level, 2.5 buys back the 4 rankings x 3 tails being looked at,
# and usable_frac is new because a mask that qualifies in a third of folds is
# not measuring what the test claims to measure.
CONTINUE_SPREAD_T = 2.5
CONTINUE_USABLE_FRAC = 0.60
CONTINUE_PERM_P = 0.05
STOP_SPREAD_T = 1.0
STOP_USABLE_FRAC = 0.35


# ============================================================
# Signal
# ============================================================

def make_bagged_tree_signal(
    feature_cols_: Sequence[str], n_select: Optional[int] = None, n_bag: int = 15
):
    """
    Mean prediction over n_bag LightGBM fits differing only in random_state.

    The calibration predictions are bagged the same way. They have to be:
    trailing_percentile compares test predictions against calibration history,
    so if one side is an average of 15 models and the other a single fit, the
    two distributions differ in spread and every percentile is biased toward
    the middle.

    Data slices are hoisted out of the bag loop -- re-slicing a 1150-column
    frame 15 times per fold costs more than the fits do.
    """
    cols = list(feature_cols_)

    def f(df, train_idx, test_idx, seed):
        y = df[TARGET_COL].to_numpy(float)
        if n_select is None:
            sel = cols
        else:
            from hull_probe import _select_in_window
            sel = _select_in_window(df.iloc[train_idx][cols], y[train_idx], n_select, seed)

        cut = int(len(train_idx) * INNER_FRAC)
        inner_tr, inner_cal = train_idx[:cut], train_idx[cut:]
        ok_i, ok_t = np.isfinite(y[inner_tr]), np.isfinite(y[train_idx])

        X_inner, y_inner = df.iloc[inner_tr[ok_i]][sel], y[inner_tr[ok_i]]
        X_cal = df.iloc[inner_cal][sel]
        X_full, y_full = df.iloc[train_idx[ok_t]][sel], y[train_idx[ok_t]]
        X_test = df.iloc[test_idx][sel]

        cals, preds = [], []
        for b in range(n_bag):
            rs = int(seed) * 1000 + b
            m_cal = LGBMRegressor(**{**TREE, "random_state": rs})
            m_cal.fit(X_inner, y_inner)
            cals.append(m_cal.predict(X_cal))

            m = LGBMRegressor(**{**TREE, "random_state": rs})
            m.fit(X_full, y_full)
            preds.append(m.predict(X_test))

        return np.mean(cals, axis=0), np.mean(preds, axis=0)

    return f


def cal_positions(train_idx: np.ndarray) -> np.ndarray:
    """Rows the calibration predictions belong to -- the same slice the signal uses."""
    return train_idx[int(len(train_idx) * INNER_FRAC):]


# ============================================================
# Percentiles
# ============================================================

def trailing_percentile_mid(
    s_cal: np.ndarray, s_test: np.ndarray, window: int = 252, min_hist: int = 60
) -> np.ndarray:
    """
    Midrank version of trailing_percentile, bounded strictly inside (0, 1).

    (arr < v).mean() hits 0 and 1 exactly whenever v is the extreme of its
    trailing window, which a trending signal does often, inflating the end
    bins. ((below + 0.5*ties) + 0.5) / (n + 1) cannot reach either endpoint.
    """
    hist = list(np.asarray(s_cal, float)[-window:])
    out = np.full(len(s_test), 0.5)
    for j, v in enumerate(np.asarray(s_test, float)):
        arr = np.asarray(hist[-window:], float)
        arr = arr[np.isfinite(arr)]
        if len(arr) >= min_hist and np.isfinite(v):
            below = float((arr < v).sum()) + 0.5 * float((arr == v).sum())
            out[j] = (below + 0.5) / (len(arr) + 1.0)
        hist.append(v)
    return out


def _level_and_deviation(
    s_cal: np.ndarray, s_test: np.ndarray, mu_window: int = MU_WINDOW
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Split the signal into its trailing level and its deviation from that level.

    mu_t is the mean of the mu_window values STRICTLY before t (the rolling
    mean is shifted by one), so nothing from day t enters its own level.
    Calibration and test are concatenated first because the level at the start
    of the test block has to come from the calibration tail -- without it the
    first mu_window test days of every fold would be undefined, which at
    84-day folds is most of the fold.

    Returns (level, deviation, n_cal) over the concatenated series.
    """
    cal = np.asarray(s_cal, float)
    test = np.asarray(s_test, float)
    s = pd.Series(np.concatenate([cal, test]))
    level = s.rolling(mu_window, min_periods=max(10, mu_window // 3)).mean().shift(1)
    return level.to_numpy(), (s - level).to_numpy(), len(cal)


def trailing_percentile_detrended(
    s_cal: np.ndarray, s_test: np.ndarray,
    window: int = 252, min_hist: int = 60, mu_window: int = MU_WINDOW,
) -> np.ndarray:
    """
    Percentile of the signal NET of its own trailing level.

    This is the ordering half of p_trail. If the v2 tail result was the mask
    tracking drift, this ranking should produce balanced masks -- a fold's
    count of extreme days stops depending on where the signal's level happened
    to sit -- and the usable-fold fraction should rise toward p_window's 1.0.
    """
    _, dev, n_cal = _level_and_deviation(s_cal, s_test, mu_window)
    return trailing_percentile(dev[:n_cal], dev[n_cal:], window, min_hist)


def trailing_percentile_drift(
    s_cal: np.ndarray, s_test: np.ndarray,
    window: int = 252, min_hist: int = 60, mu_window: int = MU_WINDOW,
) -> np.ndarray:
    """
    Percentile of the signal's own trailing level among that level's history.

    This is the drift half. p_trail ranks (level + deviation); p_drift ranks
    the level and p_trail_dm ranks the deviation, so between them they say
    which part of p_trail was carrying the tail spread.
    """
    level, _, n_cal = _level_and_deviation(s_cal, s_test, mu_window)
    return trailing_percentile(level[:n_cal], level[n_cal:], window, min_hist)


def collect(
    df: pd.DataFrame, folds: Sequence[Fold], signal_fn, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the signal over every fold and stack the test-day rows.

    Also returns the calibration predictions, which v2 discarded. They are the
    seed of every trailing-window statistic, so without them no new causal
    ranking can be computed from a cached run and the whole thing has to be
    refit. Caching them makes the next ranking experiment free.
    """
    fwd = df["forward_returns"].to_numpy(float)
    rf = df["risk_free_rate"].to_numpy(float)
    w, mh = RANK_MAP["window"], RANK_MAP["min_hist"]
    parts, cal_parts = [], []

    for f in folds:
        s_cal, s_te = signal_fn(df, f.train, f.test, seed)
        s = pd.Series(s_te)
        parts.append(pd.DataFrame({
            "fold": f.k,
            "pos": f.test,
            "s_raw": s_te,
            "p_trail": trailing_percentile(s_cal, s_te, w, mh),
            "p_trail_mid": trailing_percentile_mid(s_cal, s_te, w, mh),
            "p_trail_dm": trailing_percentile_detrended(s_cal, s_te, w, mh),
            "p_drift": trailing_percentile_drift(s_cal, s_te, w, mh),
            # rank over the whole window: what diagnostics.decile_table did
            "p_window": (s.rank(method="first") - 0.5).to_numpy() / len(s),
            "x": fwd[f.test] - rf[f.test],
        }))
        cal_parts.append(pd.DataFrame({
            "fold": f.k, "pos": cal_positions(f.train), "s_cal": np.asarray(s_cal, float),
        }))

    return pd.concat(parts, ignore_index=True), pd.concat(cal_parts, ignore_index=True)


# ============================================================
# Mask health
# ============================================================

def _masks(p: np.ndarray, tail: float) -> Tuple[np.ndarray, np.ndarray]:
    return p <= tail, p >= 1.0 - tail


def mask_health(d: pd.DataFrame, pcol: str, tail: float) -> Dict[str, float]:
    """
    Is this ranking producing deciles, or is it tracking the signal's level?

    A balanced mask puts tail*84 days in each end of every fold. drift_corr is
    the correlation between a fold's mean signal and its (n_hi - n_lo): near
    zero means the mask is selecting days, near one means it is selecting
    regimes and the pooled statistics are weighted by where the signal drifted.
    """
    g = d.groupby("fold")
    n_lo = g.apply(lambda x: int((x[pcol] <= tail).sum()), include_groups=False)
    n_hi = g.apply(lambda x: int((x[pcol] >= 1 - tail).sum()), include_groups=False)
    lvl = g.apply(lambda x: float(x["s_raw"].mean()), include_groups=False)
    both = (n_lo >= MIN_TAIL_N) & (n_hi >= MIN_TAIL_N)
    expected = tail * float(d.groupby("fold").size().mean())
    dc = np.nan
    if lvl.std() > 0 and (n_hi - n_lo).std() > 0:
        dc = float(np.corrcoef(lvl, n_hi - n_lo)[0, 1])
    return {
        "expected_per_tail": expected,
        "n_lo_min": int(n_lo.min()), "n_lo_max": int(n_lo.max()), "n_lo_sd": float(n_lo.std()),
        "n_hi_min": int(n_hi.min()), "n_hi_max": int(n_hi.max()), "n_hi_sd": float(n_hi.std()),
        "usable_folds": int(both.sum()), "usable_frac": float(both.mean()),
        "drift_corr": dc,
    }


# ============================================================
# Primary: cross-fold, seeds averaged inside the fold
# ============================================================

def fold_table(
    per_seed: Dict[int, pd.DataFrame], pcol: str, tail: float, min_n: int = MIN_TAIL_N
) -> pd.DataFrame:
    """
    One row per fold, seeds averaged.

    Averaging the seeds inside the fold before testing across folds is the
    paired design: the seeds are three looks at one fold, not three
    observations. A (fold, seed) cell is skipped when either tail has fewer
    than min_n days; a fold enters only if at least one seed qualified.
    """
    acc: Dict[int, Dict[str, List[float]]] = {}
    for s, d in per_seed.items():
        lo_m, hi_m = _masks(d[pcol].to_numpy(), tail)
        lo_m = pd.Series(lo_m, index=d.index)
        hi_m = pd.Series(hi_m, index=d.index)
        for k, g in d.groupby("fold"):
            gl, gh = lo_m[g.index], hi_m[g.index]
            if gl.sum() < min_n or gh.sum() < min_n:
                continue
            r = acc.setdefault(int(k), {"lo": [], "mid": [], "hi": [], "n_seed": []})
            r["lo"].append(float(g.loc[gl, "x"].mean()))
            r["hi"].append(float(g.loc[gh, "x"].mean()))
            r["mid"].append(float(g.loc[~gl & ~gh, "x"].mean()))
            r["n_seed"].append(1.0)

    rows = [{
        "fold": k, "n_seeds": int(sum(v["n_seed"])),
        "lo": float(np.mean(v["lo"])), "mid": float(np.mean(v["mid"])), "hi": float(np.mean(v["hi"])),
    } for k, v in sorted(acc.items())]

    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["lo_minus_mid"] = t["lo"] - t["mid"]
    t["hi_minus_mid"] = t["hi"] - t["mid"]
    t["spread"] = t["hi"] - t["lo"]
    return t


def cross_fold(v) -> Dict[str, float]:
    """Folds are the independent replicates. Plain t, no HAC: no time ordering."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 3:
        return {"n": n, "mean_ann": np.nan, "t": np.nan, "pos_frac": np.nan}
    sd = v.std(ddof=1)
    return {
        "n": n,
        "mean_ann": float(v.mean() * 252),
        "t": float(v.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan,
        "pos_frac": float((v > 0).mean()),
    }


# ============================================================
# Pooled -- kept for continuity with v2, no longer primary
# ============================================================

def _mean_hac_t(v: np.ndarray) -> float:
    if len(v) < 20:
        return np.nan
    _, _, t, _ = hac_ols(v, np.ones((len(v), 1)))
    return float(t[0])


def _two_sample_hac_t(a: np.ndarray, b: np.ndarray) -> float:
    """HAC t on mean(a) - mean(b) via a group dummy, valid for unequal sizes."""
    if len(a) < 20 or len(b) < 20:
        return np.nan
    y = np.concatenate([a, b])
    X = np.column_stack([np.ones(len(y)), np.r_[np.ones(len(a)), np.zeros(len(b))]])
    _, _, t, _ = hac_ols(y, X)
    return float(t[1])


def pooled_stats(d: pd.DataFrame, pcol: str, tail: float) -> Dict[str, float]:
    """
    v2's statistics. Reported, not used for the decision.

    spread_t applies HAC to concatenate([hi, -lo]); that series' autocorrelation
    is an artefact of the concatenation, and the pooled means weight days, so
    folds where the mask fired on half the window dominate. Both are why v3
    decides on the cross-fold numbers instead.
    """
    p, x = d[pcol].to_numpy(), d["x"].to_numpy()
    lo_m, hi_m = _masks(p, tail)
    mid_m = ~lo_m & ~hi_m
    lo, hi, mid = x[lo_m], x[hi_m], x[mid_m]
    n = min(len(lo), len(hi))
    return {
        "n_lo": int(len(lo)), "n_mid": int(len(mid)), "n_hi": int(len(hi)),
        "lo_ann": float(lo.mean() * 252) if len(lo) else np.nan,
        "mid_ann": float(mid.mean() * 252) if len(mid) else np.nan,
        "hi_ann": float(hi.mean() * 252) if len(hi) else np.nan,
        "spread_ann": float((hi.mean() - lo.mean()) * 252) if n else np.nan,
        "spread_t": _mean_hac_t(np.concatenate([hi[:n], -lo[:n]])) if n >= 20 else np.nan,
        "hi_vs_mid_t": _two_sample_hac_t(hi, mid),
        "lo_vs_mid_t": _two_sample_hac_t(lo, mid),
    }


def decile_frame(d: pd.DataFrame, pcol: str, n_bins: int = 10) -> pd.DataFrame:
    g = d.assign(bin=np.clip((d[pcol] * n_bins).astype(int), 0, n_bins - 1))
    rows = []
    for b, gg in g.groupby("bin"):
        x = gg["x"].to_numpy()
        rows.append({"bin": int(b), "n": len(x), "x_ann": x.mean() * 252,
                     "x_t": _mean_hac_t(x), "hit": float((x > 0).mean())})
    return pd.DataFrame(rows).set_index("bin").sort_index()


def spearman_of(dec: pd.DataFrame) -> float:
    """
    Rank correlation between bin index and bin mean return.

    Read it against its own null: with 10 bins the standard error is about
    1/sqrt(9) = 0.333, so v2's 0.483 was 1.45 SE and v3's 0.244 was 0.73 SE.
    Neither was distinguishable from zero; the quantity has never had the
    resolution to support the "ends landed right, middle unordered" reading.
    """
    return float(pd.Series(dec.index).corr(
        dec["x_ann"].reset_index(drop=True), method="spearman"))


def permutation_p(
    d: pd.DataFrame, pcol: str, tail: float, n_perm: int = N_PERM, seed: int = PERM_SEED
) -> Dict[str, float]:
    """
    Circularly shift x inside each fold and recompute the pooled spread.

    Bins stay put; only returns rotate. That keeps each fold's return
    autocorrelation and marginal distribution, and -- usefully here -- it keeps
    the mask imbalance, so the null is generated under the same lopsided masks
    as the observation.
    """
    rng = np.random.default_rng(seed)
    groups = [(g[pcol].to_numpy(), g["x"].to_numpy()) for _, g in d.groupby("fold")]

    def spread(shifted: List[np.ndarray]) -> float:
        lo_all, hi_all = [], []
        for (p, _), xs in zip(groups, shifted):
            lo_m, hi_m = _masks(p, tail)
            lo_all.append(xs[lo_m]); hi_all.append(xs[hi_m])
        lo_all, hi_all = np.concatenate(lo_all), np.concatenate(hi_all)
        if len(lo_all) == 0 or len(hi_all) == 0:
            return np.nan
        return float(hi_all.mean() - lo_all.mean())

    obs = spread([x for _, x in groups])
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = spread([np.roll(x, int(rng.integers(1, len(x)))) for _, x in groups])

    sd = np.nanstd(null, ddof=1)
    return {
        "observed": obs,
        "null_sd": float(sd),
        "z_vs_null": float((obs - np.nanmean(null)) / sd) if sd > 0 else np.nan,
        "p_two_sided": float(np.nanmean(np.abs(null - np.nanmean(null)) >= abs(obs - np.nanmean(null)))),
        "p_one_sided": float(np.nanmean(null >= obs)),
    }


# ============================================================
# Seed agreement -- replaces the spread_t range criterion
# ============================================================

def seed_agreement(per_seed: Dict[int, pd.DataFrame]) -> Dict[str, float]:
    """
    Do the seeds agree on the SIGNAL, which is what the v2 criterion meant.

    v2 asked whether bagging shrank the range of spread_t across three seeds.
    The range of three numbers has no resolution, and spread_t turned out to be
    so fragile a functional of the signal that seeds correlating at 0.99 gave a
    range above 1.0. Measured on the signal instead, the question is answerable:
    pair_corr near 1 and day_dispersion near 0 mean the model noise is gone and
    any remaining disagreement in a downstream statistic is that statistic's.
    """
    seeds = sorted(per_seed)
    M = np.vstack([per_seed[s]["s_raw"].to_numpy(float) for s in seeds])
    cs = [float(np.corrcoef(M[i], M[j])[0, 1])
          for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
    centre = M.mean(axis=0)
    return {
        "pair_corr": float(np.mean(cs)) if cs else np.nan,
        "pair_corr_min": float(np.min(cs)) if cs else np.nan,
        "day_dispersion": float(M.std(axis=0, ddof=1).mean() / centre.std())
        if centre.std() > 0 else np.nan,
    }


# ============================================================
# IO
# ============================================================

def sig_path(n_bag: int, seed: int) -> str:
    return os.path.join(RESULT_DIR, f"step2_sig_bag{n_bag}_seed{seed}.csv")


def cal_path(n_bag: int, seed: int) -> str:
    return os.path.join(RESULT_DIR, f"step2_cal_bag{n_bag}_seed{seed}.csv")


def load_cached() -> Dict[int, Dict[int, pd.DataFrame]]:
    """Re-read the last run's signal files so the verdict can be recomputed free."""
    out: Dict[int, Dict[int, pd.DataFrame]] = {}
    for n_bag in BAG_SIZES:
        for seed in SEEDS:
            p = sig_path(n_bag, seed)
            if os.path.exists(p):
                out.setdefault(n_bag, {})[seed] = pd.read_csv(p)
    if not out:
        raise FileNotFoundError(f"no cached signal files under {RESULT_DIR}")
    return out


# ============================================================
# main
# ============================================================

def main() -> None:
    from_cache = "--from-cache" in sys.argv
    os.makedirs(RESULT_DIR, exist_ok=True)

    if from_cache:
        sigs = load_cached()
        n_rows = n_feats = n_folds = -1
        present = [r for r in RANKINGS
                   if r in next(iter(next(iter(sigs.values())).values())).columns]
        print(f"[--from-cache] {RESULT_DIR}")
        print(f"  bags {sorted(sigs)}  rankings present: {present}")
        if set(present) != set(RANKINGS):
            print(f"  missing {sorted(set(RANKINGS) - set(present))} -- "
                  f"they need s_cal, which a pre-v3 run did not save. Refit to get them.")
    else:
        df = load()
        feats = feature_cols(df)
        n_rows, n_feats = len(df), len(feats)
        n = len(df)
        dev_end = n - int(round(n * HOLDOUT_FRAC)) - HOLDOUT_GAP
        folds = make_folds(dev_end, TRAIN_WINDOW, TEST_WINDOW, embargo=EMBARGO, start_at=START_AT)
        n_folds = len(folds)
        print(f"rows {n}, features {n_feats}, folds {n_folds}, seeds {list(SEEDS)}")
        print(f"bag sizes {list(BAG_SIZES)} -> "
              f"{n_folds * len(SEEDS) * sum(BAG_SIZES) * 2} LightGBM fits")
        present = list(RANKINGS)

        sigs = {}
        for n_bag in BAG_SIZES:
            sig = CachedSignal(make_bagged_tree_signal(feats, n_bag=n_bag))
            for seed in SEEDS:
                t0 = time.time()
                d, cal = collect(df, folds, sig, seed)
                d.to_csv(sig_path(n_bag, seed), index=False)
                cal.to_csv(cal_path(n_bag, seed), index=False)
                sigs.setdefault(n_bag, {})[seed] = d
                print(f"  bag={n_bag} seed={seed}  {time.time() - t0:.0f}s")

    big = max(sigs)

    # ---- primary: cross-fold, seeds averaged inside the fold ----
    prim_rows, health_rows, pooled_rows = [], [], []
    deciles: Dict[str, pd.DataFrame] = {}

    for n_bag, per_seed in sorted(sigs.items()):
        for pcol in present:
            for tail in TAILS:
                h = mask_health(per_seed[SEEDS[0]], pcol, tail)
                h.update({"n_bag": n_bag, "ranking": pcol, "tail": tail})
                health_rows.append(h)

                t = fold_table(per_seed, pcol, tail)
                rec = {"n_bag": n_bag, "ranking": pcol, "tail": tail,
                       "usable_folds": int(len(t)), "usable_frac": h["usable_frac"],
                       "drift_corr": h["drift_corr"]}
                for col, nm in (("spread", "spread"), ("lo_minus_mid", "lo_mid"),
                                ("hi_minus_mid", "hi_mid")):
                    r = cross_fold(t[col]) if len(t) else cross_fold([])
                    rec[f"{nm}_ann"] = r["mean_ann"]
                    rec[f"{nm}_t"] = r["t"]
                    rec[f"{nm}_pos"] = r["pos_frac"]
                if tail == TAILS[0]:
                    rec.update(permutation_p(per_seed[SEEDS[0]], pcol, tail))
                prim_rows.append(rec)

                for seed in SEEDS:
                    ps = pooled_stats(per_seed[seed], pcol, tail)
                    ps.update({"n_bag": n_bag, "seed": seed, "ranking": pcol, "tail": tail})
                    pooled_rows.append(ps)

            for seed in SEEDS:
                dec = decile_frame(per_seed[seed], pcol)
                deciles[f"bag{n_bag}_seed{seed}_{pcol}"] = dec

    prim = pd.DataFrame(prim_rows)
    health = pd.DataFrame(health_rows)
    pooled = pd.DataFrame(pooled_rows)
    prim.to_csv(os.path.join(RESULT_DIR, "step2_crossfold.csv"), index=False)
    health.to_csv(os.path.join(RESULT_DIR, "step2_mask_health.csv"), index=False)
    pooled.to_csv(os.path.join(RESULT_DIR, "step2_tails.csv"), index=False)
    pd.concat(deciles, names=["run"]).to_csv(os.path.join(RESULT_DIR, "step2_deciles.csv"))

    # ---- report ----
    print("\n=== PRIMARY: cross-fold, seeds averaged inside the fold (bag=%d) ===" % big)
    cols = ["ranking", "tail", "usable_folds", "usable_frac", "drift_corr",
            "spread_ann", "spread_t", "spread_pos", "lo_mid_t", "hi_mid_t"]
    print(prim[prim["n_bag"] == big][cols].round(3).to_string(index=False))

    print("\n=== mask health (tail=%.2f, bag=%d): is it a decile or a drift detector ===" % (TAILS[0], big))
    hc = ["ranking", "expected_per_tail", "n_lo_min", "n_lo_max", "n_lo_sd",
          "n_hi_min", "n_hi_max", "usable_frac", "drift_corr"]
    print(health[(health["n_bag"] == big) & (health["tail"] == TAILS[0])][hc].round(3).to_string(index=False))

    print("\n=== seed agreement on the SIGNAL (replaces the spread_t range test) ===")
    agree = {nb: seed_agreement(ps) for nb, ps in sorted(sigs.items())}
    for nb, a in agree.items():
        print(f"  bag={nb:<3d} pair_corr={a['pair_corr']:.4f} (min {a['pair_corr_min']:.4f})  "
              f"day_dispersion={a['day_dispersion']:.3f}")

    print("\n=== POOLED (v2's statistics, reported only) ===")
    pc = ["ranking", "seed", "lo_ann", "mid_ann", "hi_ann", "spread_t", "hi_vs_mid_t", "lo_vs_mid_t"]
    print(pooled[(pooled["n_bag"] == big) & (pooled["tail"] == TAILS[0])][pc].round(3).to_string(index=False))

    # ---- verdict ----
    def pick(rk: str) -> pd.Series:
        m = prim[(prim["n_bag"] == big) & (prim["ranking"] == rk) & (prim["tail"] == TAILS[0])]
        return m.iloc[0] if len(m) else pd.Series(dtype=float)

    verdict: Dict[str, object] = {
        "version": 3,
        "n_rows": n_rows, "n_features": n_feats, "n_folds": n_folds,
        "bag_sizes": list(BAG_SIZES), "seeds": list(SEEDS),
        "primary_statistic": "cross-fold t of (high tail - low tail), seeds averaged in fold",
        "mu_window": MU_WINDOW,
        "seed_agreement": {str(k): v for k, v in agree.items()},
    }
    for rk in present:
        r = pick(rk)
        if r.empty:
            continue
        verdict[rk] = {k: (None if pd.isna(r.get(k)) else float(r.get(k)))
                       for k in ("usable_folds", "usable_frac", "drift_corr",
                                 "spread_ann", "spread_t", "spread_pos",
                                 "lo_mid_ann", "lo_mid_t", "hi_mid_ann", "hi_mid_t",
                                 "p_two_sided")}

    tr, dm, dr, wn = pick("p_trail"), pick("p_trail_dm"), pick("p_drift"), pick("p_window")
    if not dm.empty and not wn.empty:
        verdict["look_ahead_cost_dm"] = float(wn["spread_t"] - dm["spread_t"])
    if not tr.empty and not wn.empty:
        verdict["look_ahead_cost_trail"] = float(wn["spread_t"] - tr["spread_t"])

    def decide(r: pd.Series) -> str:
        if r.empty or not np.isfinite(r.get("spread_t", np.nan)):
            return "no result"
        if r["spread_t"] < STOP_SPREAD_T or r["usable_frac"] < STOP_USABLE_FRAC:
            return "STOP"
        if (r["spread_t"] > CONTINUE_SPREAD_T and r["usable_frac"] > CONTINUE_USABLE_FRAC
                and (not np.isfinite(r.get("p_two_sided", np.nan))
                     or r["p_two_sided"] < CONTINUE_PERM_P)):
            return "CONTINUE"
        return "in between"

    verdict["decision"] = {rk: decide(pick(rk)) for rk in present}
    # --from-cache must not clobber the metadata a real run recorded (it has no
    # row/fold/feature counts of its own), so it writes beside it.
    vname = "step2_verdict_from_cache.json" if from_cache else "step2_verdict.json"
    with open(os.path.join(RESULT_DIR, vname), "w") as f:
        json.dump(verdict, f, indent=2)

    print("\n=== read-out ===")
    print(json.dumps(verdict["decision"], indent=2))
    print(f"""
Thresholds, fixed before v3 ran (cross-fold t at tail={TAILS[0]}):
  CONTINUE  spread_t > {CONTINUE_SPREAD_T}, usable_frac > {CONTINUE_USABLE_FRAC}, perm p < {CONTINUE_PERM_P}
  STOP      spread_t < {STOP_SPREAD_T} or usable_frac < {STOP_USABLE_FRAC}

What the three rankings answer together:

  p_trail_dm usable_frac >> p_trail usable_frac, with comparable spread_t
      -> the mask imbalance WAS the drift, and the ordering is real. Then
         look_ahead_cost_dm is the honest look-ahead cost, not a mixture of
         look-ahead and mask balance.

  p_drift carries the spread and p_trail_dm does not
      -> the edge is the signal's LEVEL, i.e. a slow regime call. The
         effective sample is the number of drift episodes, not the number of
         days, and neither spread_t nor the fold count is the binding
         constraint -- the number of regimes is.

  neither carries it
      -> v2's tail result was the mask selecting regimes that happened to line
         up, and no amount of extra folds recovers it.

Read usable_frac before any t-statistic. A ranking that qualifies in a third
of folds is not reporting a decile, and its pooled numbers are weighted by
wherever the signal drifted.
""")


if __name__ == "__main__":
    main()
