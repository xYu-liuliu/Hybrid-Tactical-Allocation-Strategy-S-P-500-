"""
wfo.py -- Walk-forward out-of-sample evaluation harness.

Three rules this encodes:
  1. every experiment is scored by the competition metric on non-overlapping
     OOS windows -- never by MSE, never by the public leaderboard;
  2. variants are compared PAIRED (same folds, same seeds), because
     fold-level variance dominates and cancels in the difference;
  3. the early-stopping split lives inside the training window only, and is
     chronological, never random.

Plug your own model in through `fit_predict` and the official competition
metric in through `metric_fn`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as _st
except ImportError:  # t-quantiles fall back to a normal approximation
    _st = None


# ============================================================
# 1. Fold geometry
# ============================================================

@dataclass(frozen=True)
class Fold:
    k: int
    train: np.ndarray  # positional indices into the frame
    test: np.ndarray


def make_folds(
    n: int,
    train_window: int = 756,
    test_window: int = 84,
    embargo: int = 10,
    purge: int = 1,
    expanding: bool = False,
    min_train: int = 504,
    start_at: int = 0,
) -> List[Fold]:
    """
    Non-overlapping OOS windows marching forward through the sample.

    purge     rows dropped from the END of train: the label at row t
              (forward_returns) resolves at t+1, which sits in the gap.
    embargo   extra rows dropped so the first test row's rolling features
              are not near-duplicates of the last training rows. This is not
              a leakage fix -- it is a correlation fix. Re-run with a much
              larger value as a sensitivity check; a big score drop means
              part of your edge came from boundary correlation.
    expanding True trains on everything up to the gap; False uses a rolling
              window, which lets you validate over the full history without
              old regimes contaminating recent models.
    start_at  skip the first N rows (e.g. the feature warm-up region).
    """
    gap = purge + embargo
    folds: List[Fold] = []
    start = max(start_at, 0) + train_window + gap
    k = 0
    while start + test_window <= n:
        tr_end = start - gap
        tr_start = max(start_at, 0) if expanding else max(start_at, tr_end - train_window)
        if tr_end - tr_start >= min_train:
            folds.append(
                Fold(k, np.arange(tr_start, tr_end), np.arange(start, start + test_window))
            )
            k += 1
        start += test_window
    return folds


def inner_split(
    train_idx: np.ndarray, val_frac: float = 0.10, purge: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Chronological early-stopping split inside one training window."""
    n = len(train_idx)
    n_val = max(1, int(round(n * val_frac)))
    cut = n - n_val
    return train_idx[: max(1, cut - purge)], train_idx[cut:]


# ============================================================
# 2. Metric slot
# ============================================================

def plain_sharpe(w: np.ndarray, fwd: np.ndarray, rf: np.ndarray, ann: int = 252) -> float:
    """
    SMOKE TEST ONLY -- this is NOT the competition metric.

    Replace with the official implementation linked from the competition's
    Evaluation page. The volatility and underperformance penalties change the
    ranking of strategies, so conclusions drawn from plain Sharpe do not carry.
    """
    r = w * fwd + (1.0 - w) * rf
    ex = r - rf
    sd = ex.std(ddof=1)
    return float(np.sqrt(ann) * ex.mean() / sd) if sd > 0 else 0.0


MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]
# fit_predict(df, train_idx, test_idx, seed) -> allocations for test_idx
FitPredictFn = Callable[[pd.DataFrame, np.ndarray, np.ndarray, int], np.ndarray]


# ============================================================
# 3. Runner
# ============================================================

def run_wfo(
    df: pd.DataFrame,
    folds: Sequence[Fold],
    fit_predict: FitPredictFn,
    metric_fn: MetricFn = plain_sharpe,
    seeds: Sequence[int] = (0,),
    fwd_col: str = "forward_returns",
    rf_col: str = "risk_free_rate",
    tag: str = "",
) -> pd.DataFrame:
    """One row per (fold, seed). Keep the frame -- paired_compare needs it."""
    fwd = df[fwd_col].to_numpy(float)
    rf = df[rf_col].to_numpy(float)
    rows = []

    for f in folds:
        mkt = metric_fn(np.ones(len(f.test)), fwd[f.test], rf[f.test])
        for s in seeds:
            w = np.clip(np.asarray(fit_predict(df, f.train, f.test, s), float), 0.0, 2.0)
            if w.shape != f.test.shape:
                raise ValueError(f"fold {f.k}: got {w.shape} allocations, want {f.test.shape}")
            strat = w * fwd[f.test] + (1.0 - w) * rf[f.test]
            rows.append(
                dict(
                    tag=tag,
                    fold=f.k,
                    seed=s,
                    score=metric_fn(w, fwd[f.test], rf[f.test]),
                    market_score=mkt,
                    exposure_mean=float(w.mean()),
                    exposure_std=float(w.std(ddof=1)) if len(w) > 1 else 0.0,
                    turnover=float(np.abs(np.diff(w)).mean()) if len(w) > 1 else 0.0,
                    vol_ratio=float(
                        strat.std(ddof=1) / fwd[f.test].std(ddof=1)
                    ) if fwd[f.test].std(ddof=1) > 0 else np.nan,
                    max_dd=_max_drawdown(strat),
                    test_start=int(f.test[0]),
                )
            )
    return pd.DataFrame(rows)


def _max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


# ============================================================
# 4. Summaries and paired comparison
# ============================================================

def _t_crit(dfree: int, alpha: float = 0.05) -> float:
    if _st is not None:
        return float(_st.t.ppf(1 - alpha / 2, dfree))
    return 1.96 + 2.4 / max(dfree, 1)


def summarize(scores: pd.DataFrame) -> pd.Series:
    """
    Collapse seeds first, then describe the fold distribution.

    The private test period is ONE draw from this distribution, not its mean,
    so read q10 and min alongside the mean when deciding what to ship.
    """
    per_fold = scores.groupby("fold")["score"].mean()
    mkt = scores.groupby("fold")["market_score"].mean()
    n = len(per_fold)
    m, sd = per_fold.mean(), per_fold.std(ddof=1)
    se = sd / np.sqrt(n) if n > 1 else np.nan
    half = _t_crit(n - 1) * se if n > 1 else np.nan
    stab = m / sd if sd > 0 else np.nan
    return pd.Series(
        {
            "n_folds": n,
            "mean": m,
            "std": sd,
            "stability": stab,               # mean / std across folds
            "stability_se": np.sqrt((1 + stab**2 / 2) / n) if np.isfinite(stab) else np.nan,
            "se_mean": se,
            "ci95_lo": m - half,
            "ci95_hi": m + half,
            "min": per_fold.min(),
            "q10": per_fold.quantile(0.10),
            "median": per_fold.median(),
            "beat_market_frac": float((per_fold > mkt).mean()),
            "exposure_mean": scores["exposure_mean"].mean(),
            "turnover": scores["turnover"].mean(),
            "vol_ratio_mean": scores["vol_ratio"].mean(),
            "vol_ratio_max": scores["vol_ratio"].max(),
        }
    )


def paired_compare(
    base: pd.DataFrame, variant: pd.DataFrame, n_boot: int = 10_000, seed: int = 0
) -> pd.Series:
    """
    Compare two variants on their shared folds.

    Fold-level variance is the dominant term and it cancels in the per-fold
    difference, so this is typically 3-4x more powerful than comparing two
    summarize() outputs. Run both variants over the SAME folds and seeds.
    """
    a = base.groupby("fold")["score"].mean()
    b = variant.groupby("fold")["score"].mean()
    common = a.index.intersection(b.index)
    if len(common) < 3:
        raise ValueError(f"only {len(common)} shared folds -- not comparable")
    d = (b[common] - a[common]).to_numpy()
    n = len(d)

    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    sd = d.std(ddof=1)
    return pd.Series(
        {
            "n_folds": n,
            "mean_diff": d.mean(),
            "sd_diff": sd,
            "t_stat": d.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan,
            "boot_ci95_lo": np.quantile(boot, 0.025),
            "boot_ci95_hi": np.quantile(boot, 0.975),
            "win_rate": float((d > 0).mean()),
            "detectable_at": _t_crit(n - 1) * sd / np.sqrt(n),  # smallest credible effect
        }
    )


# ============================================================
# 5. Baselines
# ============================================================
# The third one is the diagnostic that matters: if your full model does not
# beat pure volatility targeting in a paired comparison, the return model is
# contributing nothing and no amount of feature work on mu will help.

def constant_allocation(c: float = 1.0) -> FitPredictFn:
    def f(df, train_idx, test_idx, seed):
        return np.full(len(test_idx), c)
    return f


def sign_allocation(
    mu_fn: Callable[[pd.DataFrame, np.ndarray, np.ndarray, int], np.ndarray],
    hi: float = 1.5,
    lo: float = 0.5,
) -> FitPredictFn:
    def f(df, train_idx, test_idx, seed):
        mu = np.asarray(mu_fn(df, train_idx, test_idx, seed), float)
        return np.where(mu > 0, hi, lo)
    return f


def vol_target_allocation(
    vol_col: str,
    target_quantile: float = 0.5,
    lo: float = 0.0,
    hi: float = 2.0,
) -> FitPredictFn:
    """
    w_t = clip(sigma_target / sigma_hat_t, lo, hi), no return model at all.

    sigma_target is the `target_quantile` of sigma_hat over the TRAINING
    window only, so nothing from the test period enters.
    """
    def f(df, train_idx, test_idx, seed):
        s = df[vol_col].to_numpy(float)
        tgt = np.nanquantile(s[train_idx], target_quantile)
        w = tgt / np.where(np.isfinite(s[test_idx]) & (s[test_idx] > 0), s[test_idx], np.nan)
        return np.clip(np.nan_to_num(w, nan=1.0), lo, hi)
    return f


# ============================================================
# 6. Experiment log
# ============================================================

def log_experiment(path: str, tag: str, config: dict, scores: pd.DataFrame) -> None:
    """
    Append one summarized experiment. The running count of configurations you
    have tried is the required input to a deflated Sharpe / PBO correction
    later, so log every run, including the ones you abandon.
    """
    rec = summarize(scores).to_dict()
    rec.update({"tag": tag, **{f"cfg_{k}": v for k, v in config.items()}})
    row = pd.DataFrame([rec])
    header = not pd.io.common.file_exists(path)
    row.to_csv(path, mode="a", header=header, index=False)
