"""
scoring.py -- load the cached signals, and score a position against the metric.

WHAT THIS FILE IS FOR. Loading the cached signal and scoring a position are
shared machinery, not part of any one experiment, so they live here rather than
inside whichever study needs them. That keeps the dependency chain running one
way: the signal layer and the mapping layer both import this, and neither
imports the other's experiments.

WHAT IS HERE

    load_signals   the cached predictions at the frozen model cell, joined to
                   the labels and volatility columns they are scored against
    market         the benchmark position, w = 1 everywhere
    per_fold       adjusted Sharpe and its components, one row per fold
    pooled         the same over the whole span as one series
    paired         a paired contrast between two per-fold columns, with the
                   difference it could resolve
    cross          mean, t and positive fraction of a per-fold column

POOLED AND PER-FOLD ARE NOT INTERCHANGEABLE, and the difference has bitten this
project more than once. `wfo.run_wfo` scores each 84-row block and averages,
which cannot see volatility control: four scale variants spanning 0.546 to 0.677
pooled came out as ties per fold, because the metric's penalties are computed on
whatever window it is handed. The competition scores one series, so `pooled` is
the number and `per_fold` is the paired inference beside it.

`paired` returns `detectable`, twice the standard error of the difference. A gap
smaller than that is not separable from zero on the folds available, and this
project reports it beside every contrast rather than leaning on a p-value.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import paths
import capacity_window_grid as G
from hull_probe import POS_PARAMS, kaggle_adjusted_sharpe

SEEDS = (0, 1, 2)
WINDOW, TARGET = G.REF_W, 117     # the frozen model cell, from capacity_window_grid
VOL_WIN = 252                     # trailing window for sigmabar, causal


def load_signals() -> pd.DataFrame:
    """
    One row per (fold, pos): the percentile averaged over seeds, joined to the
    labels and the volatility columns a position rule needs.

    Nothing is refitted. The predictions come from capacity_window_grid's cache
    at the frozen cell, so the model layer is held exactly constant and only
    what is done with the signal varies.
    """
    import split_data as SD

    parts = [pd.read_csv(G.sig_file(WINDOW, TARGET, s)).assign(seed=s) for s in SEEDS]
    sig = pd.concat(parts, ignore_index=True)
    g = (sig.groupby(["fold", "pos"], as_index=False)
            .agg(p=("p_trail_dm", "mean"), s_raw=("s_raw", "mean"), x=("x", "first")))

    # Labels only, and from the development slice. `pos` is a feature-table row,
    # so it is shifted onto the slice. VOL_WIN is 252 and the slice carries 757
    # rows of history, so the trailing statistics below are identical to what the
    # whole table would give for every scored row.
    feat = paths.load_split("train")[
        ["date_id", "forward_returns", "risk_free_rate",
         POS_PARAMS["vol_col"], POS_PARAMS["mom_col"]]]
    idx = g["pos"].to_numpy(int) - SD.span_offset("train")
    g["fr"] = feat["forward_returns"].to_numpy(float)[idx]
    g["rf"] = feat["risk_free_rate"].to_numpy(float)[idx]

    v = feat[POS_PARAMS["vol_col"]].ffill().to_numpy(float)
    vi = v[idx]
    g["vol"] = np.where(np.isfinite(vi) & (vi >= POS_PARAMS["vol_floor"]),
                        vi, POS_PARAMS["vol_floor"])
    # trailing mean of vol^2 over the slice, then indexed: causal by construction
    vbar = pd.Series(v ** 2).rolling(VOL_WIN, min_periods=60).mean().shift(1).to_numpy()
    g["vbar"] = vbar[idx]
    g["mom"] = feat[POS_PARAMS["mom_col"]].ffill().to_numpy(float)[idx]
    return g.sort_values(["fold", "pos"]).reset_index(drop=True)


def market(g: pd.DataFrame) -> np.ndarray:
    """The benchmark: w = 1 every day. In this metric it is also the market, so
    its adjusted Sharpe is its plain Sharpe and neither penalty can bind."""
    return np.ones(len(g))


def per_fold(g: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    """Adjusted Sharpe and its components, one row per fold, indexed by fold."""
    fr_all, rf_all = g["fr"].to_numpy(float), g["rf"].to_numpy(float)
    rows = []
    for fold, idx in g.groupby("fold").indices.items():
        ww, fr, rf = w[idx], fr_all[idx], rf_all[idx]
        adj, c = kaggle_adjusted_sharpe(ww, fr, rf, return_components=True)
        if not c:
            continue
        rows.append({"fold": int(fold), "adj": adj, "sharpe": c["sharpe"],
                     "vol_ratio": c["strategy_vol_annual"] / c["market_vol_annual"],
                     "vol_pen": c["vol_penalty"], "ret_pen": c["return_penalty"],
                     "exposure": float(ww.mean()), "pos_sd": float(ww.std()),
                     "pinned": float(((ww <= 1e-9) | (ww >= 2.0 - 1e-9)).mean()),
                     "turnover": float(np.abs(np.diff(ww)).mean())})
    return pd.DataFrame(rows).set_index("fold")


def pooled(g: pd.DataFrame, w: np.ndarray) -> Dict[str, float]:
    """The whole span scored as one series, which is how the competition scores."""
    adj, c = kaggle_adjusted_sharpe(w, g["fr"].to_numpy(float), g["rf"].to_numpy(float),
                                    return_components=True)
    return {"adj": adj, "sharpe": c.get("sharpe", np.nan),
            "vol_ratio": (c["strategy_vol_annual"] / c["market_vol_annual"]) if c else np.nan,
            "vol_pen": c.get("vol_penalty", np.nan),
            "ret_pen": c.get("return_penalty", np.nan),
            "exposure": float(w.mean()), "pos_sd": float(w.std()),
            "pinned": float(((w <= 1e-9) | (w >= 2.0 - 1e-9)).mean())}


def block_ic(block, v, x):
    """
    The mean per-block Spearman between a signal and the next day's excess
    return, with its t and the fraction of blocks that are positive.

    One block is the 84 rows a single fit predicts, so this measures ordering
    INSIDE a fit and is blind to the level a position sits at over a span. That
    blindness is the point of docs/METHOD.md 2.5: the configuration that orders
    best there is not the one that pays best.

    Returns (block_ic, t, positive_fraction) as a tuple, which is how every
    caller in this project unpacks it.
    """
    ok = np.isfinite(v) & np.isfinite(x)
    per = []
    for b in np.unique(block):
        m = (block == b) & ok
        if m.sum() > 10 and np.ptp(v[m]) > 0:
            per.append(spearmanr(v[m], x[m]).correlation)
    per = np.array([q for q in per if np.isfinite(q)])
    se = per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else np.nan
    return (float(per.mean()), float(per.mean() / se) if se else np.nan,
            float((per > 0).mean()))


def cross(v: pd.Series) -> Dict[str, float]:
    """Mean, t and positive fraction of a per-fold column."""
    a = v.dropna().to_numpy(float)
    n = len(a)
    if n < 2:
        return {"n": n, "mean": np.nan, "t": np.nan, "pos_frac": np.nan}
    se = a.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(a.mean()),
            "t": float(a.mean() / se) if se else np.nan,
            "pos_frac": float((a > 0).mean())}


def paired(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    """
    b against a on the folds they share.

    `detectable` is twice the standard error of the difference: a gap smaller
    than it is not separable from zero on these folds, whatever its sign.
    """
    j = a.index.intersection(b.index)
    d = (b.loc[j] - a.loc[j]).dropna()
    n = len(d)
    if n < 2:
        return {"n": n, "diff": np.nan, "t": np.nan,
                "win_frac": np.nan, "detectable": np.nan}
    se = d.std(ddof=1) / np.sqrt(n)
    return {"n": n, "diff": float(d.mean()), "t": float(d.mean() / se) if se else np.nan,
            "win_frac": float((d > 0).mean()), "detectable": float(2.0 * se)}
