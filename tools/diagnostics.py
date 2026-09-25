"""
diagnostics.py -- decide whether the problem is mu-hat, sigma-hat, or the mapping.

Run order:
    0.  split_holdout        lock away the final segment before anything else
    1.  run_baselines        market vs pure vol targeting          (no model)
    1b. feature_ic_panel     per-feature marginal IC               (no model)
    2.  spanning_regression  does the signal add timing alpha      (cheap ridge)
    3.  decile_table         is the signal only useful in the tails

Diagnostics 2 and 3 bypass the position mapping entirely, so a bad result
there cannot be blamed on the mapping -- and a good result there means the
mapping is where the loss is happening.

Depends on wfo.py for fold geometry and summaries.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from wfo import (
    Fold,
    constant_allocation,
    make_folds,
    paired_compare,
    run_wfo,
    summarize,
    vol_target_allocation,
)

SignalFn = Callable[[pd.DataFrame, np.ndarray, np.ndarray, int], np.ndarray]


# ============================================================
# 0. Holdout
# ============================================================

def split_holdout(
    n: int,
    holdout_frac: float = 0.20,
    gap: int = 21,
    start_at: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carve the final segment off and do not look at it again until the
    method is frozen.

    The gap matters: without it the last development fold's rolling features
    overlap the holdout, and every hyperparameter you pick on development
    data has partially seen it.

    Returns (dev_idx, holdout_idx). Everything else in this module takes
    dev_idx only. Touching the holdout more than once destroys it as an
    unbiased estimate -- each look makes the number you eventually report a
    selected maximum rather than a sample.
    """
    n_hold = int(round(n * holdout_frac))
    hold_start = n - n_hold
    dev_end = hold_start - gap
    if dev_end - start_at < 1000:
        raise ValueError(f"only {dev_end - start_at} development rows left")
    return np.arange(start_at, dev_end), np.arange(hold_start, n)


# ============================================================
# Newey-West OLS
# ============================================================

def hac_ols(
    y: np.ndarray, X: np.ndarray, lags: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """OLS with Newey-West standard errors. X must include its own constant."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    if n <= k + 2:
        nan = np.full(k, np.nan)
        return nan, nan, nan, 0
    if lags is None:
        lags = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    Xu = X * u[:, None]

    S = Xu.T @ Xu
    for j in range(1, min(lags, n - 1) + 1):
        w = 1.0 - j / (lags + 1.0)
        G = Xu[j:].T @ Xu[:-j]
        S += w * (G + G.T)

    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    return beta, se, t, lags


# ============================================================
# 1. Baselines -- market vs pure volatility targeting
# ============================================================

def run_baselines(
    df: pd.DataFrame,
    folds: Sequence[Fold],
    metric_fn,
    vol_cols: Sequence[str],
    target_quantiles: Sequence[float] = (0.4, 0.5, 0.6),
    fwd_col: str = "forward_returns",
    rf_col: str = "risk_free_rate",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Market baseline plus a small grid of volatility-targeting variants.

    The grid is a multiple-testing surface: the best of 9 configurations is
    biased upward. Read the median of the grid, not its max, and log the
    count for the deflated-Sharpe correction later.

    Returns (summary_table, raw_scores).
    """
    runs: Dict[str, pd.DataFrame] = {}
    runs["market_w1"] = run_wfo(
        df, folds, constant_allocation(1.0), metric_fn,
        fwd_col=fwd_col, rf_col=rf_col, tag="market_w1",
    )
    for vc in vol_cols:
        if vc not in df.columns:
            continue
        for q in target_quantiles:
            tag = f"voltgt::{vc}::q{q:g}"
            runs[tag] = run_wfo(
                df, folds, vol_target_allocation(vc, target_quantile=q), metric_fn,
                fwd_col=fwd_col, rf_col=rf_col, tag=tag,
            )

    rows = []
    base = runs["market_w1"]
    for tag, sc in runs.items():
        s = summarize(sc)
        s["tag"] = tag
        if tag != "market_w1":
            cmp_ = paired_compare(base, sc)
            s["vs_market_diff"] = cmp_["mean_diff"]
            s["vs_market_t"] = cmp_["t_stat"]
            s["vs_market_win"] = cmp_["win_rate"]
            s["detectable_at"] = cmp_["detectable_at"]
        rows.append(s)

    table = pd.DataFrame(rows).set_index("tag").sort_values("mean", ascending=False)
    return table, pd.concat(runs.values(), ignore_index=True)


# ============================================================
# 1b. Per-feature marginal IC -- no model at all
# ============================================================

def feature_ic_panel(
    df: pd.DataFrame,
    folds: Sequence[Fold],
    feature_cols: Sequence[str],
    fwd_col: str = "forward_returns",
    rf_col: str = "risk_free_rate",
    method: str = "spearman",
    min_obs: int = 40,
) -> pd.DataFrame:
    """
    Rank correlation of each feature with the realized excess return, computed
    per fold's OOS window, then described across folds.

    ic_t is the cross-fold t-statistic of the mean IC. With ~1000 features it
    is a multiple-testing display, not a selection rule: at 5% you expect ~50
    features with |t| > 2 under the null. Read the histogram of ic_t against
    a N(0,1) reference -- a real signal shows up as excess mass in the tails,
    not as any individual name.
    """
    x = (df[fwd_col].to_numpy(float) - df[rf_col].to_numpy(float))
    cols = [c for c in feature_cols if c in df.columns]
    per_fold = np.full((len(folds), len(cols)), np.nan)

    for i, f in enumerate(folds):
        blk = df.iloc[f.test]
        xt = pd.Series(x[f.test], index=blk.index)
        for j, c in enumerate(cols):
            s = blk[c]
            ok = s.notna() & xt.notna()
            if ok.sum() >= min_obs and s[ok].nunique() > 2:
                per_fold[i, j] = s[ok].corr(xt[ok], method=method)

    ic_mean = np.nanmean(per_fold, axis=0)
    ic_std = np.nanstd(per_fold, axis=0, ddof=1)
    n_ok = np.sum(np.isfinite(per_fold), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic_t = np.where((ic_std > 0) & (n_ok > 2), ic_mean / (ic_std / np.sqrt(n_ok)), np.nan)

    return (
        pd.DataFrame(
            {
                "feature": cols,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_t": ic_t,
                "n_folds": n_ok,
                "sign_stability": np.nanmean(np.sign(per_fold) == np.sign(ic_mean), axis=0),
            }
        )
        .set_index("feature")
        .sort_values("ic_t", key=np.abs, ascending=False)
    )


# ============================================================
# 2-3. Signal collection
# ============================================================

def make_ridge_signal(
    feature_cols: Sequence[str],
    target_col: str,
    alphas: Sequence[float] = (1.0, 10.0, 100.0, 1000.0, 10_000.0),
) -> SignalFn:
    """
    Deliberately the cheapest usable signal: median-impute and standardize on
    the training window, RidgeCV, predict.

    Ridge rather than LightGBM on purpose. One hyperparameter, seed variance
    of essentially zero, seconds per fold -- so in a paired comparison the
    model contributes no noise and all the power goes to detecting whether
    the FEATURES carry anything. Its weakness as a predictor is not the point.
    """
    from sklearn.linear_model import RidgeCV

    cols = list(feature_cols)

    def f(df, train_idx, test_idx, seed):
        Xtr = df.iloc[train_idx][cols].to_numpy(float)
        Xte = df.iloc[test_idx][cols].to_numpy(float)
        ytr = df.iloc[train_idx][target_col].to_numpy(float)

        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        Xte = np.where(np.isfinite(Xte), Xte, med)

        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd = np.where(sd > 1e-10, sd, 1.0)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

        keep = np.isfinite(ytr)
        model = RidgeCV(alphas=list(alphas)).fit(Xtr[keep], ytr[keep])
        return model.predict(Xte)

    return f


def collect_signal(
    df: pd.DataFrame,
    folds: Sequence[Fold],
    signal_fn: SignalFn,
    seed: int = 0,
    fwd_col: str = "forward_returns",
    rf_col: str = "risk_free_rate",
) -> pd.DataFrame:
    """
    Run the signal over every fold's OOS window and stack the results.

    s_z is demeaned and scaled by the pooled OOS standard deviation. That
    scaling uses the whole OOS sample, which would be circular for a point
    estimate -- but the t-statistic of alpha is invariant to positive scaling
    of s, so the inference below is unaffected and only the readability of
    alpha's magnitude depends on it.
    """
    x = df[fwd_col].to_numpy(float) - df[rf_col].to_numpy(float)
    parts = []
    for f in folds:
        s = np.asarray(signal_fn(df, f.train, f.test, seed), float)
        parts.append(pd.DataFrame({"fold": f.k, "pos": f.test, "s_raw": s, "x": x[f.test]}))

    out = pd.concat(parts, ignore_index=True)
    sd = out["s_raw"].std(ddof=1)
    out["s_z"] = (out["s_raw"] - out["s_raw"].mean()) / (sd if sd > 1e-12 else 1.0)
    return out


# ============================================================
# 2. Spanning regression
# ============================================================

def spanning_regression(sig: pd.DataFrame, lags: Optional[int] = None) -> Dict[str, object]:
    """
    Regress the managed return m_t = s_t * x_t on the buy-and-hold return x_t.

    A positive alpha means the timing signal delivers something a constant
    exposure cannot. This is the test that matters when the mapping looks
    exhausted: any monotone map w = g(s) expands near its mean as
    w_bar + k*s + O(s^2), so the first-order timing contribution is
    k * E[s x]. If that expectation is statistically zero, no reshaping of
    g touches it -- you would be tuning the coefficient on a quantity that
    is itself zero.

    Returns the pooled fit plus the per-fold distribution. Individual folds
    have ~84 observations and their t-statistics are close to unreadable;
    the per-fold table is for the stability picture, not for inference.
    """
    m = (sig["s_z"] * sig["x"]).to_numpy()
    X = np.column_stack([np.ones(len(sig)), sig["x"].to_numpy()])
    beta, se, t, used = hac_ols(m, X, lags)

    rows = []
    for k, g in sig.groupby("fold"):
        if len(g) < 30:
            continue
        mk = (g["s_z"] * g["x"]).to_numpy()
        Xk = np.column_stack([np.ones(len(g)), g["x"].to_numpy()])
        bk, sek, tk, _ = hac_ols(mk, Xk, lags)
        rows.append({"fold": k, "alpha": bk[0], "alpha_t": tk[0], "beta": bk[1], "n": len(g)})
    per_fold = pd.DataFrame(rows)

    pooled = {
        "n_obs": int(len(sig)),
        "alpha": beta[0],
        "alpha_se": se[0],
        "alpha_t": t[0],
        "alpha_ann": beta[0] * 252,
        "beta": beta[1],
        "beta_t": t[1],
        "nw_lags": used,
        "corr_s_x": float(sig["s_z"].corr(sig["x"])),
    }
    if len(per_fold):
        pooled["alpha_pos_frac"] = float((per_fold["alpha"] > 0).mean())
        pooled["alpha_fold_std"] = float(per_fold["alpha"].std(ddof=1))
    return {"pooled": pd.Series(pooled), "per_fold": per_fold}


# ============================================================
# 3. Decile table
# ============================================================

def decile_table(sig: pd.DataFrame, n_bins: int = 10, by_fold: bool = True) -> pd.DataFrame:
    """
    Mean realized excess return by signal bin.

    The companion to the spanning regression. E[s x] near zero with clean
    separation between the extreme bins means the signal works only in its
    tails, and a nonlinear mapping still has room. E[s x] near zero with flat
    bins means there is nothing to map.

    by_fold ranks within each fold, which removes drift in the signal's own
    level across regimes; set False to rank pooled.
    """
    d = sig.copy()
    if by_fold:
        d["bin"] = d.groupby("fold")["s_z"].transform(
            lambda s: pd.qcut(s.rank(method="first"), n_bins, labels=False, duplicates="drop")
        )
    else:
        d["bin"] = pd.qcut(d["s_z"].rank(method="first"), n_bins, labels=False, duplicates="drop")

    rows = []
    for b, g in d.dropna(subset=["bin"]).groupby("bin"):
        xb = g["x"].to_numpy()
        _, _, t, _ = hac_ols(xb, np.ones((len(xb), 1)))
        rows.append(
            {
                "bin": int(b),
                "n": len(g),
                "s_mean": g["s_z"].mean(),
                "x_mean": xb.mean(),
                "x_mean_ann": xb.mean() * 252,
                "x_t": t[0],
                "hit_rate": float((xb > 0).mean()),
            }
        )
    tab = pd.DataFrame(rows).set_index("bin").sort_index()

    top, bot = d[d["bin"] == tab.index.max()], d[d["bin"] == tab.index.min()]
    spread = np.concatenate([top["x"].to_numpy(), -bot["x"].to_numpy()])
    _, _, ts, _ = hac_ols(spread, np.ones((len(spread), 1)))
    tab.attrs["top_minus_bottom"] = float(top["x"].mean() - bot["x"].mean())
    tab.attrs["top_minus_bottom_t"] = float(ts[0])
    tab.attrs["monotonicity"] = float(
        pd.Series(tab.index).corr(tab["x_mean"].reset_index(drop=True), method="spearman")
    )
    return tab
