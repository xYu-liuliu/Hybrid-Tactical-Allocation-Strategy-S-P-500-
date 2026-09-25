"""
hull_probe.py (v2) -- model comparison, mapping comparison, causal deciles.

The 31-fold run said three things that shape this version:
  * the market baseline averages 1.170 and the linear+tree pipeline 0.762,
    so the pipeline is a negative contribution, not a neutral one;
  * the tree on all features scored 1.084 under the SAME mapping, so the
    model comparison is a clean paired test that just needs its raw scores
    written out;
  * alpha_t was 0.57 while the top-minus-bottom t was 2.64, so the payoff
    sits in the tails and a linear coefficient cannot reach it.

Three things are fixed here:

  SIGNAL CACHE. A signal is fitted once per (fold, seed) and reused by every
  mapping, so comparing six mappings costs one model run rather than six.
  Mappings are then compared on identical predictions -- the difference is
  the mapping and nothing else.

  CAUSAL RANKING. decile_table in diagnostics.py ranks within the whole
  84-day test window, which uses values from after the day being ranked.
  That made the 2.64 optimistic. Here the percentile of s_t is taken over a
  trailing window ending at t, built from out-of-fit predictions on the tail
  of the training window plus the test days already seen.

  RANK-BASED MAPPING. tanh(K * mu/sigma) depends on the absolute scale of
  mu-hat, and the selection-stability table (top-60 overlap between the
  first and last fold: 0.10) says the signal's own distribution drifts. A
  fixed K against a drifting scale is why K barely mattered. Mapping the
  trailing percentile instead removes the scale entirely.

Stages are independent; set RUN_STAGES to run a subset.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

import mapping
from wfo import Fold, make_folds, paired_compare, run_wfo, summarize
from diagnostics import hac_ols, spanning_regression

# ============================================================
# Config
# ============================================================

# Paths live in paths.py. RESULT_DIR is shared with select_probe.py and step2_tails.py.
from paths import FEATURE_PATH, PROBE_DIR as RESULT_DIR

TARGET_COL = "market_forward_excess_returns"
LABEL_COLS = ["forward_returns", "risk_free_rate", "market_forward_excess_returns", "excess_returns"]
META_COLS = ["date_id", "is_scored"]

MIN_INVESTMENT, MAX_INVESTMENT = 0.0, 2.0

TRAIN_WINDOW = 756
TEST_WINDOW = 84
EMBARGO = 10
START_AT = 504
HOLDOUT_FRAC = 0.20
HOLDOUT_GAP = 21
SEEDS = (0, 1, 2)

RUN_STAGES = ("models", "baselines", "mappings", "deciles")

# original position rule, kept for reference comparisons
POS_PARAMS = {
    "K": 6.0,
    "max_leverage": 2.0,
    "min_leverage": 0.0,
    "vol_col": "lagged_forward_returns_std21",
    "mom_col": "lagged_forward_returns_mean21",
    "vol_floor": 1e-4,
    "crash_mom_threshold": -0.0005,
}

# Bounds read off the constant-leverage table, not tuned:
# 0.8 -> 0.922, 1.0 -> 1.170, 1.2 -> 1.158, 1.5 -> 0.884. The quadratic
# return penalty punishes average exposure below 1.0 and the volatility
# penalty starts biting past 1.2, so the usable band is roughly [1.0, 1.2].
# The pipeline was averaging 0.923, inside the penalised region.
RANK_MAP = {
    "lo": 0.90,
    "hi": 1.25,
    "center": 1.08,
    "tail": 0.10,          # fraction of days sent to each extreme
    "window": 252,         # trailing window for the percentile
    "min_hist": 60,
    "ewma_span": 10,       # None disables smoothing
}

LINEAR_PARAMS = dict(
    loss="squared_error", penalty="l2", alpha=1e-4, max_iter=1000,
    learning_rate="invscaling", eta0=0.01, random_state=42,
)

# Capacity frozen by capacity_window_grid at (rolling 756, target 117): about
# 91 leaves actually grown. The previous 31/100 realised only FIVE leaves on a
# 756-row window, and moving off it was the single largest effect this project
# measured (+0.442, t=2.77). Everything downstream that reads TREE overrides
# num_leaves and min_child_samples anyway (capacity_data.tree_params,
# capacity_window_grid.tree_params, tree_sensitivity.params_for), so this
# changes hull_probe's own signals and nothing else.
TREE = dict(
    objective="regression", n_estimators=250, learning_rate=0.05,
    num_leaves=117, min_child_samples=4, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.7, reg_lambda=5.0, n_jobs=-1, verbose=-1,
)

N_SELECT = 60
INNER_FRAC = 0.60          # train[:cut] fits, train[cut:] supplies calibration


# ============================================================
# Metric and the original position rule, ported from the notebook this project
# started from (archive/model-tactical.py)
# ============================================================

def position_rule_vec(raw_pred, vol, mom, p=POS_PARAMS) -> np.ndarray:
    raw_pred = np.asarray(raw_pred, float)
    v = np.asarray(vol, float).copy()
    v[~np.isfinite(v) | (v < p["vol_floor"])] = p["vol_floor"]
    pos = 1.0 + np.tanh(p["K"] * (raw_pred / v))
    m = np.asarray(mom, float)
    pos = np.where(np.isfinite(m) & (m < p["crash_mom_threshold"]) & (pos > 1.0), 1.0, pos)
    return np.clip(pos, p["min_leverage"], p["max_leverage"])


def kaggle_adjusted_sharpe(position, fr, rf, trading_days_per_yr=252, return_components=False):
    position = np.clip(np.asarray(position, float), MIN_INVESTMENT, MAX_INVESTMENT)
    fr = np.asarray(fr, float)
    rf = np.asarray(rf, float)

    strat_ret = rf * (1.0 - position) + position * fr
    strat_excess = strat_ret - rf
    strat_mean_excess = np.prod(1.0 + strat_excess) ** (1.0 / len(strat_excess)) - 1.0
    strat_std = strat_ret.std()
    if strat_std == 0:
        return (0.0, {}) if return_components else 0.0

    sharpe = strat_mean_excess / strat_std * np.sqrt(trading_days_per_yr)
    strat_vol_annual = float(strat_std * np.sqrt(trading_days_per_yr) * 100.0)

    mkt_excess = fr - rf
    mkt_mean_excess = np.prod(1.0 + mkt_excess) ** (1.0 / len(mkt_excess)) - 1.0
    mkt_std = fr.std()
    if mkt_std == 0:
        return (0.0, {}) if return_components else 0.0
    mkt_vol_annual = float(mkt_std * np.sqrt(trading_days_per_yr) * 100.0)

    excess_vol = max(0.0, strat_vol_annual / mkt_vol_annual - 1.2) if mkt_vol_annual > 0 else 0.0
    vol_penalty = 1.0 + excess_vol
    return_gap = max(0.0, (mkt_mean_excess - strat_mean_excess) * 100.0 * trading_days_per_yr)
    return_penalty = 1.0 + (return_gap ** 2) / 100.0

    adj = float(min(sharpe / (vol_penalty * return_penalty), 1_000_000))
    if return_components:
        return adj, dict(
            sharpe=float(sharpe), strategy_vol_annual=strat_vol_annual,
            market_vol_annual=mkt_vol_annual, vol_penalty=float(vol_penalty),
            return_penalty=float(return_penalty),
            strat_mean_excess=float(strat_mean_excess), mkt_mean_excess=float(mkt_mean_excess),
        )
    return adj


def metric_fn(w, fwd, rf):
    return kaggle_adjusted_sharpe(w, fwd, rf)


# ============================================================
# Signals -- each returns (calibration predictions, test predictions)
# ============================================================

LINEAR_CLIP = 10.0        # training standard deviations; beyond this is an outlier
LINEAR_VAR_FLOOR = 1e-8   # relative to the column's own level


def _prep_linear(X_tr: pd.DataFrame, X_te: pd.DataFrame,
                 clip: float = LINEAR_CLIP, floor: float = LINEAR_VAR_FLOOR):
    """
    Standardise, with a guard against a near-constant training column.

    StandardScaler divides by the training standard deviation. A column that
    barely moves inside one 756-row window gets a near-zero divisor, and the
    same column moving normally in the test block then arrives at the model in
    the billions. Measured on this data with all 1132 columns: 599 of 5796
    development rows produced Ridge predictions above 1e6 and one reached
    7.6e11, against a median of 4.75e-3. Every linear result computed on
    development before this fix is affected; valid and test came back clean.

    Two guards, because either alone leaves a gap. Columns whose training
    standard deviation is below `floor` times their own level are left
    unscaled, which catches the degenerate case at its source. Everything is
    then clipped to `clip` standard deviations, which bounds a column that is
    small but not small enough to trip the first test.

    sklearn already replaces an EXACTLY zero scale with 1.0; the failures here
    were all tiny-but-nonzero, which it passes through.
    """
    tr = X_tr.ffill()
    sc = StandardScaler().fit(tr)
    degenerate = sc.scale_ < floor * np.maximum(np.abs(sc.mean_), 1.0)
    sc.scale_ = np.where(degenerate, 1.0, sc.scale_)

    def f(X):
        v = sc.transform(X.ffill())
        return np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), -clip, clip)

    return f(X_tr), f(X_te)


def _select_in_window(X: pd.DataFrame, y: np.ndarray, n_keep: int, seed: int) -> List[str]:
    from select_probe import rank_features_in_window   # imported late to avoid a cycle
    return rank_features_in_window(X, y, seed)[:n_keep]


def make_tree_signal(feature_cols: Sequence[str], n_select: Optional[int] = None):
    """
    Tree on every column. n_select=None is the default because the sweep put
    k=all at 1.084 and k=300 at 0.810, a paired difference of -0.274 with
    t=-2.64; every level below 300 was indistinguishable. The one real effect
    in that ladder was the cut from all columns down to 300.
    """
    cols = list(feature_cols)

    def f(df, train_idx, test_idx, seed):
        y = df[TARGET_COL].to_numpy(float)
        sel = cols if n_select is None else _select_in_window(
            df.iloc[train_idx][cols], y[train_idx], n_select, seed)

        cut = int(len(train_idx) * INNER_FRAC)
        inner_tr, inner_cal = train_idx[:cut], train_idx[cut:]

        # Each model gets the capacity of the set it is actually fitted on.
        # The inner model sees INNER_FRAC of the window, so holding
        # min_child_samples fixed would grow it a different number of leaves
        # than the full model and s_cal would then be drawn from a different
        # distribution than s_test -- which is exactly the mismatch that
        # inflated step2's usable-fold fraction. capacity_data.bagged_signal
        # does the same thing for the same reason.
        ok_i, ok_t = np.isfinite(y[inner_tr]), np.isfinite(y[train_idx])
        n_in, n_full = int(ok_i.sum()), int(ok_t.sum())
        p_inner = {**TREE, "min_child_samples": max(
            3, max(1, int(n_in * TREE["subsample"])) // TREE["num_leaves"])}
        p_full = {**TREE, "min_child_samples": max(
            3, max(1, int(n_full * TREE["subsample"])) // TREE["num_leaves"])}

        m_cal = LGBMRegressor(**{**p_inner, "random_state": seed})
        m_cal.fit(df.iloc[inner_tr[ok_i]][sel], y[inner_tr[ok_i]])
        s_cal = m_cal.predict(df.iloc[inner_cal][sel])

        m = LGBMRegressor(**{**p_full, "random_state": seed})
        m.fit(df.iloc[train_idx[ok_t]][sel], y[train_idx[ok_t]])
        return s_cal, m.predict(df.iloc[test_idx][sel])

    return f


def make_linear_tree_signal(feature_cols: Sequence[str], n_select: int = N_SELECT):
    """The original notebook's shape: a linear model plus a tree on its residuals."""
    cols = list(feature_cols)

    def f(df, train_idx, test_idx, seed):
        y = df[TARGET_COL].to_numpy(float)
        sel = _select_in_window(df.iloc[train_idx][cols], y[train_idx], n_select, seed)
        cut = int(len(train_idx) * INNER_FRAC)

        Xtr_s, Xte_s = _prep_linear(df.iloc[train_idx][sel], df.iloc[test_idx][sel])
        ytr = y[train_idx]
        lin_inner = SGDRegressor(**LINEAR_PARAMS).fit(Xtr_s[:cut], ytr[:cut])
        resid = ytr[cut:] - lin_inner.predict(Xtr_s[cut:])

        lin = SGDRegressor(**LINEAR_PARAMS).fit(Xtr_s, ytr)
        pred = lin.predict(Xte_s)
        s_cal = lin_inner.predict(Xtr_s[cut:])

        if len(resid) > 100:
            tree = LGBMRegressor(**{**TREE, "random_state": seed})
            tree.fit(df.iloc[train_idx[cut:]][sel], resid)
            pred = pred + tree.predict(df.iloc[test_idx][sel])
            s_cal = s_cal + tree.predict(df.iloc[train_idx[cut:]][sel])
        return s_cal, pred

    return f


class CachedSignal:
    """
    Fit once per (fold, seed); every mapping reads the same predictions.

    Without this, comparing six mappings refits the model six times and the
    paired differences pick up model randomness that has nothing to do with
    the mapping.
    """

    def __init__(self, fn: Callable):
        self.fn = fn
        self.cache: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def __call__(self, df, train_idx, test_idx, seed):
        key = (int(train_idx[0]), int(train_idx[-1]), int(test_idx[0]), int(seed))
        if key not in self.cache:
            self.cache[key] = self.fn(df, train_idx, test_idx, seed)
        return self.cache[key]


# ============================================================
# Mappings
# ============================================================

def trailing_percentile(
    s_cal: np.ndarray, s_test: np.ndarray, window: int = 252, min_hist: int = 60
) -> np.ndarray:
    """
    Percentile of s_t among the most recent `window` signal values ending at t.

    History starts from out-of-fit predictions on the tail of the training
    window, then grows with each test day as it is observed. Nothing from
    after day t enters day t's percentile. Days with less than min_hist of
    history return 0.5, which maps to the neutral exposure.
    """
    hist = list(np.asarray(s_cal, float)[-window:])
    out = np.full(len(s_test), 0.5)
    for j, v in enumerate(np.asarray(s_test, float)):
        arr = np.asarray(hist[-window:], float)
        arr = arr[np.isfinite(arr)]
        if len(arr) >= min_hist and np.isfinite(v):
            out[j] = float((arr < v).mean())
        hist.append(v)
    return out


def _ewma(w: np.ndarray, span: Optional[int]) -> np.ndarray:
    if not span:
        return np.asarray(w, float)
    return pd.Series(np.asarray(w, float)).ewm(span=span, adjust=False).mean().to_numpy()


def rank_to_position(p: np.ndarray, cfg: dict = RANK_MAP, mode: str = "piecewise") -> np.ndarray:
    """
    piecewise : only the extreme `tail` fractions leave the centre. This is
                what the decile table supports -- D1 through D7 rose from
                -0.099 to 0.279 annualised while D8 fell to -0.097, so the
                middle is not reliably ordered and a monotone map through it
                is fitting a shape the data does not show.
    linear    : lo to hi straight across the percentile.
    smooth    : centred tanh, for pricing how much the piecewise edges cost.
    """
    lo, hi, c, tail = cfg["lo"], cfg["hi"], cfg["center"], cfg["tail"]
    p = np.asarray(p, float)
    if mode == "piecewise":
        w = np.full(len(p), c)
        w[p <= tail] = lo
        w[p >= 1.0 - tail] = hi
        return w
    if mode == "linear":
        return lo + p * (hi - lo)
    if mode == "smooth":
        return c + (hi - lo) / 2.0 * np.tanh(3.0 * (p - 0.5) * 2.0)
    raise ValueError(mode)


def mapper_original(df: pd.DataFrame, p: dict = POS_PARAMS):
    """tanh(K * mu/sigma), the original notebook's position rule."""
    vol = df[p["vol_col"]].ffill().to_numpy(float)
    mom = df[p["mom_col"]].ffill().to_numpy(float)

    def m(d, tr, te, s_cal, s_te):
        return position_rule_vec(s_te, vol[te], mom[te], p)
    return m


def mapper_original_clamped(df: pd.DataFrame, cfg: dict = RANK_MAP, p: dict = POS_PARAMS):
    """The same rule, clipped into the usable band and smoothed."""
    base = mapper_original(df, p)

    def m(d, tr, te, s_cal, s_te):
        return _ewma(np.clip(base(d, tr, te, s_cal, s_te), cfg["lo"], cfg["hi"]), cfg["ewma_span"])
    return m


def mapper_rank(cfg: dict = RANK_MAP, mode: str = "piecewise", smooth: bool = True):
    def m(d, tr, te, s_cal, s_te):
        p = trailing_percentile(s_cal, s_te, cfg["window"], cfg["min_hist"])
        w = rank_to_position(p, cfg, mode)
        return _ewma(w, cfg["ewma_span"]) if smooth else w
    return m


def compose(signal_fn, mapper) -> Callable:
    def fit_predict(df, train_idx, test_idx, seed):
        s_cal, s_te = signal_fn(df, train_idx, test_idx, seed)
        return mapper(df, train_idx, test_idx, s_cal, s_te)
    return fit_predict


# ============================================================
# Causal decile table
# ============================================================

def collect_signal_trailing(
    df: pd.DataFrame, folds: Sequence[Fold], signal_fn, seed: int = 0, cfg: dict = RANK_MAP
) -> pd.DataFrame:
    fwd = df["forward_returns"].to_numpy(float)
    rf = df["risk_free_rate"].to_numpy(float)
    parts = []
    for f in folds:
        s_cal, s_te = signal_fn(df, f.train, f.test, seed)
        parts.append(pd.DataFrame({
            "fold": f.k,
            "pos": f.test,
            "s_raw": s_te,
            "p_trail": trailing_percentile(s_cal, s_te, cfg["window"], cfg["min_hist"]),
            "x": fwd[f.test] - rf[f.test],
        }))
    out = pd.concat(parts, ignore_index=True)
    sd = out["s_raw"].std(ddof=1)
    out["s_z"] = (out["s_raw"] - out["s_raw"].mean()) / (sd if sd > 1e-12 else 1.0)
    return out


def decile_table_trailing(sig: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Bins from the trailing percentile, so a day's bin uses only its own past.

    diagnostics.decile_table ranks inside the whole test window and therefore
    reports a separation that could not have been traded. Expect the
    top-minus-bottom t to fall from 2.64; the question is how far.
    """
    d = sig.dropna(subset=["p_trail"]).copy()
    d["bin"] = np.clip((d["p_trail"] * n_bins).astype(int), 0, n_bins - 1)

    rows = []
    for b, g in d.groupby("bin"):
        xb = g["x"].to_numpy()
        _, _, t, _ = hac_ols(xb, np.ones((len(xb), 1)))
        rows.append({"bin": int(b), "n": len(g), "p_mean": g["p_trail"].mean(),
                     "x_mean": xb.mean(), "x_mean_ann": xb.mean() * 252,
                     "x_t": t[0], "hit_rate": float((xb > 0).mean())})
    tab = pd.DataFrame(rows).set_index("bin").sort_index()

    top = d[d["bin"] == tab.index.max()]["x"].to_numpy()
    bot = d[d["bin"] == tab.index.min()]["x"].to_numpy()
    _, _, ts, _ = hac_ols(np.concatenate([top, -bot]),
                          np.ones((len(top) + len(bot), 1)))
    tab.attrs["top_minus_bottom"] = float(top.mean() - bot.mean())
    tab.attrs["top_minus_bottom_t"] = float(ts[0])
    tab.attrs["monotonicity"] = float(
        pd.Series(tab.index).corr(tab["x_mean"].reset_index(drop=True), method="spearman"))
    return tab


# ============================================================
# IO
# ============================================================

def load() -> pd.DataFrame:
    df = (pd.read_parquet(FEATURE_PATH) if FEATURE_PATH.suffix == ".parquet"
          else pd.read_csv(FEATURE_PATH))
    return df.reset_index(drop=True)


def feature_cols(df: pd.DataFrame) -> List[str]:
    num = df.select_dtypes(include=[np.number]).columns
    return [c for c in num if c not in LABEL_COLS and c not in META_COLS]


def save_runs(runs: Dict[str, pd.DataFrame], name: str) -> None:
    """Per-fold-per-seed scores, so any two runs can be paired later."""
    pd.concat(runs.values(), ignore_index=True).to_csv(
        os.path.join(RESULT_DIR, f"raw_{name}.csv"), index=False)


def load_runs(name: str, result_dir: str = RESULT_DIR) -> Dict[str, pd.DataFrame]:
    """Reload a stage so a comparison can be made without rerunning anything."""
    raw = pd.read_csv(os.path.join(result_dir, f"raw_{name}.csv"))
    return {t: g.reset_index(drop=True) for t, g in raw.groupby("tag")}


def report(runs: Dict[str, pd.DataFrame], pairs: Sequence[Tuple[str, str]], name: str):
    tbl = pd.DataFrame({k: summarize(v) for k, v in runs.items()}).T
    tbl.to_csv(os.path.join(RESULT_DIR, f"{name}_summary.csv"))
    print("\n" + tbl[["n_folds", "mean", "std", "stability", "min", "q10",
                      "exposure_mean", "turnover", "vol_ratio_mean"]].round(3).to_string())

    cmp_tbl = pd.DataFrame(
        {f"{b} - {a}": paired_compare(runs[a], runs[b]) for a, b in pairs}).T
    cmp_tbl.to_csv(os.path.join(RESULT_DIR, f"{name}_paired.csv"))
    print("\n" + cmp_tbl[["mean_diff", "sd_diff", "t_stat", "win_rate",
                          "detectable_at"]].round(3).to_string())
    save_runs(runs, name)
    return tbl, cmp_tbl


# ============================================================
# main
# ============================================================

def main() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    df = load()
    feats = feature_cols(df)
    n = len(df)
    dev_end = n - int(round(n * HOLDOUT_FRAC)) - HOLDOUT_GAP
    folds = make_folds(dev_end, TRAIN_WINDOW, TEST_WINDOW, embargo=EMBARGO, start_at=START_AT)
    print(f"rows {n}, features {len(feats)}, folds {len(folds)}")
    print(f"holdout [{n - int(round(n * HOLDOUT_FRAC))}, {n}) -- untouched")

    sig_tree = CachedSignal(make_tree_signal(feats))              # k=all, the winner
    sig_lintree = CachedSignal(make_linear_tree_signal(feats))    # the current pipeline
    orig = mapper_original(df)
    out: Dict[str, object] = {"n_folds": len(folds)}

    def market_run():
        return run_wfo(df, folds, lambda d, tr, te, s: np.ones(len(te)), metric_fn, tag="market")

    # ---- does the linear stage belong in the pipeline ----
    if "models" in RUN_STAGES:
        print("\n[models] same mapping, different signal")
        runs = {
            "lintree_k60": run_wfo(df, folds, compose(sig_lintree, orig), metric_fn,
                                   seeds=SEEDS, tag="lintree_k60"),
            "tree_kall": run_wfo(df, folds, compose(sig_tree, orig), metric_fn,
                                 seeds=SEEDS, tag="tree_kall"),
            "market": market_run(),
        }
        _, cmp_tbl = report(runs, [("lintree_k60", "tree_kall"),
                                   ("market", "tree_kall")], "models")
        out["tree_minus_lintree"] = float(cmp_tbl.loc["tree_kall - lintree_k60", "mean_diff"])
        out["tree_minus_lintree_t"] = float(cmp_tbl.loc["tree_kall - lintree_k60", "t_stat"])

    # ---- reference points around the usable band ----
    if "baselines" in RUN_STAGES:
        print("\n[baselines]")
        runs = {"market": market_run()}
        for c in (0.9, 1.08, 1.2, 1.25):
            runs[f"const{c}"] = run_wfo(
                df, folds, (lambda cc: lambda d, tr, te, s: np.full(len(te), cc))(c),
                metric_fn, tag=f"const{c}")
        report(runs, [("market", f"const{c}") for c in (0.9, 1.08, 1.2, 1.25)], "baselines")

    # ---- frozen signal, mapping only ----
    if "mappings" in RUN_STAGES:
        print("\n[mappings] signal frozen at tree_kall")
        variants = {
            "map_original": orig,
            "map_orig_clamped": mapper_original_clamped(df),
            "map_rank_piecewise": mapper_rank(mode="piecewise", smooth=True),
            "map_rank_piecewise_raw": mapper_rank(mode="piecewise", smooth=False),
            "map_rank_linear": mapper_rank(mode="linear", smooth=True),
            "map_rank_smooth": mapper_rank(mode="smooth", smooth=True),
            # the budget-solved rank map (mapping.py); tau re-solved every refit
            "map_budget_rank": mapping.make_mapper(),
            "map_budget_fixed": mapping.make_mapper(tau_source="fixed"),
        }
        runs = {k: run_wfo(df, folds, compose(sig_tree, mp), metric_fn, seeds=SEEDS, tag=k)
                for k, mp in variants.items()}
        runs["market"] = market_run()
        pairs = [("map_original", k) for k in variants if k != "map_original"]
        pairs += [("market", "map_rank_piecewise"),
                  ("map_rank_piecewise_raw", "map_rank_piecewise")]
        tbl, _ = report(runs, pairs, "mappings")
        out["best_mapping"] = str(tbl["mean"].idxmax())
        out["best_mapping_mean"] = float(tbl["mean"].max())

    # ---- is the tail separation still there without look-ahead ----
    if "deciles" in RUN_STAGES:
        print("\n[deciles] trailing-window ranking")
        sig = collect_signal_trailing(df, folds, sig_tree)
        sig.to_csv(os.path.join(RESULT_DIR, "signal_trailing.csv"), index=False)

        tab = decile_table_trailing(sig)
        tab.to_csv(os.path.join(RESULT_DIR, "deciles_trailing.csv"))
        print("\n" + tab.round(5).to_string())
        print(f"top-minus-bottom {tab.attrs['top_minus_bottom']:+.6f} "
              f"(t={tab.attrs['top_minus_bottom_t']:+.2f}), "
              f"monotonicity {tab.attrs['monotonicity']:+.2f}")
        print("  in-window ranking previously gave t=+2.64; the gap is the look-ahead")

        span = spanning_regression(sig)
        print("\n" + span["pooled"].round(5).to_string())
        out["tail_t_trailing"] = float(tab.attrs["top_minus_bottom_t"])
        out["alpha_t_trailing"] = float(span["pooled"]["alpha_t"])

    with open(os.path.join(RESULT_DIR, "verdict_v2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
