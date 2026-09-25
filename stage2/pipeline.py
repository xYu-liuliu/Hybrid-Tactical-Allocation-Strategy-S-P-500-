"""
pipeline.py -- the strategy, with the search removed.

This is the result of the work in docs/METHOD.md, written as a single path from
raw features to a daily position. Every other script in this repository exists
to justify one of the constants below; none of them is imported here, because a
strategy should be readable without its search history.

THREE CONFIGURATIONS, one per selection rule, not three attempts. All were
chosen on dev and valid alone, scored by the competition metric with nothing
added to it, inside the constraint that the position must be smoothed:

    maximin      maximise min(dev, valid)             the pre-registered rule
    max_valid    maximise valid subject to dev > 0    what two of the three rules
                                                      pick, since maximising the
                                                      mean of the two selects it too
    boxcar       the best flat-window mean            the top of the ranking is a
                                                      tie across the three ways of
                                                      computing the mean, and a
                                                      flat window is the simplest

They are reported together because the choice of rule is itself a degree of
freedom: a result that holds under every reasonable rule is worth more than one
that needs a particular rule to appear.

WHAT EACH STEP IS FOR, in one line each:

    Ridge on a rolling window      the model orders returns; it does not
                                   predict their size, so the fit is kept
                                   linear and heavily shrunk
    subtract the recent mean       a trailing 252-day percentile already removes
                                   a slow level once; removing it a second time
                                   over a LONG window destroys the ordering, and
                                   over a short one it pays for reasons the
                                   ordering statistics do not show
    trailing percentile            turn the de-levelled signal into an order,
                                   which is the only part of it that carries
                                   information
    solve the scale daily          target the metric's volatility kink at 1.2
                                   from below, re-solved every day so realised
                                   volatility cannot drift over it
    smooth the position            an unconstrained rule trades 25-50% of
                                   notional a day; the metric does not charge
                                   for that, and a deployable rule cannot do it

SPLIT DISCIPLINE. dev fits, valid selects, test is held out. Nothing in this
file tunes anything -- the constants arrived already decided -- so running it on
all three spans is a report, not a search.

    python pipeline.py                     # plan: geometry, constants, cost
    python pipeline.py --commit            # fit the signal, score all three
    python pipeline.py --score-only        # re-score the cached signal
    python pipeline.py --commit --refit    # discard the cache and fit again
"""

from __future__ import annotations

# The repo keeps entry scripts in stage1/ and stage2/ and shared modules in
# tools/. Put all three on sys.path so every `import <module>` below resolves
# no matter which directory the script is launched from.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / d)
                 for d in ("", "tools", "stage1", "stage2")]

import json
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import paths
from hull_probe import (
    HOLDOUT_FRAC, HOLDOUT_GAP, START_AT, TEST_WINDOW, TRAIN_WINDOW,
    _prep_linear, feature_cols, kaggle_adjusted_sharpe, load,
)
from mapping import solve_tau
from wfo import make_folds

OUT = os.path.join(paths.PROBE_DIR, "pipeline")

# ---------------------------------------------------------------- frozen
# Fold geometry. purge=1 because the label at row t resolves at t+1; embargo=0
# because an embargo is a correlation adjustment, not a leakage fix, and the
# protocol carries none.
EMBARGO = 0
PURGE = 1

RIDGE_ALPHA = 1000.0      # 1132 columns on 756 rows needs real shrinkage
WBAR = 1.00               # mean exposure; the frontier is flat over 0.9 - 1.1
BUDGET = 1.2              # the volatility penalty's kink, read off the metric
TAU_WINDOW = 756          # trailing rows the scale solver may use
TAU_MIN_HIST = 120
TAU_FALLBACK = 0.30       # before the solver has enough history
RANK_MIN_HIST = 40
MIN_POS, MAX_POS = 0.0, 2.0

# Three configurations, one per selection rule. All three were chosen on dev and
# valid alone, scored by the competition metric with nothing added to it, inside
# the constraint that the position must be smoothed (docs/METHOD.md, 2.7).
#
#   maximin    maximise min(dev, valid)              the pre-registered rule
#   max_valid  maximise valid subject to dev > 0     also what max mean(dev,valid)
#                                                    picks, so two rules agree here
#   boxcar     the best flat-window mean             included because the top of
#                                                    the ranking is a tie and a
#                                                    flat window is the simplest
#                                                    form of the operation, not
#                                                    because of what it scores
CONFIGS: Dict[str, dict] = {
    "maximin":   dict(drift=("ewm", 3.0),  rank_window=252, smooth_halflife=3.0),
    "max_valid": dict(drift=("mean", 5),   rank_window=504, smooth_halflife=2.0),
    "boxcar":    dict(drift=("mean", 8),   rank_window=252, smooth_halflife=5.0),
}
COST_BPS = (0, 1, 2, 5)
SPANS = ("dev", "valid", "test")


# ------------------------------------------------------------------ split
def spans(df: pd.DataFrame) -> Dict[str, list]:
    """dev fits, valid selects, test is read once. Blocks never overlap."""
    n = len(df)
    dev_end = n - int(round(n * HOLDOUT_FRAC)) - HOLDOUT_GAP
    start = dev_end + HOLDOUT_GAP
    every = make_folds(n, TRAIN_WINDOW, TEST_WINDOW, embargo=EMBARGO,
                       purge=PURGE, start_at=START_AT)
    hold = [f for f in every if int(f.test[0]) >= start]
    cut = len(hold) // 2
    return {"dev": [f for f in every if int(f.test[-1]) < dev_end],
            "valid": hold[:cut], "test": hold[cut:]}


# ------------------------------------------------------------------ model
def ridge_signal(df: pd.DataFrame, cols: Sequence[str], folds) -> np.ndarray:
    """
    Out-of-sample predictions, refit at every block.

    _prep_linear guards the standardisation: a training column that is nearly
    constant otherwise divides by a near-zero scale and sends test predictions
    into the billions.
    """
    y = (df["forward_returns"] - df["risk_free_rate"]).to_numpy(float)
    out = []
    for f in folds:
        Xtr, Xte = _prep_linear(df.iloc[f.train][cols], df.iloc[f.test][cols])
        ok = np.isfinite(y[f.train])
        out.append(Ridge(alpha=RIDGE_ALPHA).fit(Xtr[ok], y[f.train][ok]).predict(Xte))
    return np.concatenate(out)


# ---------------------------------------------------------------- mapping
def remove_drift(s: np.ndarray, drift: Tuple[str, float]) -> np.ndarray:
    """
    Subtract the recent mean of the model's own predictions.

    Three ways of computing that mean were swept and the top of the ranking is a
    tie between them, so the kind is a free choice (docs/METHOD.md, Selection):

        ("mean", k)   flat window: the last k predictions, equally weighted
        ("ewm",  h)   exponentially weighted, half-life h predictions
        ("med",  k)   rolling median of the last k

    Everything is shifted by one row, so today never enters its own level.
    min_periods is capped at the window because windows shorter than ten rows
    are in use.
    """
    kind, par = drift
    q = pd.Series(s)
    if kind == "ewm":
        lvl = q.ewm(halflife=par, adjust=False, min_periods=max(2, int(par))).mean()
    else:
        k = int(par)
        mp = max(2, min(k, max(10, k // 3)))
        r = q.rolling(k, min_periods=mp)
        lvl = r.mean() if kind == "mean" else r.median()
    return (q - lvl.shift(1)).to_numpy()


def trailing_percentile(v: np.ndarray, window: int,
                        min_hist: int = RANK_MIN_HIST) -> np.ndarray:
    """Where today sits among the last `window` days. Strictly causal."""
    p = np.full(len(v), np.nan)
    for t in range(len(v)):
        h = v[max(0, t - window):t]
        h = h[np.isfinite(h)]
        if np.isfinite(v[t]) and len(h) >= min_hist:
            p[t] = (np.sum(h < v[t]) + 0.5) / (len(h) + 1.0)
    return p


def daily_scale(z: np.ndarray, mkt_excess: np.ndarray) -> np.ndarray:
    """
    The scale that puts realised volatility on the budget, re-solved each day.

    Solving once and holding it lets realised volatility drift over the kink,
    which the metric taxes. Only rows strictly before t are used.
    """
    tau = np.full(len(z), np.nan)
    for t in range(TAU_MIN_HIST, len(z)):
        lo = max(0, t - TAU_WINDOW)
        tau[t] = solve_tau(z[lo:t], mkt_excess[lo:t], wbar=WBAR, budget=BUDGET)
    return pd.Series(tau).ffill().fillna(TAU_FALLBACK).to_numpy()


def position(s: np.ndarray, mkt_excess: np.ndarray, drift: Tuple[str, float],
             rank_window: int, smooth_halflife: float) -> np.ndarray:
    """
    Signal to daily position. This is the whole mapping.

    The smoothing is applied after the scale is solved, which leaves realised
    volatility below the budget rather than on it -- see the note at the end of
    docs/METHOD.md.
    """
    v = remove_drift(s, drift)
    p = trailing_percentile(v, rank_window)
    z = np.sqrt(3.0) * (2.0 * np.nan_to_num(p, nan=0.5) - 1.0)
    tau = daily_scale(z, mkt_excess)
    w = np.clip(WBAR + tau * z, MIN_POS, MAX_POS)
    w = pd.Series(w).ewm(halflife=smooth_halflife, adjust=False).mean().to_numpy()
    return np.clip(w, MIN_POS, MAX_POS)


# ---------------------------------------------------------------- scoring
def adjusted_sharpe(w: np.ndarray, fr: np.ndarray, rf: np.ndarray,
                    bps: float = 0.0, days: int = 252) -> float:
    """
    The competition metric, with an optional linear cost on |dw|.

    The cost leaves the strategy return before the Sharpe and both penalties are
    computed, so a busy rule is charged twice: once in the numerator and again
    through the return penalty it then triggers. Day one is charged against a
    starting position of 1.0.
    """
    if bps == 0:
        return kaggle_adjusted_sharpe(w, fr, rf)
    turn = np.abs(np.diff(np.concatenate([[1.0], np.asarray(w, float)])))
    strat = rf * (1.0 - w) + w * fr - bps / 1e4 * turn
    ex = strat - rf
    sd = strat.std()
    if sd == 0:
        return 0.0
    mean_ex = np.prod(1.0 + ex) ** (1.0 / len(ex)) - 1.0
    sharpe = mean_ex / sd * np.sqrt(days)
    mkt_ex = np.prod(1.0 + (fr - rf)) ** (1.0 / len(fr)) - 1.0
    vol_pen = 1.0 + max(0.0, (sd / fr.std()) - BUDGET)
    gap = max(0.0, (mkt_ex - mean_ex) * 100.0 * days)
    return float(sharpe / (vol_pen * (1.0 + gap ** 2 / 100.0)))


def report(w: np.ndarray, fr: np.ndarray, rf: np.ndarray) -> dict:
    adj, c = kaggle_adjusted_sharpe(w, fr, rf, return_components=True)
    bh = kaggle_adjusted_sharpe(np.ones(len(w)), fr, rf)
    out = {"adj": adj, "buy_and_hold": bh, "vs_bh": adj - bh,
           "vol_ratio": (c["strategy_vol_annual"] / c["market_vol_annual"]) if c else np.nan,
           "exposure": float(w.mean()), "pos_sd": float(w.std()),
           "turnover": float(np.abs(np.diff(w)).mean())}
    for b in COST_BPS:
        out[f"vs_bh_{b}bps"] = (adjusted_sharpe(w, fr, rf, b)
                               - adjusted_sharpe(np.ones(len(w)), fr, rf, b))
    return out


# ------------------------------------------------------------------- main
def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    commit = "--commit" in sys.argv
    score_only = "--score-only" in sys.argv
    cache = os.path.join(OUT, "signal.csv")

    df = load()
    cols = feature_cols(df)
    sp = spans(df)
    folds = sp["dev"] + sp["valid"] + sp["test"]
    idx = np.concatenate([f.test for f in folds])
    span_of = np.concatenate(
        [np.full(sum(len(f.test) for f in sp[s]), s) for s in SPANS])
    assert (np.diff(idx) > 0).all(), "spans are not contiguous in time"

    fr = df["forward_returns"].to_numpy(float)[idx]
    rf = df["risk_free_rate"].to_numpy(float)[idx]
    x = fr - rf

    print("=" * 96)
    print("PIPELINE")
    print("=" * 96)
    print(f"   {len(df)} rows, {len(cols)} feature columns")
    print(f"   rolling {TRAIN_WINDOW}, block {TEST_WINDOW}, purge {PURGE}, "
          f"embargo {EMBARGO}, Ridge(alpha={RIDGE_ALPHA:.0f})")
    for s in SPANS:
        m = span_of == s
        role = {"dev": "fits", "valid": "selects", "test": "read once"}[s]
        print(f"   {s:<6} {len(sp[s]):>3} blocks, {int(m.sum()):>5} rows, "
              f"buy_and_hold {kaggle_adjusted_sharpe(np.ones(int(m.sum())), fr[m], rf[m]):.3f}"
              f"   ({role})")
    print()
    for name, cfg in CONFIGS.items():
        k, par = cfg["drift"]
        print(f"   {name:<10} drift {k}{par:g}, rank window {cfg['rank_window']:>3}, "
              f"position EWMA half-life {cfg['smooth_halflife']:.0f}")

    have = os.path.exists(cache)
    if not (commit or score_only):
        print(f"\n   {len(folds)} closed-form Ridge fits on {TRAIN_WINDOW} rows; "
              f"a few minutes")
        print("   the fit is deterministic, so --refit reproduces the cache exactly")
        print(f"   cache {'EXISTS -- --score-only re-reads it' if have else 'absent'}")
        print("\n   --plan only. Re-run with --commit.")
        return

    refit = "--refit" in sys.argv
    if (score_only or have) and not refit:
        if not have:
            raise SystemExit(f"--score-only but {cache} is missing")
        d = pd.read_csv(cache)
        assert np.array_equal(d["pos"].to_numpy(int), idx), \
            "the cached signal does not match the current fold geometry"
        s = d["s"].to_numpy(float)
        print(f"\n   signal: cached, {len(s)} rows")
    else:
        t0 = time.time()
        s = ridge_signal(df, cols, folds)
        pd.DataFrame({"pos": idx, "s": s}).to_csv(cache, index=False)
        print(f"\n   signal: {len(folds)} fits in {time.time() - t0:.0f}s")

    # The mapping runs once over the whole out-of-sample series and is sliced
    # afterwards. A system running on the valid span would have the dev span
    # behind it, so warming the trailing windows across the boundary is what
    # deployment looks like, not a convenience.
    rows: List[dict] = []
    for name, cfg in CONFIGS.items():
        w = position(s, x, **cfg)
        for span in SPANS:
            m = span_of == span
            rows.append({"config": name, "span": span, **report(w[m], fr[m], rf[m])})
    for span in SPANS:
        m = span_of == span
        rows.append({"config": "buy_and_hold", "span": span,
                     **report(np.ones(int(m.sum())), fr[m], rf[m])})
    t = pd.DataFrame(rows)

    print(f"\n{'=' * 96}\nRESULT -- adjusted Sharpe against buy-and-hold\n{'=' * 96}")
    for name in list(CONFIGS) + ["buy_and_hold"]:
        d = t[t["config"] == name].set_index("span").reindex(SPANS)
        print(f"\n   --- {name} ---")
        print("   " + d[["adj", "buy_and_hold", "vs_bh", "vs_bh_1bps",
                         "vs_bh_2bps", "vs_bh_5bps", "exposure", "vol_ratio",
                         "turnover"]].round(3).to_string().replace("\n", "\n   "))

    print(f"""
Reading it.

  ALL THREE ROWS WERE CHOSEN ON dev AND valid, by the three rules named at the
  top of this file, inside the constraint that the position must be smoothed.
  test was scored afterwards. Re-running this file does not spend it again;
  changing a constant and re-running it does.

  vs_bh IS THE COMPETITION'S OWN NUMBER. The metric contains no transaction
  costs, so that column is what it would have scored; the bps columns are a
  robustness read and selected nothing.

  THE THREE AGREEING IS THE POINT. The top of the dev+valid ranking is a tie --
  the first eight cells differ by 0.004 in maximin score -- so no single row
  here is "the" answer. What the evidence supports is the family.

  vol_ratio SITS NEAR 1.06 AGAINST AN ALLOWANCE OF 1.2, because the position is
  smoothed after the scale is solved. Reclaiming that allowance was tested in
  scale_order.py and loses on dev.

written to """ + OUT)

    with open(os.path.join(OUT, "pipeline.json"), "w") as f:
        json.dump({"config": {"train_window": TRAIN_WINDOW, "block": TEST_WINDOW,
                              "purge": PURGE, "embargo": EMBARGO,
                              "ridge_alpha": RIDGE_ALPHA, "wbar": WBAR,
                              "budget": BUDGET, "configs": CONFIGS,
                              "cost_bps": list(COST_BPS)},
                   "blocks": {s: len(sp[s]) for s in SPANS},
                   "result": rows}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
