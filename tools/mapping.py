"""
mapping.py -- the frozen position mapping. Prediction in, position out.

This is the whole of the position layer, deliberately small. It has ONE free
number in it (BUDGET, which is read off the metric rather than fitted) and no
estimated calibration slope anywhere, which is the point: see mapping_form.py
and detrend_sweep.py for the measurements that removed everything else.

    s_t   the model's prediction of the excess return, rolling 756 / target 117

    1. DETREND        level_t = mean of the MU_WINDOW values strictly before t
                      dev_t   = s_t - level_t
    2. RANK           p_t     = trailing percentile of dev_t inside the last
                                WINDOW values of dev, strictly before t
    3. POSITION       w_t     = clip(WBAR + tau_t * sqrt(3) * (2*p_t - 1), 0, 2)
                                tau_t solved on trailing rows so the realised
                                volatility ratio sits on the metric's kink

Step 2 is what replaces a Mincer-Zarnowitz calibration. Estimating b in
mu = b*s and substituting into a closed-form optimal weight was tried and
dropped: b moved over 0.155 / 0.23 / 0.25 / 0.27 / 0.42 depending only on how
the signal was demeaned, a 2.7x swing in a number that multiplies the entire
position, and the analytic weight is nonlinear in it so the error does not
average out. A percentile needs no slope. It uses the signal's own history and
is bounded by construction, so nothing downstream can be destabilised by it.

WHY sqrt(3). p is a percentile, so 2p-1 is Uniform(-1, 1) with sd 1/sqrt(3).
Multiplying by sqrt(3) makes the bracket unit-variance, so tau_t IS the
position's standard deviation and the solver below can work in that unit.

WHY tau IS SOLVED AND NOT SET. The metric divides by
1 + max(0, sigma_strat/sigma_mkt - 1.2), so volatility is free up to 1.2x and
taxed after. The obvious closed form sqrt(WBAR^2 + tau^2) assumes the position
is uncorrelated with the return; it is not -- the signal leans long when
volatility is high -- and it overstated the affordable timing budget by about
77%, putting every configuration outside the free region. solve_tau inverts the
realised ratio instead.

THE SHORT VERSION. tau_t * sqrt(3) averaged 1.002 over the development folds,
so the whole mapping collapses to w = 2*p, a rule with no parameters at all
that scored 0.671 against the calibrated version's 0.662 -- a paired difference
of +0.007 (t=0.15) on 69 periods. Use position_simple when the point is to
explain the method. Use position when the point is to run it, because the
calibrated form lands the volatility ratio at 1.19 by construction while
w = 2*p landed at 1.30 by luck and paid a 10% volatility tax for it.

MEASURED, on the 69 development periods, never on the holdout:

    market            pooled adj 0.406
    const 1.2                    0.388     the corner that spends the budget on beta
    map_original                 0.414     1 + tanh(6*s/vol), the rule this replaces
    w = 2*p                      0.671
    position(WBAR=1.0)           0.662     vol_ratio 1.190, vol_pen 1.000
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MU_WINDOW = 63          # step 1, matches step2_tails.MU_WINDOW
WINDOW = 252            # step 2, matches capacity_data's p_trail_dm
MIN_HIST = 60
WBAR = 1.00             # mean exposure; the frontier is flat over 0.9 - 1.1
BUDGET = 1.2            # the vol_penalty kink, read off the metric
TAU_WINDOW = 756        # trailing rows the solver may use
TAU_MIN_HIST = 120
TAU_FALLBACK = 0.30     # before the solver has enough history
MIN_POS, MAX_POS = 0.0, 2.0


def detrended_percentile(s: np.ndarray, mu_window: int = MU_WINDOW,
                         window: int = WINDOW, min_hist: int = MIN_HIST) -> np.ndarray:
    """
    Steps 1 and 2. Everything is shifted by one row, so day t never enters its
    own level or its own reference distribution.

    Pass the calibration tail CONCATENATED ahead of the live rows. Without it
    the first mu_window rows have no level and the first min_hist have no
    reference window, which on 84-row blocks is most of the block.
    """
    z = pd.Series(np.asarray(s, float))
    level = z.rolling(mu_window, min_periods=max(10, mu_window // 3)).mean().shift(1)
    dev = (z - level).to_numpy()

    p = np.full(len(dev), np.nan)
    for t in range(len(dev)):
        lo = max(0, t - window)
        hist = dev[lo:t]
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(dev[t]) or len(hist) < min_hist:
            continue
        p[t] = (np.sum(hist < dev[t]) + 0.5) / (len(hist) + 1.0)
    return p


def solve_tau(z: np.ndarray, mkt_excess: np.ndarray, wbar: float = WBAR,
              budget: float = BUDGET) -> float:
    """
    The tau whose realised volatility ratio is the budget, on the rows given.

    With w = wbar + tau*z the strategy return is rf + w*(fr - rf), so its
    variance is quadratic in tau:

        var(wbar*m) + 2*tau*cov(wbar*m, z*m) + tau^2*var(z*m) = (budget*sd(m))^2

    The smaller positive root is the one that scales up from zero rather than
    coming back through the far side of the parabola. The cross term is exactly
    the signal-volatility correlation that the closed form drops.
    """
    m = np.asarray(mkt_excess, float)
    ok = np.isfinite(m) & np.isfinite(z)
    m, zz = m[ok], np.asarray(z, float)[ok]
    if len(m) < 2:
        return TAU_FALLBACK
    b2 = np.var(zz * m)
    if b2 <= 0:
        return 0.0
    c = [b2, 2.0 * np.cov(wbar * m, zz * m)[0, 1],
         np.var(wbar * m) - (budget * np.std(m)) ** 2]
    roots = [r.real for r in np.roots(c) if abs(r.imag) < 1e-9 and r.real > 0]
    return float(min(roots)) if roots else 0.0


def position(p: np.ndarray, mkt_excess: np.ndarray, wbar: float = WBAR,
             budget: float = BUDGET, window: int = TAU_WINDOW,
             min_hist: int = TAU_MIN_HIST) -> np.ndarray:
    """
    Step 3. p from detrended_percentile, mkt_excess the realised market excess
    return aligned to it -- only rows STRICTLY before t are ever read, so this
    is usable live.
    """
    p = np.asarray(p, float)
    z = np.sqrt(3.0) * (2.0 * np.where(np.isfinite(p), p, 0.5) - 1.0)
    m = np.asarray(mkt_excess, float)

    tau = np.full(len(p), np.nan)
    for t in range(len(p)):
        lo = max(0, t - window)
        if t - lo >= min_hist:
            tau[t] = solve_tau(z[lo:t], m[lo:t], wbar, budget)
    tau = pd.Series(tau).ffill().fillna(TAU_FALLBACK).to_numpy()
    return np.clip(wbar + tau * z, MIN_POS, MAX_POS)


def make_mapper(mu_window: int = MU_WINDOW, window: int = WINDOW,
                min_hist: int = MIN_HIST, wbar: float = WBAR,
                budget: float = BUDGET, tau_source: str = "cal"):
    """
    A hull_probe-compatible mapper: m(df, train_idx, test_idx, s_cal, s_te) -> w.

    tau_source picks where the scale is solved, and the two options are not
    interchangeable:

      "cal"    the fold's own inner-calibration block, about 302 rows of
               out-of-fit predictions sitting immediately before the test
               window, EXTENDED each day by the test rows already observed.
               Self-contained, carries no state between folds, and re-solves
               daily, which is what makes it fit the walk-forward architecture.
      "fixed"  no solving at all; tau is held at TAU_FALLBACK. Only useful as
               a control to show what the solver is worth.

    THE DAILY RE-SOLVE IS NOT COSMETIC. Solving once per fold and holding tau
    for all 84 days lets the realised volatility drift off the kink: measured,
    it landed at 1.231 and paid a 3.1% tax, for a pooled 0.546. Re-solving each
    day on the window ending at t brings it back to 1.199, vol_pen 1.000, and
    the pooled score to 0.677. The per-fold paired difference is inside its own
    resolution (-0.016, t=-1.02); the gain is entirely in volatility control,
    which only the pooled number can see.

    That 0.677 is also the best configuration measured anywhere in this project
    -- above the 0.601 of the rolling-756 solver that runs ACROSS fold
    boundaries, which this cannot use because at fold time the earlier folds'
    signals came from different models. The walk-forward-evaluable path is not
    a compromise here; it wins, because s_cal gives the percentile a proper
    warm-up that a contiguous replay has to do without.
    """
    if tau_source not in ("cal", "fixed"):
        raise ValueError(f"tau_source must be 'cal' or 'fixed', got {tau_source!r}")

    def m(df, train_idx, test_idx, s_cal, s_te):
        s_cal = np.asarray(s_cal, float)
        s_te = np.asarray(s_te, float)
        p_all = detrended_percentile(np.concatenate([s_cal, s_te]),
                                     mu_window, window, min_hist)
        z_all = np.sqrt(3.0) * (2.0 * np.where(np.isfinite(p_all), p_all, 0.5) - 1.0)

        n_cal = len(s_cal)
        if tau_source == "fixed":
            tau = np.full(len(s_te), TAU_FALLBACK)
        else:
            # realised market excess over the calibration block and then the
            # test block; row t only ever reads rows strictly before it
            idx = np.concatenate([np.asarray(train_idx)[-n_cal:],
                                  np.asarray(test_idx)])
            mkt = (df["forward_returns"].to_numpy(float)[idx]
                   - df["risk_free_rate"].to_numpy(float)[idx])
            tau = np.empty(len(s_te))
            last = TAU_FALLBACK
            for k in range(len(s_te)):
                end = n_cal + k
                last = (solve_tau(z_all[:end], mkt[:end], wbar, budget)
                        if end >= TAU_MIN_HIST else last)
                tau[k] = last

        return np.clip(wbar + tau * z_all[n_cal:], MIN_POS, MAX_POS)

    return m


def position_simple(p: np.ndarray) -> np.ndarray:
    """
    The same rule with the scale taken as the constant it converged to.

    tau_t * sqrt(3) averaged 1.002 on the development folds, which is what
    collapses the bracket to 2p - 1 and the whole mapping to twice the
    percentile. Statistically indistinguishable from position(); it simply does
    not control its own volatility ratio, so prefer position() to run.
    """
    return np.clip(2.0 * np.asarray(p, float), MIN_POS, MAX_POS)
