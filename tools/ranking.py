"""
ranking.py -- turn a prediction series into an order, and take a block apart.

WHAT THIS FILE IS FOR. Every script that turns a prediction into a position needs
the same handful of transforms, and they have to agree exactly -- two scripts
ranking against slightly different reference windows produce numbers that cannot
be compared. They are defined once here and imported, never re-implemented.

THE CAUSAL / LOOK-AHEAD SPLIT IS THE POINT OF THE FILE. A deployed rule may use
only the causal half. The look-ahead half exists to measure ceilings: what a
perfect estimate of a block's level, or a perfect rank inside it, would have been
worth. Mixing the two silently is the one mistake this file is arranged to
prevent, so each function is labelled below and in its own docstring.

WHAT IS HERE, and which of them a deployed rule may use:

    detrend                 CAUSAL   subtract a trailing rolling mean
    trailing_rank           CAUSAL   percentile against the last `win` rows
    block_expanding_mean    CAUSAL   the block's mean of the days seen so far
    block_expanding_rank    CAUSAL   day k against days 1..k-1 of its own block
    window_rank_lookahead   LOOK-AHEAD  the full within-block rank
    block_stat              LOOK-AHEAD  any per-block statistic, broadcast back

The two look-ahead functions are not candidates. They are ceilings: the best any
causal approximation of them could reach. `window_rank_lookahead` needs the rest
of the block to know where today sits in it, and `block_stat` is how the block's
true mean and spread get measured. Keeping them in the same file as the causal
ones is deliberate -- every table that quotes a ceiling should make it obvious
that it is one.

EVERY CAUSAL FUNCTION SHIFTS BY ONE ROW, so today never enters its own level or
its own reference distribution. The defaults come from mapping.py, so the frozen
constants have one definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mapping as MAP

MIN_HIST = 20          # a block's first rows have too little of themselves to rank
RANK_MIN_HIST = 40     # rows a trailing window needs before it will score


# ------------------------------------------------------------------ causal
def detrend(s: np.ndarray, window: int = None) -> np.ndarray:
    """
    Subtract a trailing rolling mean. `window` defaults to mapping.MU_WINDOW.

    This is the step docs/METHOD.md 2.6 sweeps: at the frozen 63 it removes the
    slow level a second time, after the rank window has already removed it once.
    """
    w = MAP.MU_WINDOW if window is None else int(window)
    lvl = pd.Series(s).rolling(w, min_periods=max(2, min(w, max(10, w // 3)))) \
                      .mean().shift(1)
    return (pd.Series(s) - lvl).to_numpy()


def trailing_rank(v: np.ndarray, win: int = MAP.WINDOW,
                  minp: int = RANK_MIN_HIST) -> np.ndarray:
    """Where today sits among the last `win` rows. Strictly causal."""
    p = np.full(len(v), np.nan)
    for t in range(len(v)):
        h = v[max(0, t - win):t]
        h = h[np.isfinite(h)]
        if np.isfinite(v[t]) and len(h) >= minp:
            p[t] = (np.sum(h < v[t]) + 0.5) / (len(h) + 1.0)
    return p


def block_expanding_mean(s: np.ndarray, blk: np.ndarray) -> np.ndarray:
    """
    The block's own mean of the days seen so far. Causal, exact at the boundary.

    Unlike a rolling mean it cannot straddle the point where the model was
    refitted, which is why it is the natural causal estimate of a block's level.
    """
    out = np.full(len(s), np.nan)
    for b in np.unique(blk):
        i = np.where(blk == b)[0]
        out[i] = pd.Series(s[i]).expanding().mean().shift(1).to_numpy()
    return out


def block_expanding_rank(s: np.ndarray, blk: np.ndarray,
                         min_hist: int = MIN_HIST) -> np.ndarray:
    """
    Day k of a block ranked against days 1..k-1 of that block. Causal.

    The opening of every block has almost nothing to rank against, so this
    scores only about three quarters of the rows -- which is part of why it does
    not reach the look-ahead ceiling.
    """
    out = np.full(len(s), np.nan)
    for b in np.unique(blk):
        i = np.where(blk == b)[0]
        v = s[i]
        for k in range(len(v)):
            h = v[:k]
            h = h[np.isfinite(h)]
            if np.isfinite(v[k]) and len(h) >= min_hist:
                out[i[k]] = (np.sum(h < v[k]) + 0.5) / (len(h) + 1.0)
    return out


def frozen_z(s: np.ndarray) -> np.ndarray:
    """
    The mapping that shipped before the sweep: detrend 63, trailing percentile
    252, then scaled to unit variance. Contiguous, causal, no warm-up block.
    """
    p = trailing_rank(detrend(s))
    return np.sqrt(3.0) * (2.0 * np.nan_to_num(p, nan=0.5) - 1.0)


# ------------------------------------------------------------- look-ahead
def window_rank_lookahead(s: np.ndarray, blk: np.ndarray) -> np.ndarray:
    """
    The FULL within-block rank. Needs the rest of the block, so NOT deployable.

    The ceiling on anything that tries to rank inside one fit's own output.
    """
    out = np.full(len(s), np.nan)
    for b in np.unique(blk):
        i = np.where(blk == b)[0]
        r = pd.Series(s[i]).rank(method="first").to_numpy()
        out[i] = (r - 0.5) / len(i)
    return out


def block_stat(s: np.ndarray, blk: np.ndarray, fn) -> np.ndarray:
    """A per-block statistic broadcast back over the block. LOOK-AHEAD."""
    out = np.full(len(s), np.nan)
    for b in np.unique(blk):
        i = np.where(blk == b)[0]
        out[i] = float(fn(s[i]))
    return out


# ----------------------------------------------------------------- scale
def tau_daily(z: np.ndarray, mkt_excess: np.ndarray):
    """
    The position scale, re-solved every day on strictly trailing rows.

    Returns (tau, z) with z cleaned of non-finite values, because every caller
    wants both. Solving once and holding it lets realised volatility drift over
    the metric's kink -- measured at 1.231 and a 3.1% tax against 1.199 and none.
    """
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    t = np.array([MAP.solve_tau(z[max(0, i - MAP.TAU_WINDOW):i],
                                mkt_excess[max(0, i - MAP.TAU_WINDOW):i])
                  if i >= MAP.TAU_MIN_HIST else np.nan for i in range(len(z))])
    return pd.Series(t).ffill().fillna(MAP.TAU_FALLBACK).to_numpy(), z
