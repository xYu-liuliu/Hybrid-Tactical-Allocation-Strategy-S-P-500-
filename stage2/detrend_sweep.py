"""
detrend_sweep.py -- what gets removed before the ranking, and what it costs.

docs/METHOD.md 2.3, 2.5, 2.6, 2.7 and 2.8. Three sweeps over one axis, in one
file because they read the same two cached signals over the same rows and differ
only in how finely they cut the same knob:

    A  GRID     10 detrends x 4 rank windows, plus the look-ahead ceilings.
                Coarse, and it asks whether the axis matters at all, and whether
                a cell chosen on one span means anything on the next.  2.3, 2.5.
    B  AXIS     12 rolling-mean lengths x 4 rank windows x 2 recentrings, priced
                at 4 cost levels. Fine, and it asks whether the winner in A is a
                region or one lucky cell -- and what the short end costs in
                turnover.  2.6, 2.7.
    C  CONTROL  15 ways of computing the removed mean x 7 turnover controls x
                2 rank windows. It asks whether the drift can be removed fast
                WITHOUT trading fast.  2.8.

WHY ONE AXIS AND NOT TWO. How much level a configuration removes is the detrend
and the rank window TOGETHER: a short rank window removes the slow level by
itself and needs less help in front of it, a long one needs more. Sweeping one
at a time attributes the whole effect to whichever was swept first, so every
section here is a grid and the structure to look for is anti-diagonal.

NOTHING IS REFITTED. The Ridge comes from ridge_signal.py's cache and the tree
from embargo_capacity.py's. The tree is not the shipped model -- 2.4 drops it --
but the contrast that drops it is measured here, in how a cell chosen on one
span transfers to the next, so the arm is loaded whenever its cache exists.

RANKED CONTIGUOUSLY, SCORED BY SPAN. The spans tile the timeline with no gaps,
so they are concatenated into one series before ranking: a 756-row rank window
has history to work with that valid's own 840 rows cannot supply. Slicing
happens only at scoring.

TEST IS WITHHELD BY DEFAULT. Every selection in 2.6 and 2.8 is made on dev and
valid; the test columns exist because 2.4's transfer correlations and 2.6's sign
structure are claims about all three spans, and they are printed only when the
gate is opened.

    python detrend_sweep.py                 # all three, dev and valid
    python detrend_sweep.py --grid          # A only
    python detrend_sweep.py --axis          # B only
    python detrend_sweep.py --control       # C only
    python detrend_sweep.py --reveal-test   # also score test
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
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import mapping as MAP
import paths
import ranking as RK
import ridge_signal as RS
import scoring as SC
import split_data as SD
from hull_probe import kaggle_adjusted_sharpe
from mapping_form import adj_after_costs

OUT = os.path.join(paths.PROBE_DIR, "detrend_sweep")

SELECT_SPANS = ("dev", "valid")   # what a configuration may be chosen on
COST_BPS = (0, 1, 2, 5)           # the metric itself charges nothing
BELIEVE = 0.10                    # the project's pre-registered noise floor

# ---- A ---------------------------------------------------------------------
# (label, kind, parameter). "none" ignores the parameter; "roll" is a rolling
# mean over that many rows; "blk" is the mean of the previous N complete blocks,
# or every previous block when N is 0.
DETRENDS: List[Tuple[str, str, int]] = [
    ("none", "none", 0),
    ("roll21", "roll", 21), ("roll42", "roll", 42), ("roll63", "roll", 63),
    ("roll126", "roll", 126), ("roll252", "roll", 252),
    ("blk1", "blk", 1), ("blk3", "blk", 3), ("blk6", "blk", 6),
    ("blk_all", "blk", 0),
]
GRID_WINDOWS = (126, 252, 504, 756)
INCUMBENT = ("roll63", 252)        # what shipped before the sweep

# ---- B ---------------------------------------------------------------------
# None is "no detrend"; the rest are rolling-mean lengths, filled in around 21
FINE: List[Optional[int]] = [None, 5, 8, 13, 21, 34, 42, 55, 63, 89, 126, 252]
FINE_WINDOWS = (126, 252, 504, 756)
RECENTRE_WIN = 252                 # the percentile's own trailing mean
CANDIDATE = ("ridge", 21, 252)

# ---- C ---------------------------------------------------------------------
# (label, kind, parameter, family)
LEVELS: List[Tuple[str, str, int, str]] = [
    ("none", "none", 0, "control"),
    ("mean5", "mean", 5, "short"), ("mean8", "mean", 8, "short"),
    ("mean13", "mean", 13, "short"), ("mean21", "mean", 21, "short"),
    ("mean34", "mean", 34, "long"), ("mean63", "mean", 63, "long"),
    ("ewm3", "ewm", 3, "short"), ("ewm5", "ewm", 5, "short"),
    ("ewm8", "ewm", 8, "short"), ("ewm13", "ewm", 13, "short"),
    ("ewm21", "ewm", 21, "long"),
    ("med8", "med", 8, "short"), ("med13", "med", 13, "short"),
    ("med21", "med", 21, "short"),
]
# (label, kind, parameter)
CONTROLS: List[Tuple[str, str, float]] = [
    ("raw", "raw", 0.0),
    ("smooth2", "smooth", 2.0), ("smooth3", "smooth", 3.0), ("smooth5", "smooth", 5.0),
    # the position's own sd is 0.33 to 0.62 depending on span, so a band below
    # about 0.05 is inside the daily wiggle and holds nothing back
    ("band05", "band", 0.05), ("band10", "band", 0.10), ("band20", "band", 0.20),
]
CONTROL_WINDOWS = (252, 504)
ROBUST_BPS = 2            # the extra cost column printed beside 0, for robustness


# ------------------------------------------------------------------ context
class Ctx(NamedTuple):
    """The rows every section scores, loaded once."""
    idx: np.ndarray                 # feature-table rows, strictly increasing
    blk: np.ndarray                 # the fit that produced each row
    span_of: np.ndarray             # which span each row is scored in
    x: np.ndarray                   # next day's excess return
    fr: np.ndarray                  # forward_returns, already at idx
    rf: np.ndarray                  # risk_free_rate, already at idx
    spans: Tuple[str, ...]          # active spans, in time order
    n_blocks: Dict[str, int]

    def mask(self, span: str) -> np.ndarray:
        return self.span_of == span

    def bh(self, span: str, bps: float = 0.0) -> float:
        m = self.mask(span)
        return adj_after_costs(np.ones(int(m.sum())), self.fr[m], self.rf[m], bps)


def load_ctx(spans: Sequence[str]) -> Tuple[Ctx, Dict[str, np.ndarray]]:
    """
    The active spans as one contiguous series, with the cached signals over it.

    Signals come from the caches, labels from the split slices, and the fold
    geometry from split_data -- one definition, so the three cannot disagree.
    The assertion below is what guarantees it.
    """
    spans = tuple(spans)
    pos, s_ridge, span_of = RS.load_series(spans)

    sf = SD.scoring_folds()
    blk = np.concatenate([np.concatenate(
        [np.full(len(f.test), int(f.test[0])) for f in sf[k]]) for k in spans])
    want = np.concatenate([np.concatenate([f.test for f in sf[k]]) for k in spans])
    assert np.array_equal(pos, want), \
        "the cached signals do not match the current fold geometry"
    assert (np.diff(pos) > 0).all(), "the spans are not in time order"

    df, _, lo = SD.load_through(_slice_of(spans[-1]))
    r = pos - lo
    fr = df["forward_returns"].to_numpy(float)[r]
    rf = df["risk_free_rate"].to_numpy(float)[r]

    sigs: Dict[str, np.ndarray] = {"ridge": s_ridge}
    tree = RS.load_tree_series(spans)
    if tree is not None and len(tree) == len(pos):
        sigs["tree"] = tree
    for k, v in sigs.items():
        assert len(v) == len(pos), f"{k}: {len(v)} signal rows against {len(pos)}"

    ctx = Ctx(idx=pos, blk=blk, span_of=span_of, x=fr - rf, fr=fr, rf=rf,
              spans=spans, n_blocks={k: len(sf[k]) for k in spans})
    return ctx, sigs


def _slice_of(span: str) -> str:
    return RS.SPAN_TO_SLICE[span]


# ------------------------------------------------------------------ mapping
def position(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The project's frozen mapping tail: rank -> z -> tau solved daily -> clip."""
    z = np.sqrt(3.0) * (2.0 * np.nan_to_num(p, nan=0.5) - 1.0)
    tau, zz = RK.tau_daily(z, x)
    return np.clip(MAP.WBAR + tau * zz, 0.0, 2.0)


def score(w: np.ndarray, fr: np.ndarray, rf: np.ndarray) -> dict:
    adj, c = kaggle_adjusted_sharpe(w, fr, rf, return_components=True)
    bh = kaggle_adjusted_sharpe(np.ones(len(fr)), fr, rf)
    return {"adj": adj, "bh": bh, "vs_bh": adj - bh,
            "vol_ratio": (c["strategy_vol_annual"] / c["market_vol_annual"]) if c else np.nan,
            "vol_pen": c.get("vol_penalty", np.nan),
            "exposure": float(w.mean()), "pos_sd": float(w.std()),
            "turnover": float(np.abs(np.diff(w)).mean())}


def blocked_level(s: np.ndarray, blk: np.ndarray, n_blocks: int) -> np.ndarray:
    """
    The mean of the previous n complete blocks, held constant through this one.
    n_blocks = 0 uses every previous block.

    Causal in the strong sense: decided before the block starts and never
    updated inside it, so unlike a rolling mean it cannot straddle the boundary
    where the model changed.
    """
    out = np.full(len(s), np.nan)
    means: List[float] = []
    for b in np.unique(blk):
        i = np.where(blk == b)[0]
        if means:
            take = means if n_blocks == 0 else means[-n_blocks:]
            out[i] = float(np.mean(take))
        means.append(float(np.nanmean(s[i])))
    return out


def apply_detrend(s: np.ndarray, blk: np.ndarray, kind: str, par: int) -> np.ndarray:
    if kind == "none":
        return s
    if kind == "roll":
        return RK.detrend(s, par)
    return s - blocked_level(s, blk, par)


def label(d: Optional[int]) -> str:
    return "none" if d is None else f"roll{d}"


def rank_recentre(p: np.ndarray, win: int = RECENTRE_WIN,
                  minp: int = 60) -> np.ndarray:
    """
    Put the percentile back on 0.5 using only its own past.

    This is NOT another detrend of the signal. Inside the trailing window the
    ordering is untouched; only the level of p moves, which is exactly the thing
    that leaks into average exposure. Shifted by one row, so today's percentile
    never enters its own correction.
    """
    q = pd.Series(p)
    m = q.rolling(win, min_periods=minp).mean().shift(1)
    out = (q - m + 0.5).to_numpy()
    return np.clip(np.where(np.isfinite(out), out, p), 0.001, 0.999)


def make_level(s: np.ndarray, kind: str, par: int) -> Optional[np.ndarray]:
    """The drift estimate, always shifted so today never enters its own level."""
    if kind == "none":
        return None
    q = pd.Series(s)
    if kind == "mean":
        mp = max(2, min(par, max(10, par // 3)))
        lv = q.rolling(par, min_periods=mp).mean()
    elif kind == "ewm":
        lv = q.ewm(halflife=par, adjust=False, min_periods=max(2, par)).mean()
    elif kind == "med":
        mp = max(2, min(par, max(10, par // 3)))
        lv = q.rolling(par, min_periods=mp).median()
    else:
        raise ValueError(kind)
    return lv.shift(1).to_numpy()


def apply_control(w: np.ndarray, kind: str, par: float) -> np.ndarray:
    """
    Turnover control on the POSITION, after tau has been solved.

    smooth  an EWMA of the target. Every day still moves, but less.
    band    a genuine no-trade band: the held position is carried forward
            untouched until the target has drifted more than `par` away, so most
            days trade nothing at all. Starts at 1.0, which is what the cost
            function also charges the first day against.
    """
    if kind == "raw":
        return w
    if kind == "smooth":
        return pd.Series(w).ewm(halflife=par, adjust=False).mean().to_numpy()
    if kind == "band":
        # Rebalance to the BOUNDARY, not to the target -- the standard no-trade
        # band. Snapping all the way to the target cuts the number of trades but
        # not the volume traded, and the cost is charged on volume: measured on
        # a random walk it moved turnover from 0.448 only to 0.440 while
        # dropping trading days from 300 to 258. Stopping at the edge is what
        # actually reduces the bill.
        out = np.empty(len(w))
        held = 1.0
        for t in range(len(w)):
            d = float(w[t]) - held
            if abs(d) > par:
                held += d - np.sign(d) * par
            out[t] = held
        return out
    raise ValueError(kind)


def cell_ic(c: Ctx, p: np.ndarray, m: np.ndarray) -> Tuple[float, float, float]:
    ok = m & np.isfinite(p)
    if ok.sum() <= 100:
        return np.nan, np.nan, np.nan
    return SC.block_ic(c.blk[ok], p[ok], c.x[ok])


# ============================================================== A.  the grid
def study_grid(c: Ctx, sigs: Dict[str, np.ndarray]) -> dict:
    """
    2.3 and 2.5: does the reference set matter, and does a cell chosen on one
    span mean anything on the next?

    The specific claim is narrow and carries a prior registered in advance: the
    'none' ROW should beat 'roll63' at rank window 252, because the rank window
    already removes the slow level once and the 63-day mean removes it a second
    time. Everything else in the grid is the trade between the two axes.
    """
    order = list(c.spans)
    print("=" * 116)
    print("A.  THE GRID -- how many times should the slow level be removed?")
    print("=" * 116)
    print(f"   signals {list(sigs)}, {len(DETRENDS)} detrends x "
          f"{len(GRID_WINDOWS)} rank windows = "
          f"{len(DETRENDS) * len(GRID_WINDOWS) * len(sigs)} cells")
    print(f"   incumbent {INCUMBENT[0]} x rank {INCUMBENT[1]}; every cell is a "
          f"paired difference against it\n")

    rows: List[dict] = []
    for sname, s in sigs.items():
        for dlabel, kind, par in DETRENDS:
            v = apply_detrend(s, c.blk, kind, par)
            for win in GRID_WINDOWS:
                p = RK.trailing_rank(v, win)
                w = position(p, c.x)
                for span in order:
                    m = c.mask(span)
                    ok = m & np.isfinite(p)
                    ic, tt, pf = cell_ic(c, p, m)
                    rows.append({"signal": sname, "detrend": dlabel, "rank_win": win,
                                 "span": span, "block_ic": ic, "t": tt, "pos_frac": pf,
                                 "avail": float(ok.sum() / max(1, m.sum())),
                                 **score(w[m], c.fr[m], c.rf[m])})
            print(f"   {sname:<6} {dlabel:<8} done")

    # The ceilings, at the incumbent rank window. 2.3's table is the block_ic
    # column of this block: what a perfect centring would be worth.
    ceil_rows: List[dict] = []
    for sname, s in sigs.items():
        bem = RK.block_expanding_mean(s, c.blk)
        bm = RK.block_stat(s, c.blk, np.nanmean)
        for lab, p in (("center_causal", RK.trailing_rank(s - bem, INCUMBENT[1])),
                       ("center_TRUE_LOOKAHEAD", RK.trailing_rank(s - bm, INCUMBENT[1])),
                       ("p_window_LOOKAHEAD", RK.window_rank_lookahead(s, c.blk)),
                       ("p_block_expanding", RK.block_expanding_rank(s, c.blk))):
            w = position(p, c.x)
            for span in order:
                m = c.mask(span)
                ceil_rows.append({"signal": sname, "rank": lab, "span": span,
                                  "block_ic": cell_ic(c, p, m)[0],
                                  **score(w[m], c.fr[m], c.rf[m])})
    for span in order:
        m = c.mask(span)
        ceil_rows.append({"signal": "none", "rank": "buy_and_hold", "span": span,
                          "block_ic": np.nan,
                          **score(np.ones(int(m.sum())), c.fr[m], c.rf[m])})

    t = pd.DataFrame(rows)
    inc = t[(t["detrend"] == INCUMBENT[0]) & (t["rank_win"] == INCUMBENT[1])]
    inc_key = inc.set_index(["signal", "span"])["vs_bh"]
    t["vs_incumbent"] = [r["vs_bh"] - inc_key.get((r["signal"], r["span"]), np.nan)
                         for _, r in t.iterrows()]

    bar = "-" * 116
    print("\n" + bar)
    print("A1. vs_bh -- THE PRIMARY. rows are detrend, columns are the rank window")
    print(bar)
    for sname in sigs:
        for span in order:
            d = t[(t["signal"] == sname) & (t["span"] == span)]
            piv = (d.pivot_table(index="detrend", columns="rank_win", values="vs_bh")
                   .reindex([q[0] for q in DETRENDS]))
            print(f"\n   --- {sname} / {span} (buy_and_hold {d['bh'].iloc[0]:.3f}) ---")
            print("   " + piv.round(3).to_string().replace("\n", "\n   "))

    print("\n" + bar)
    print("A2. block_ic -- the diagnostic, same grid")
    print(bar)
    for sname in sigs:
        for span in order:
            d = t[(t["signal"] == sname) & (t["span"] == span)]
            piv = (d.pivot_table(index="detrend", columns="rank_win", values="block_ic")
                   .reindex([q[0] for q in DETRENDS]))
            print(f"\n   --- {sname} / {span} ---")
            print("   " + piv.round(4).to_string().replace("\n", "\n   "))

    print("\n" + bar)
    print(f"A3. every cell positive on ALL of {order}, and the ceilings")
    print(bar)
    piv = t.pivot_table(index=["signal", "detrend", "rank_win"], columns="span",
                        values="vs_bh")[order]
    piv["all_pos"] = (piv[order] > 0).all(axis=1)
    piv["dev_rank"] = piv["dev"].rank(ascending=False).astype(int)
    keep = piv[piv["all_pos"]].sort_values("dev", ascending=False)
    print(f"\n   {len(keep)} of {len(piv)} cells positive on {', '.join(order)}")
    print("   " + (keep.round(3).to_string().replace("\n", "\n   ")
                   if len(keep) else "(none)"))
    print("\n   --- top 10 by dev ---")
    print("   " + piv.sort_values("dev", ascending=False).head(10)
          .round(3).to_string().replace("\n", "\n   "))
    print("\n   --- incumbent and ceilings ---")
    print("   " + piv.loc[[(s, INCUMBENT[0], INCUMBENT[1]) for s in sigs]]
          .round(3).to_string().replace("\n", "\n   "))
    cl = pd.DataFrame(ceil_rows).pivot_table(index=["signal", "rank"],
                                             columns="span", values="vs_bh")[order]
    print("   " + cl.round(3).to_string().replace("\n", "\n   "))
    print("\n   --- block_ic of the same rows: this is 2.3's table ---")
    ic = pd.DataFrame(ceil_rows).pivot_table(index=["signal", "rank"],
                                             columns="span", values="block_ic")[order]
    print("   " + ic.round(4).to_string().replace("\n", "\n   "))

    print("\n" + bar)
    print("A4. does a cell chosen on dev mean anything? rank correlation across cells")
    print(bar)
    tr_rows = []
    for sname in sigs:
        q = piv.loc[sname]
        for a, b in [(u, v) for i, u in enumerate(order) for v in order[i + 1:]]:
            tr_rows.append({"signal": sname, "pair": f"{a} -> {b}", "cells": len(q),
                            "pearson": float(q[a].corr(q[b])),
                            "spearman": float(q[a].corr(q[b], method="spearman"))})
    print(pd.DataFrame(tr_rows).round(3).to_string(index=False))

    print("""
Reading it.

  A1 IS THE ANSWER AND A4 DECIDES WHETHER TO BELIEVE IT. The grid will have a
  best cell whatever the truth is; what makes a best cell mean anything is that
  dev ordering carries to valid. The number to beat is this project's own +0.070
  across 37 vol overlay candidates, where 0 of 37 were positive on both spans. A
  rank correlation near zero says the detrend does not matter either way -- not
  that the top cell is the new configuration.

  THE SPECIFIC CLAIM IS NARROW AND SHOULD BE READ ON ITS OWN. It is that the
  'none' ROW beats the 'roll63' row at rank window 252, because the rank window
  already removes the level once. That is a single paired contrast predicted in
  advance, not a maximum taken over the grid, and it is the only cell here that
  carries a prior.

  THE TRADE BETWEEN THE AXES IS THE REST OF THE GRID. If the story holds, short
  rank windows should prefer no detrend and long ones should prefer some, and the
  good cells should lie on a diagonal rather than in a row or a column. A flat
  grid means the mapping's level handling was never where the money was.

  A2 IS THERE TO BE DISAGREED WITH. This project has already seen the span with
  the better ordering deliver the worse position -- block_ic 0.0946 and -0.164
  against buy-and-hold. Where A1 and A2 disagree, A1 is the one reported.
""")
    return {"grid": rows, "ceilings": ceil_rows, "transfer": tr_rows}


# ============================================================== B.  the axis
def study_axis(c: Ctx, sigs: Dict[str, np.ndarray]) -> dict:
    """
    2.6 and 2.7: is the short end a region or one lucky cell, what is the
    mechanism, and what does it cost in turnover?

    B1 fills the axis so a winner has to have neighbours. B2 tests the mechanism
    directly: if the advantage is the percentile's LEVEL leaking into average
    exposure, then recentring the percentile causally should reproduce it from
    any detrend window and flatten the curve. B3 prices every cell, which is a
    robustness read rather than a score -- the metric contains no execution.
    """
    order = list(c.spans)
    n_cells = len(FINE) * len(FINE_WINDOWS) * 2 * len(sigs)
    print("=" * 116)
    print("B.  THE AXIS -- is the short end a region, is exposure the mechanism, "
          "what does it cost?")
    print("=" * 116)
    print(f"   {len(FINE)} detrends x {len(FINE_WINDOWS)} rank windows x "
          f"2 recentrings x {len(sigs)} signals = {n_cells} cells\n")

    rows: List[dict] = []
    for sname, s in sigs.items():
        for d in FINE:
            v = apply_detrend(s, c.blk, "none" if d is None else "roll", d or 0)
            for win in FINE_WINDOWS:
                p0 = RK.trailing_rank(v, win)
                for fix, p in (("off", p0), ("on", rank_recentre(p0))):
                    w = position(p, c.x)
                    for span in order:
                        m = c.mask(span)
                        r = {"signal": sname, "detrend": label(d),
                             "d": -1 if d is None else d, "rank_win": win,
                             "recentre": fix, "span": span,
                             "block_ic": cell_ic(c, p, m)[0],
                             **score(w[m], c.fr[m], c.rf[m])}
                        for b in COST_BPS:
                            r[f"vsbh_{b}bps"] = (
                                adj_after_costs(w[m], c.fr[m], c.rf[m], b)
                                - c.bh(span, b))
                        rows.append(r)
            print(f"   {sname:<6} {label(d):<8} done")

    t = pd.DataFrame(rows)
    bar = "-" * 116

    print("\n" + bar)
    print("B1. vs_bh along the detrend axis -- IS THE SHORT END A REGION OR A POINT?")
    print(bar)
    for sname in sigs:
        for fix in ("off", "on"):
            d = t[(t["signal"] == sname) & (t["recentre"] == fix) &
                  (t["rank_win"] == CANDIDATE[2])]
            piv = (d.pivot_table(index="detrend", columns="span", values="vs_bh")[order]
                   .reindex([label(q) for q in FINE]))
            piv["all_pos"] = (piv[order] > 0).all(axis=1)
            print(f"\n   --- {sname}, rank {CANDIDATE[2]}, recentre {fix} ---")
            print("   " + piv.round(3).to_string().replace("\n", "\n   "))

    print("\n" + bar)
    print("B2. THE NEIGHBOURHOOD VERDICT: which lengths clear every active span?")
    print(bar)
    nb_rows = []
    seq = [q for q in FINE if q is not None]
    for sname in sigs:
        for fix in ("off", "on"):
            for win in FINE_WINDOWS:
                d = t[(t["signal"] == sname) & (t["recentre"] == fix) &
                      (t["rank_win"] == win)]
                pv = d.pivot_table(index="d", columns="span", values="vs_bh")[order]
                pos = (pv > 0).all(axis=1)
                run = [q for q in seq if bool(pos.get(q, False))]
                row = {"signal": sname, "recentre": fix, "rank_win": win,
                       "n_all_pos": int(pos.sum()),
                       "all_pos_detrends": ",".join(map(str, run))}
                for q in (13, 21, 34, 42):
                    row[f"d{q}"] = (float(pv.loc[q, order[-1]])
                                    if q in pv.index else np.nan)
                nb_rows.append(row)
    nb = pd.DataFrame(nb_rows)
    print(nb.round(3).to_string(index=False))
    print(f"\n   the d13..d42 columns are {order[-1]}, the last active span")

    print("\n" + bar)
    print("B3. does causal recentring of the percentile reproduce the advantage?")
    print(bar)
    print("   if exposure is the mechanism, 'on' should lift the LONG detrends")
    print("   toward the short end and flatten the curve\n")
    mech = []
    for sname in sigs:
        for span in order:
            for d in (None, 21, 63, 252):
                q = t[(t["signal"] == sname) & (t["span"] == span) &
                      (t["detrend"] == label(d)) & (t["rank_win"] == CANDIDATE[2])]
                if q.empty:
                    continue
                off = q[q["recentre"] == "off"]
                on = q[q["recentre"] == "on"]
                mech.append({"signal": sname, "span": span, "detrend": label(d),
                             "vs_bh_off": float(off["vs_bh"].iloc[0]),
                             "vs_bh_on": float(on["vs_bh"].iloc[0]),
                             "lift": float(on["vs_bh"].iloc[0] - off["vs_bh"].iloc[0]),
                             "exposure_off": float(off["exposure"].iloc[0]),
                             "exposure_on": float(on["exposure"].iloc[0])})
    print(pd.DataFrame(mech).round(3).to_string(index=False))

    print("\n" + bar)
    print("B4. TURNOVER along the axis, and what a cost would do to it")
    print(bar)
    d = t[(t["signal"] == CANDIDATE[0]) & (t["rank_win"] == CANDIDATE[2]) &
          (t["recentre"] == "off")]
    print("\n   daily turnover, rows are detrend")
    print("   " + d.pivot_table(index="detrend", columns="span", values="turnover")[order]
          .reindex([label(q) for q in FINE]).round(3)
          .to_string().replace("\n", "\n   "))
    surv = []
    for b in COST_BPS:
        pv = t.pivot_table(index=["signal", "detrend", "rank_win", "recentre"],
                           columns="span", values=f"vsbh_{b}bps")[order]
        ap = pv[(pv[order] > 0).all(axis=1)]
        surv.append({"bps": b, "all_pos_cells": len(ap), "of": len(pv),
                     "best_dev": float(ap["dev"].max()) if len(ap) else np.nan,
                     "worst_span_of_best": float(
                         ap.loc[ap["dev"].idxmax(), order].min()) if len(ap) else np.nan})
        if b in (0, ROBUST_BPS):
            print(f"\n   --- {b} bps: {len(ap)} of {len(pv)} cells positive on all "
                  f"of {', '.join(order)} ---")
            print("   " + (ap.sort_values("dev", ascending=False).head(12)
                           .round(3).to_string().replace("\n", "\n   ")
                           if len(ap) else "(none)"))
    print("\n   survival by cost level")
    print("   " + pd.DataFrame(surv).round(3).to_string(index=False)
          .replace("\n", "\n   "))

    print("\n" + bar)
    print(f"B5. the candidate itself: {CANDIDATE[0]} x roll{CANDIDATE[1]} x "
          f"rank{CANDIDATE[2]}")
    print(bar)
    q = t[(t["signal"] == CANDIDATE[0]) & (t["detrend"] == f"roll{CANDIDATE[1]}") &
          (t["rank_win"] == CANDIDATE[2])]
    cols = ["span", "recentre", "block_ic", "adj", "vs_bh", "exposure", "pos_sd",
            "vol_ratio", "turnover"] + [f"vsbh_{b}bps" for b in COST_BPS]
    print(q[cols].sort_values(["recentre", "span"]).round(3).to_string(index=False))

    print(f"""
Reading it.

  B2 IS THE DECISION AND IT IS ONE COLUMN. all_pos_detrends lists every detrend
  length positive on every active span. A list like "8,13,21" is a region and the
  effect is real: removing the level FAST is what pays, and the exact number is
  where the grid happened to land. A single entry with nothing either side of it
  is one lucky cell out of the {len(FINE)} tried, and the candidate dies here.

  B3 DECIDES WHAT TO SHIP IF B2 PASSES. The exposure story says the percentile's
  level, not the signal's, is what costs money. If recentring lifts roll63 and
  roll252 up to where the short end already is, the mechanism is confirmed and
  the rule to ship is the recentring -- one causal line, applies to any detrend,
  no tuned window. If the lift is near zero everywhere, the explanation was
  wrong, and a candidate that works for an unknown reason is worth much less
  than one that works for a known one.

  B4 IS NOT A SCORE. The metric contains no execution: it scores a position
  series, so a rule trading 40% of notional a day and one trading 5% receive the
  same treatment. 0bps is the scored number and the rest is a robustness read.
  The turnover table is the one that matters, because it is the reason the
  position is required to be smoothed at all -- a constraint from outside the
  metric, not a measurement.

  {BELIEVE:.2f} IS THE PRE-REGISTERED THRESHOLD. Passing every check here makes a
  candidate real, not large.
""")
    return {"grid": rows, "neighbourhood": nb_rows, "mechanism": mech,
            "survival": surv}


# =========================================================== C.  the control
def study_control(c: Ctx, sigs: Dict[str, np.ndarray]) -> dict:
    """
    2.8: can the drift be removed fast WITHOUT trading fast?

    B leaves one knob doing two jobs -- turnover is monotone in the detrend
    length, through the same operation -- so this section separates them. The
    removed mean and the turnover control are varied independently, and the
    verdict is read at FAMILY level so it cannot be a maximum over cells.
    """
    spans = list(c.spans)
    bh_cost = {k: {b: c.bh(k, b) for b in COST_BPS} for k in spans}

    print("=" * 118)
    print("C.  THE CONTROL -- remove the fit's offset fast, without trading fast")
    print("=" * 118)
    print(f"   {len(LEVELS)} drift removals x {len(CONTROLS)} turnover controls x "
          f"{len(CONTROL_WINDOWS)} rank windows x {len(sigs)} signals = "
          f"{len(LEVELS) * len(CONTROLS) * len(CONTROL_WINDOWS) * len(sigs)} cells")
    print(f"   verdict is read UNCHARGED, at FAMILY level, pre-registered "
          f"before the run; the cost columns are a robustness read beside it")

    rows: List[dict] = []
    for sname, s in sigs.items():
        for llabel, lkind, lpar, fam in LEVELS:
            lv = make_level(s, lkind, lpar)
            v = s if lv is None else s - lv
            for win in CONTROL_WINDOWS:
                p = RK.trailing_rank(v, win)
                w0 = position(p, c.x)
                for clabel, ckind, cpar in CONTROLS:
                    w = np.clip(apply_control(w0, ckind, cpar),
                                MAP.MIN_POS, MAP.MAX_POS)
                    for span in spans:
                        m = c.mask(span)
                        r = {"signal": sname, "level": llabel, "family": fam,
                             "control": clabel, "rank_win": win, "span": span,
                             "block_ic": cell_ic(c, p, m)[0],
                             **score(w[m], c.fr[m], c.rf[m])}
                        for b in COST_BPS:
                            r[f"vsbh_{b}"] = (adj_after_costs(w[m], c.fr[m], c.rf[m], b)
                                              - bh_cost[span][b])
                        rows.append(r)
            print(f"   {sname:<6} {llabel:<8} done")

    t = pd.DataFrame(rows)
    key = ["signal", "level", "family", "control", "rank_win"]
    bar = "-" * 118

    def allpos(bps: int) -> pd.DataFrame:
        pv = t.pivot_table(index=key, columns="span", values=f"vsbh_{bps}")[spans]
        pv["all_pos"] = (pv[spans] > 0).all(axis=1)
        pv["worst"] = pv[spans].min(axis=1)
        return pv.reset_index()

    print("\n" + bar)
    print("C1. THE VERDICT -- fraction of each family clearing every active span")
    print(bar)
    fam_rows = []
    for b in COST_BPS:
        pv = allpos(b)
        for sname in sigs:
            for fam in ("short", "long", "control"):
                q = pv[(pv["signal"] == sname) & (pv["family"] == fam)]
                if q.empty:
                    continue
                fam_rows.append({"bps": b, "signal": sname, "family": fam,
                                 "cells": len(q), "all_pos": int(q["all_pos"].sum()),
                                 "frac": float(q["all_pos"].mean()),
                                 "median_worst": float(q["worst"].median())})
    print(pd.DataFrame(fam_rows).round(3).to_string(index=False))
    print(f"\n   PASS needs the short family's frac HIGH and the long family's LOW.\n"
          f"   Scattered cells with no family gap is the same null this project "
          f"has hit four times.")

    print("\n" + bar)
    print("C2. ridge, rank 252 -- rows are drift removal, columns are turnover control")
    print(bar)
    for span in spans:
        d = t[(t["signal"] == "ridge") & (t["rank_win"] == 252) & (t["span"] == span)]
        piv = (d.pivot_table(index="level", columns="control", values="vsbh_0")
               .reindex([q[0] for q in LEVELS])[[q[0] for q in CONTROLS]])
        print(f"\n   --- {span} (buy_and_hold {bh_cost[span][0]:.3f}) ---")
        print("   " + piv.round(3).to_string().replace("\n", "\n   "))

    print("\n" + bar)
    print("C3. cells positive on every active span, by cost")
    print(bar)
    for b in COST_BPS:
        pv = allpos(b)
        ap = pv[pv["all_pos"]].sort_values("worst", ascending=False)
        print(f"\n   --- {b}bps: {len(ap)} of {len(pv)} ---")
        if len(ap):
            print("   " + ap.head(15).round(3).to_string(index=False)
                  .replace("\n", "\n   "))

    print("\n" + bar)
    print("C4. what each turnover control does to turnover, edge and the vol budget")
    print(bar)
    d = t[(t["signal"] == "ridge") & (t["rank_win"] == 252) &
          (t["level"].isin(["mean8", "mean21", "ewm5", "ewm13"]))]
    print(d.pivot_table(index=["level", "control"], columns="span",
                        values="turnover")[spans].round(3).to_string())
    print("\n   vol_ratio (1.2 is the penalty kink; under it the budget is unspent)")
    print(d.pivot_table(index=["level", "control"], columns="span",
                        values="vol_ratio")[spans].round(3).to_string())
    print("\n   vs_bh, uncharged -- the scored number")
    print(d.pivot_table(index=["level", "control"], columns="span",
                        values="vsbh_0")[spans].round(3).to_string())

    print("""
Reading it.

  C1 IS THE ANSWER AND IT IS TWO NUMBERS: the short family's frac against the
  long family's frac. A gap means the structure B found is a property of the
  drift removal and not of one window number. No gap means the previous result
  was the turnover talking, and this line closes on the same verdict as the last
  four.

  ewm AGAINST mean AT THE SAME SPEED IS THE ONE COMPARISON WITH A PRIOR. A boxcar
  mean forces a trade every time a large value falls out of the back of the
  window, and that trade carries no information. If ewm5 holds the edge of mean8
  at lower turnover, the mechanism is confirmed and the rule to ship is
  exponential, not boxcar.

  band AGAINST smooth IS A DIFFERENT TRADE-OFF. smooth moves every day by less;
  band moves on most days not at all. band is the cheaper of the two in a real
  book and the cruder of the two here, and which one keeps more of the edge says
  whether the signal's value is in its level or in its daily wiggle.

  C4 IS THE HONESTY CHECK ON C1. A control that clears by dropping vol_ratio to
  0.9 has not found an edge, it has taken less risk -- and against a benchmark at
  w=1 that shows up as an edge in a flat market and a disaster in a rising one.
  Read the vol_ratio table before believing the vs_bh table.

  DEV CANNOT PICK A WINDOW. It was flat from 0.032 to 0.098 across the whole short
  range in B, so nothing here should be selected on dev either. The family verdict
  is the only claim this section is entitled to make.
""")
    return {"grid": rows, "families": fam_rows}


# -------------------------------------------------------------------- main
def main() -> None:
    argv = sys.argv[1:]
    reveal = "--reveal-test" in argv
    want = {k for k in ("grid", "axis", "control") if f"--{k}" in argv} \
        or {"grid", "axis", "control"}
    spans = SELECT_SPANS + (("test",) if reveal else ())
    os.makedirs(OUT, exist_ok=True)

    c, sigs = load_ctx(spans)

    print("=" * 118)
    print("DETREND SWEEP -- what gets removed before the ranking, and what it costs")
    print("=" * 118)
    for k in c.spans:
        m = c.mask(k)
        role = {"dev": "fits", "valid": "selects", "test": "read once"}[k]
        print(f"   {k:<6} {c.n_blocks[k]:>3} blocks, {int(m.sum()):>5} rows, "
              f"buy_and_hold {c.bh(k):.3f}   ({role})")
    print(f"   feature-table rows {c.idx.min()}..{c.idx.max()}, "
          f"signals {list(sigs)}")
    print(f"   sections {sorted(want)}"
          + ("" if reveal else "   (test withheld; --reveal-test to include)"))
    print()

    out: Dict[str, object] = {
        "spans": list(c.spans), "reveal_test": reveal, "believe": BELIEVE,
        "cost_bps": list(COST_BPS), "robust_bps": ROBUST_BPS,
        "blocks": {k: c.n_blocks[k] for k in c.spans},
    }
    if "grid" in want:
        out["A_grid"] = study_grid(c, sigs)
    if "axis" in want:
        out["B_axis"] = study_axis(c, sigs)
    if "control" in want:
        out["C_control"] = study_control(c, sigs)

    with open(os.path.join(OUT, "detrend_sweep.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print()
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
