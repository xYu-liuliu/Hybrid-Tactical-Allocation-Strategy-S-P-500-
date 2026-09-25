"""
scale_order.py -- the smoothing is applied after the scale is solved. Should it be?

THE PROBLEM. pipeline.py solves the daily scale so that `1 + tau*z` would run at
a realised volatility ratio of 1.2 -- the point where the metric's penalty
begins -- and THEN smooths the position, which lowers realised volatility again.
The shipped rule ends at a ratio of 1.04 to 1.06. The budget is requested and
then partly handed back, so the penalty never binds and part of the allowance
goes unspent.

THE FIX IS ALGEBRAIC, NOT A SEARCH. An EWMA is a linear filter whose weights sum
to one, so for the unclipped position

    EWMA(wbar + tau*z)  ==  wbar + tau*EWMA(z)

Smoothing the POSITION and smoothing the SIGNAL are the same operation. The only
thing the current order actually changes is what the scale solver sees: it is
handed the unsmoothed z, so it sizes for a series that is never traded. Swapping
the two steps -- smooth z first, then solve tau on the smoothed z -- produces
the same shape with the scale aimed at what will actually be held.

Clipping to [0, 2] is the one nonlinearity, so the two orders are not identical
in general; the pinned fraction says how much that can matter.

THREE ARMS, and the third exists to separate two explanations:

    smooth_after   solve tau on z, form the position, then smooth it.  SHIPPED.
    smooth_first   smooth z, solve tau on the smoothed z, form the position.
                   THE PROPOSAL -- same filter, scale aimed correctly.
    smooth_z_only  smooth z but keep the tau solved on the UNSMOOTHED z. A
                   control: if this matches smooth_after, then the difference
                   between the first two arms is the re-targeting and not the
                   change of what gets filtered.

THIS IS NOT A PARAMETER SEARCH. There is one alternative ordering, it follows
from the linearity of the filter, and it has no free parameter of its own. Both
shipped configurations are carried through unchanged.

    python scale_order.py                  # dev and valid
    python scale_order.py --reveal-test    # also score test

WHAT REVEALING TEST COSTS. test has already been read once, for the
configurations selected in docs/METHOD.md. Reading it again for this variant is
a second look and has to be reported as one -- weaker selection than choosing
among 420 cells, since this is a single pre-registered deterministic change, but
not free. The default is dev and valid.
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
from typing import Dict, List

import numpy as np
import pandas as pd

import paths
import pipeline as PL
from hull_probe import load

OUT = os.path.join(paths.PROBE_DIR, "scale_order")
COST_BPS = (0, 1, 2, 5)


def smooth(v: np.ndarray, halflife: float) -> np.ndarray:
    return pd.Series(v).ewm(halflife=halflife, adjust=False).mean().to_numpy()


def arm_smooth_after(s: np.ndarray, x: np.ndarray, cfg: dict) -> np.ndarray:
    """What ships: size on z, then filter the position."""
    v = PL.remove_drift(s, cfg["drift"])
    p = PL.trailing_percentile(v, cfg["rank_window"])
    z = np.sqrt(3.0) * (2.0 * np.nan_to_num(p, nan=0.5) - 1.0)
    tau = PL.daily_scale(z, x)
    w = np.clip(PL.WBAR + tau * z, PL.MIN_POS, PL.MAX_POS)
    return np.clip(smooth(w, cfg["smooth_halflife"]), PL.MIN_POS, PL.MAX_POS)


def arm_smooth_first(s: np.ndarray, x: np.ndarray, cfg: dict) -> np.ndarray:
    """The proposal: filter the signal, then size what will actually be held."""
    v = PL.remove_drift(s, cfg["drift"])
    p = PL.trailing_percentile(v, cfg["rank_window"])
    z = np.sqrt(3.0) * (2.0 * np.nan_to_num(p, nan=0.5) - 1.0)
    zs = smooth(z, cfg["smooth_halflife"])
    tau = PL.daily_scale(zs, x)
    return np.clip(PL.WBAR + tau * zs, PL.MIN_POS, PL.MAX_POS)


def arm_smooth_z_only(s: np.ndarray, x: np.ndarray, cfg: dict) -> np.ndarray:
    """Control: filter the signal but keep the scale solved on the raw z."""
    v = PL.remove_drift(s, cfg["drift"])
    p = PL.trailing_percentile(v, cfg["rank_window"])
    z = np.sqrt(3.0) * (2.0 * np.nan_to_num(p, nan=0.5) - 1.0)
    tau = PL.daily_scale(z, x)
    zs = smooth(z, cfg["smooth_halflife"])
    return np.clip(PL.WBAR + tau * zs, PL.MIN_POS, PL.MAX_POS)


ARMS = {"smooth_after": arm_smooth_after,
        "smooth_first": arm_smooth_first,
        "smooth_z_only": arm_smooth_z_only}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    reveal = "--reveal-test" in sys.argv
    spans = ["dev", "valid"] + (["test"] if reveal else [])

    df = load()
    sp = PL.spans(df)
    folds = sp["dev"] + sp["valid"] + sp["test"]
    idx = np.concatenate([f.test for f in folds])
    span_of = np.concatenate(
        [np.full(sum(len(f.test) for f in sp[s]), s) for s in PL.SPANS])

    cache = os.path.join(PL.OUT, "signal.csv")
    if not os.path.exists(cache):
        raise SystemExit(f"{cache} is missing; run pipeline.py --commit first")
    d = pd.read_csv(cache)
    assert np.array_equal(d["pos"].to_numpy(int), idx), \
        "the cached signal does not match the current fold geometry"
    s_all = d["s"].to_numpy(float)

    fr = df["forward_returns"].to_numpy(float)[idx]
    rf = df["risk_free_rate"].to_numpy(float)[idx]
    x = fr - rf

    print("=" * 104)
    print("SCALE ORDER -- should the scale be solved before or after the smoothing?")
    print("=" * 104)
    print(f"   EWMA is linear, so EWMA(wbar + tau*z) == wbar + tau*EWMA(z);")
    print(f"   the orders differ only through the [0, 2] clip and through what")
    print(f"   the scale solver is shown.")
    print(f"   scoring {spans}" + ("" if reveal else "   (test withheld; --reveal-test to include)"))
    for name, cfg in PL.CONFIGS.items():
        print(f"   {name:<10} drift {cfg['drift']}, rank {cfg['rank_window']}, "
              f"half-life {cfg['smooth_halflife']:.0f}")
    print()

    rows: List[dict] = []
    for cname, cfg in PL.CONFIGS.items():
        ws = {a: f(s_all, x, cfg) for a, f in ARMS.items()}
        # how far the clip bites, which is the only reason the two orders can
        # differ at all once the filter's linearity is granted
        pin = {a: float(((w <= 1e-9) | (w >= PL.MAX_POS - 1e-9)).mean())
               for a, w in ws.items()}
        for span in spans:
            m = span_of == span
            for arm, w in ws.items():
                rows.append({"config": cname, "arm": arm, "span": span,
                             "pinned": pin[arm],
                             **PL.report(w[m], fr[m], rf[m])})

    t = pd.DataFrame(rows)
    cols = ["adj", "vs_bh", "vs_bh_1bps", "vs_bh_2bps", "vs_bh_5bps",
            "vol_ratio", "exposure", "turnover", "pinned"]
    for cname in PL.CONFIGS:
        print("=" * 104)
        print(f"{cname}")
        print("=" * 104)
        d2 = t[t["config"] == cname]
        for span in spans:
            print(f"\n   --- {span} (buy_and_hold "
                  f"{d2[d2['span'] == span]['buy_and_hold'].iloc[0]:.3f}) ---")
            print("   " + d2[d2["span"] == span].set_index("arm")[cols]
                  .reindex(list(ARMS)).round(3).to_string().replace("\n", "\n   "))

    print(f"""
Reading it.

  vol_ratio IS THE FIRST COLUMN TO READ. smooth_after should sit near 1.05 --
  the allowance requested and then handed back -- and smooth_first should sit
  near the 1.2 it was solved for. If it does not, the clip is binding harder
  than expected and the algebra does not carry.

  smooth_z_only SEPARATES THE TWO EXPLANATIONS. It filters the same series as
  smooth_first but keeps the old scale, so it should land on smooth_after's
  volatility. A gap between smooth_first and smooth_z_only is the re-targeting;
  a gap between smooth_z_only and smooth_after is the clip.

  MORE VOLATILITY IS NOT AUTOMATICALLY BETTER. Spending the rest of the
  allowance raises the position's swing, and turnover in the same proportion.
  The scored column is vs_bh at 0bps, because the metric contains no execution;
  the cost columns are a robustness read beside it, not the criterion. 2.2
  measured the signal's unpenalised Sharpe saturating near a ratio of 1.3, which
  says there is something left between 1.05 and 1.2 but not much beyond it --
  and a wider swing amplifies whatever is in the signal, noise included.

  THE BAR IS dev AND valid TOGETHER, the same maximin rule the configurations
  were chosen under. A variant that helps one and hurts the other has not
  earned a second look at test.

written to """ + OUT)

    with open(os.path.join(OUT, "scale_order.json"), "w") as f:
        json.dump({"spans": spans, "configs": PL.CONFIGS, "rows": rows},
                  f, indent=2, default=float)


if __name__ == "__main__":
    main()
