"""
signal_diagnostics.py -- what kind of signal is this, and why the mapping is
built the way it is.

THE ONE FACT THE WHOLE MAPPING RESTS ON. The model's ORDERING is informative
and its LEVEL is not. Everything the position rule does follows from that, so
it is worth establishing directly rather than assuming: the rule keeps the rank
and throws the magnitude away, and that is only correct if the magnitude is
genuinely broken.

Expect, on the development folds:

    R2_oos of the raw prediction                       large and NEGATIVE
    R2_oos after a causal rescale                      small and positive
    rank IC                                            positive, high t
    prediction sd / justified forecast sd              about 6x too confident

A model that fails the squared-error test by that margin would be thrown out on
Gu-Kelly-Xiu's Table 1 criteria, and it still produces a usable strategy. That
is not a contradiction; it is what "ranker" means, and it is the reason the
frozen mapping has no calibration slope anywhere in it.

TWO TRAPS ARE REPRODUCED HERE ON PURPOSE. Reading the mapping's shape off the
conditional mean E[x | s] failed twice in this project, both times because the
signal has to be normalised before the relationship is visible and the choice
of normalisation decides what you see:

    pooled across folds       the middle 80% looks FLAT (slope ~0.00, t~0.1)
                              and only the tails carry slope. This suggests a
                              two-tail or step mapping. It is an artifact: the
                              signal's level drifts between folds and pooling
                              different scales attenuates the middle.
    demeaned within fold      the relationship looks monotone and strong
                              (slope ~0.42). Also wrong: the fold mean uses the
                              whole fold, so it is look-ahead.
    demeaned causally         the honest number, roughly halfway. This is what
                              a deployed rule can actually see.

Section C prints all three side by side. The lesson is not which slope is
right; it is that a mapping argued from E[x | s] inherits whichever
normalisation the analyst happened to pick, which is why the mapping in this
project is chosen by scoring POSITIONS instead (mapping_form.py).

    python signal_diagnostics.py
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

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import scoring as M
import paths

OUT = os.path.join(paths.PROBE_DIR, "signal_diagnostics")
N_BINS = 12
MIN_HIST_MEAN = 252          # warm-up for the causal historical-mean benchmark
CAUSAL_DEMEAN = (21, 63)     # trailing windows for the causal demeaning arms
BLEND = (0.0, 0.25, 0.5, 0.75, 1.0)


def r2_oos(pred: np.ndarray, bench: np.ndarray, y: np.ndarray) -> float:
    """Percent reduction in squared error against a benchmark forecast."""
    return 100.0 * (1.0 - np.sum((y - pred) ** 2) / np.sum((y - bench) ** 2))


def slope_t(a: np.ndarray, b: np.ndarray):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 3 or np.var(a) == 0:
        return np.nan, np.nan, n
    sl, ic = np.polyfit(a, b, 1)
    resid = b - (ic + sl * a)
    se = np.sqrt((resid @ resid / (n - 2)) / ((a - a.mean()) @ (a - a.mean())))
    return sl, sl / se if se else np.nan, n


def rank01_by_fold(g: pd.DataFrame, v: np.ndarray) -> np.ndarray:
    out = np.empty(len(v))
    for _, idx in g.groupby("fold").indices.items():
        out[idx] = (pd.Series(v[idx]).rank(method="first") - 0.5) / len(idx)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    g = M.load_signals()
    s, x = g["s_raw"].to_numpy(float), g["x"].to_numpy(float)
    ok = np.isfinite(s) & np.isfinite(x)
    s, x, gg = s[ok], x[ok], g[ok].reset_index(drop=True)
    out = {}

    # ---- A. level vs ordering ------------------------------------------
    print("=" * 96)
    print("A.  the level is broken and the ordering is not")
    print("=" * 96)
    hm = pd.Series(x).expanding(min_periods=MIN_HIST_MEAN).mean().shift(1).to_numpy()
    k = np.isfinite(hm)
    b_full = np.cov(s, x)[0, 1] / np.var(s)
    b_causal = np.full(len(s), np.nan)
    for t in range(MIN_HIST_MEAN, len(s)):
        if np.var(s[:t]) > 0:
            b_causal[t] = np.cov(s[:t], x[:t])[0, 1] / np.var(s[:t])
    b_causal = pd.Series(b_causal).ffill().fillna(0.0).to_numpy()
    kk = k & (np.arange(len(s)) >= MIN_HIST_MEAN)

    rows = [("raw prediction", r2_oos(s[k], hm[k], x[k])),
            (f"rescaled by the full-sample b={b_full:.3f} (LOOK-AHEAD)",
             r2_oos(b_full * s[k], hm[k], x[k])),
            (f"rescaled causally (expanding b, mean {np.nanmean(b_causal[kk]):.3f})",
             r2_oos(b_causal[kk] * s[kk], hm[kk], x[kk])),
            ("the historical mean itself", 0.0)]
    print("   R2_oos against the causal historical mean, the benchmark Gu-Kelly-Xiu")
    print("   say is the right one for an aggregate index (they use zero for single stocks):")
    for lab, v in rows:
        print(f"      {lab:<58} {v:+8.3f}%")
    print(f"\n   against a zero forecast instead:                           "
          f"{r2_oos(s, 0.0, x):+8.3f}%")
    print(f"   prediction sd {s.std():.5f} vs realised sd {x.std():.5f}")
    print(f"   optimal b = {b_full:.3f}, so the prediction is "
          f"{1 / b_full:.1f}x OVER-CONFIDENT")
    ic_fold = np.array([spearmanr(g["p"].to_numpy()[i], g["x"].to_numpy()[i]).correlation
                        for _, i in g.groupby("fold").indices.items()])
    ic_fold = ic_fold[np.isfinite(ic_fold)]
    print(f"\n   rank IC per fold: mean {ic_fold.mean():+.4f}  "
          f"t={ic_fold.mean() / (ic_fold.std(ddof=1) / np.sqrt(len(ic_fold))):+.2f}  "
          f"positive in {100 * (ic_fold > 0).mean():.0f}% of {len(ic_fold)} periods")
    print(f"   Pearson(s, x) = {np.corrcoef(s, x)[0, 1]:+.4f}, "
          f"Spearman = {spearmanr(s, x).correlation:+.4f}")
    out["level_vs_order"] = {"r2_raw": rows[0][1], "r2_lookahead": rows[1][1],
                             "r2_causal": rows[2][1], "b_full": b_full,
                             "overconfidence": 1 / b_full,
                             "ic_mean": float(ic_fold.mean())}

    # ---- B. what the ordering is worth ---------------------------------
    print(f"\n{'=' * 96}\nB.  what the ordering is worth, and the ceiling\n{'=' * 96}")
    mkt = M.per_fold(g, M.market(g))
    p_rank, x_rank = rank01_by_fold(g, g["p"].to_numpy(float)), rank01_by_fold(g, g["x"].to_numpy(float))
    rows = []
    for a in BLEND:
        blend = rank01_by_fold(g, (1 - a) * p_rank + a * x_rank)
        w = np.clip(2.0 * blend, 0.0, 2.0)
        d = M.paired(mkt["adj"], M.per_fold(g, w)["adj"])
        rows.append({"oracle_mixed_in": f"{100 * a:.0f}%",
                     "pooled_ic": spearmanr(blend, g["x"]).correlation,
                     "pooled_adj": M.pooled(g, w)["adj"], "vs_mkt": d["diff"]})
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    fit = np.polyfit(t["pooled_ic"], t["pooled_adj"], 1)
    print(f"""
   w = 2p on a PERFECTLY ordered signal is the ceiling of this mapping, and the
   blend traces the path to it. Fitting adj on IC over these points gives
   about adj = {fit[1]:.2f} + {fit[0]:.1f} * IC -- no saturation anywhere in
   the reachable range, so every bit of rank IC converts at roughly a constant
   rate. That is the whole case for the model layer being the only remaining
   lever: the mapping is a linear amplifier and it is already at its gain.""")
    out["ceiling"] = t.to_dict("records")

    # ---- C. the two traps ----------------------------------------------
    print(f"\n{'=' * 96}\nC.  E[x|s] under three normalisations -- reproduce the traps\n{'=' * 96}")
    variants = {"pooled, not demeaned (TRAP 1)": s,
                "demeaned within fold (TRAP 2: look-ahead)":
                    (gg["s_raw"] - gg.groupby("fold")["s_raw"].transform("mean")).to_numpy()}
    for wn in CAUSAL_DEMEAN:
        variants[f"demeaned causally, trailing {wn}"] = (
            gg["s_raw"] - gg.groupby("fold")["s_raw"].transform(
                lambda z: z.rolling(wn, min_periods=5).mean().shift(1))).to_numpy()

    print("   %-44s %9s %9s %11s %11s" %
          ("normalisation", "slope", "t", "middle 80%", "outer 20%"))
    rows = []
    for lab, v in variants.items():
        sl, tt, n = slope_t(v, x)
        f = np.isfinite(v)
        lo, hi = np.quantile(v[f], [0.1, 0.9])
        mid = slope_t(v[f & (v > lo) & (v < hi)], x[np.isfinite(v) & (v > lo) & (v < hi)])
        out_ = slope_t(v[f & ((v <= lo) | (v >= hi))],
                       x[np.isfinite(v) & ((v <= lo) | (v >= hi))])
        rows.append({"normalisation": lab, "slope": sl, "t": tt,
                     "mid80_slope": mid[0], "mid80_t": mid[1],
                     "tail20_slope": out_[0], "tail20_t": out_[1]})
        print("   %-44s %9.4f %+9.2f %11s %11s" %
              (lab, sl, tt, "%.3f(t=%.1f)" % (mid[0], mid[1]),
               "%.3f(t=%.1f)" % (out_[0], out_[1])))
    print("""
   THE SLOPE MOVES BY A FACTOR APPROACHING THREE across these rows and the
   middle-80% column flips from "no relationship" to "clear relationship". Any
   mapping shape argued from this table would be an artifact of the row the
   analyst stopped on. That is why the mapping is chosen by scoring positions.""")
    out["calibration"] = rows

    # ---- D. rank against magnitude -------------------------------------
    print(f"\n{'=' * 96}\nD.  does the magnitude add anything the rank does not\n{'=' * 96}")
    causal = variants[f"demeaned causally, trailing {CAUSAL_DEMEAN[-1]}"]
    sd = pd.Series(causal).rolling(252, min_periods=60).std().shift(1).to_numpy()
    mag = np.nan_to_num(np.clip(causal / np.where(sd > 0, sd, np.nan), -4, 4), nan=0.0)
    rows = []
    for lab, v in (("trailing percentile (frozen)", g["p"].to_numpy(float)),
                   ("causally standardised magnitude", mag)):
        q = pd.qcut(pd.Series(v), N_BINS, labels=False, duplicates="drop")
        tab = pd.DataFrame({"b": q, "x": x}).groupby("b")["x"].mean() * 252
        sp = pd.Series(tab.index).corr(pd.Series(tab.to_numpy()), method="spearman")
        rows.append({"input": lab, "spearman_bin_vs_return": sp, "n_bins": len(tab)})
        print(f"   {lab:<40} Spearman(bin, return) = {sp:+.3f}  over {len(tab)} bins")
    print(f"""
   A GAP UNDER ABOUT 0.30 IS NOT A GAP. Spearman on {N_BINS} bins has a standard
   error near 1/sqrt({N_BINS - 1}) = {1 / np.sqrt(N_BINS - 1):.2f}, so this
   comparison cannot separate them, and mapping_form.py settles it properly by
   scoring both end to end instead.""")
    out["rank_vs_magnitude"] = rows

    with open(os.path.join(OUT, "signal_diagnostics.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
