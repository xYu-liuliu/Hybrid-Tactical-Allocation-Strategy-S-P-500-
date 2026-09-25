"""
mapping_form.py -- what the mapping does with the order, and how hard it swings.

docs/METHOD.md 2.1 and 2.2, in one file because they read the same cached signals
and answer two halves of one question: given a number that says where today sits
among the last 252 days, what position does it become?

    A  THE CURVE. Seven forms from the literature -- linear, van der Waerden
       normal scores, GKX decile buckets, an extreme-decile spread, a dead band,
       a multi-horizon rank average, and a causal isotonic fit. Each gets its own
       causally-solved scale so that all reach the same realised volatility and
       none can win by taking more risk. `vs_lin` against `detectable` is the
       answer; the pre-registered bar is that a winner must beat linear by more
       than BELIEVE.

    B  THE AMPLITUDE. The position is `wbar + tau*z`, and tau is solved so `w`
       runs at a chosen ratio of market volatility. The metric taxes above 1.2
       linearly, so the question is whether to stop at the kink or cross it --
       read on the UNPENALISED Sharpe, which has the tax divided out and so
       measures the signal rather than the metric. Then the solver's own numbers:
       where it reads its history, how often it re-solves, its warm-up length and
       its fallback, and how seeds are combined.

NOTHING IS REFITTED. Both studies replay capacity_window_grid's cache at the
frozen cell, so the model layer is held exactly constant and only what is done
with the signal varies. The whole file runs in seconds.

    python mapping_form.py              # both studies
    python mapping_form.py --curves     # A only
    python mapping_form.py --params     # B only
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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

import capacity_window_grid as G
import mapping as MAP
import paths
import scoring as M

OUT = os.path.join(paths.PROBE_DIR, "mapping_form")


MU_WINDOW = MAP.MU_WINDOW        # 63, the frozen detrend
WINDOW = MAP.WINDOW              # 252, the frozen ranking window
WBAR = MAP.WBAR                  # 1.00
N_BUCKETS = 10                   # GKX use deciles
DEADBAND = 0.8                   # |u| below this carries no position
EXTREME = 0.1                    # top/bottom decile for the GKX-style rule
HORIZONS = (63, 126, 252)        # multi-horizon rank ensemble
REFIT = 84                       # isotonic refit cadence, one test block
BELIEVE = 0.10                   # pre-registered: smaller than this is noise
COST_BPS = (0, 1, 2, 5)          # charged on |dw|; the metric itself has no costs
SEEDS = (0, 1, 2)                # the bagging seeds the cache was written at

# ---- C, the detrend window, the ranking window and the coefficient ---------
MU_WINDOWS = [None, 21, 42, 63, 126, 252]        # None = no detrend
RANK_WINDOWS = [63, 126, 252, 504, None]         # None = expanding
COEFFS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2)
MIN_HIST = 40            # lower than mapping.MIN_HIST so the 63 window is usable
PAIRED_MUS = (21, 42)    # tested against the frozen 63
PAIRED_WINS = [126, 252, 504, None]

def adj_after_costs(w: np.ndarray, fr: np.ndarray, rf: np.ndarray,
                    bps: float, days: int = 252) -> float:
    """
    The competition metric with a linear cost of `bps` per unit of turnover.

    The cost comes out of the strategy return before the Sharpe and both
    penalties are computed, so a high-turnover rule is charged twice: once in
    the numerator and again through the return penalty it then triggers. Day
    one is charged against a starting position of 1.0 rather than nothing.
    """
    w = np.asarray(w, float)
    turn = np.abs(np.diff(np.concatenate([[1.0], w])))
    strat = rf * (1.0 - w) + w * fr - bps / 1e4 * turn
    ex = strat - rf
    sd = strat.std()
    if sd == 0:
        return 0.0
    mean_ex = np.prod(1.0 + ex) ** (1.0 / len(ex)) - 1.0
    sharpe = mean_ex / sd * np.sqrt(days)
    mkt_ex = np.prod(1.0 + (fr - rf)) ** (1.0 / len(fr)) - 1.0
    vol_pen = 1.0 + max(0.0, (sd * np.sqrt(days)) / (fr.std() * np.sqrt(days)) - 1.2)
    gap = max(0.0, (mkt_ex - mean_ex) * 100.0 * days)
    return float(sharpe / (vol_pen * (1.0 + gap ** 2 / 100.0)))

def percentile(s: np.ndarray, mu_window: int = MU_WINDOW,
               window: int = WINDOW, min_hist: int = 40) -> np.ndarray:
    """Detrended trailing percentile on the contiguous series (strictly causal)."""
    z = pd.Series(s)
    lvl = z.rolling(mu_window, min_periods=max(10, mu_window // 3)).mean().shift(1)
    dev = (z - lvl).to_numpy()
    p = np.full(len(dev), np.nan)
    for t in range(len(dev)):
        h = dev[max(0, t - window):t]
        h = h[np.isfinite(h)]
        if np.isfinite(dev[t]) and len(h) >= min_hist:
            p[t] = (np.sum(h < dev[t]) + 0.5) / (len(h) + 1.0)
    return p

def _std(z: np.ndarray) -> np.ndarray:
    z = np.nan_to_num(np.asarray(z, float), nan=0.0, posinf=0.0, neginf=0.0)
    sd = z.std()
    return z / sd if sd > 0 else z

def m_linear(p, s, x):
    return _std(2.0 * p - 1.0)

def m_normal_score(p, s, x):
    """van der Waerden. Phi^{-1} of the rank, clipped so a 1/253 tail is finite."""
    q = np.clip(np.nan_to_num(p, nan=0.5), 1e-3, 1 - 1e-3)
    return _std(norm.ppf(q))

def m_decile(p, s, x):
    """GKX-style bucketing: constant response inside each of ten buckets."""
    b = np.clip((np.nan_to_num(p, nan=0.5) * N_BUCKETS).astype(int), 0, N_BUCKETS - 1)
    return _std((b + 0.5) / N_BUCKETS * 2.0 - 1.0)

def m_extreme(p, s, x):
    """Only the top and bottom decile carry a view, as in a long-short spread."""
    q = np.nan_to_num(p, nan=0.5)
    return _std(np.where(q >= 1 - EXTREME, 1.0, np.where(q <= EXTREME, -1.0, 0.0)))

def m_deadband(p, s, x):
    """Soft threshold: no view until the rank is far enough from the middle."""
    u = 2.0 * np.nan_to_num(p, nan=0.5) - 1.0
    return _std(np.sign(u) * np.maximum(0.0, np.abs(u) - DEADBAND))

def m_multi(p, s, x):
    """Average the rank over three ranking windows before mapping."""
    ps = [percentile(s, MU_WINDOW, w) for w in HORIZONS]
    return _std(2.0 * np.nanmean(np.vstack(ps), axis=0) - 1.0)

def m_isotonic(p, s, x):
    """
    Causal isotonic regression of the realised return on the rank.

    Refitted every REFIT rows on all rows strictly before the block, then
    applied forward. This estimates a conditional mean -- the object that
    failed as a free calibration slope -- but inside a monotone hypothesis
    class, which cannot produce the sign flips that sank b_hat.
    """
    from sklearn.isotonic import IsotonicRegression
    q = np.nan_to_num(p, nan=0.5)
    out = np.zeros(len(q))
    for start in range(REFIT, len(q), REFIT):
        tr = slice(0, start)
        ok = np.isfinite(q[tr]) & np.isfinite(x[tr])
        if ok.sum() < 252:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(q[tr][ok], x[tr][ok])
        end = min(start + REFIT, len(q))
        out[start:end] = iso.predict(q[start:end])
    return _std(out)

CATALOGUE = {
    "linear": m_linear,
    "normal_score": m_normal_score,
    "decile10": m_decile,
    "extreme_decile": m_extreme,
    "deadband": m_deadband,
    "multi_horizon": m_multi,
    "isotonic": m_isotonic,
}

def score(g, z, m, wbar=MAP.WBAR, budget=MAP.BUDGET, window=MAP.TAU_WINDOW,
          min_hist=MAP.TAU_MIN_HIST, fallback=MAP.TAU_FALLBACK, fixed_tau=None):
    if fixed_tau is not None:
        tau = np.full(len(z), float(fixed_tau))
    else:
        tau = np.array([MAP.solve_tau(z[max(0, t - window):t], m[max(0, t - window):t],
                                      wbar, budget)
                        if t >= min_hist else np.nan for t in range(len(z))])
        tau = pd.Series(tau).ffill().fillna(fallback).to_numpy()
    w = np.clip(wbar + tau * z, 0.0, 2.0)
    return w, M.pooled(g, w), M.per_fold(g, w), float(np.nanmean(tau))

def row(lab, g, mkt, base_adj, w, pl, fold, tau_mean):
    vm = M.paired(mkt["adj"], fold["adj"])
    vb = M.paired(base_adj, fold["adj"]) if base_adj is not None else {}
    return {"config": lab, "pooled_adj": pl["adj"], "volr": pl["vol_ratio"],
            "vol_pen": pl["vol_pen"], "sharpe": pl["sharpe"], "tau_mean": tau_mean,
            "vs_mkt": vm["diff"], "vs_mkt_t": vm["t"],
            "vs_base": vb.get("diff"), "vs_base_t": vb.get("t"),
            "detect": vb.get("detectable"), "turnover": fold["turnover"].mean(),
            "pinned": pl["pinned"]}

def study_curves(g, s, x, m, mkt) -> dict:

    p = percentile(s)
    print(f"{len(g)} contiguous rows, {g['fold'].nunique()} periods, "
          f"detrend {MU_WINDOW}, rank window {WINDOW}, rank available "
          f"{100 * np.isfinite(p).mean():.0f}%")
    print(f"matched budget: wbar={WBAR}, tau solved causally to volr={MAP.BUDGET}")
    print(f"pre-registered: a winner needs > {BELIEVE:+.2f} against linear\n")

    res: Dict[str, dict] = {}
    for name, fn in CATALOGUE.items():
        z = fn(p, s, x)
        tau = np.array([MAP.solve_tau(z[max(0, t - MAP.TAU_WINDOW):t],
                                      m[max(0, t - MAP.TAU_WINDOW):t], WBAR)
                        if t >= MAP.TAU_MIN_HIST else np.nan for t in range(len(z))])
        tau = pd.Series(tau).ffill().fillna(MAP.TAU_FALLBACK).to_numpy()
        w = np.clip(WBAR + tau * z, 0.0, 2.0)
        pl, t = M.pooled(g, w), M.per_fold(g, w)
        res[name] = {"pooled": pl, "fold": t, "w": w,
                     "active": float(np.mean(np.abs(z) > 1e-9))}

    base = res["linear"]["fold"]["adj"]
    rows = []
    for name, r in res.items():
        vm, vl = M.paired(mkt["adj"], r["fold"]["adj"]), M.paired(base, r["fold"]["adj"])
        rows.append({"map": name, "pooled_adj": r["pooled"]["adj"],
                     "volr": r["pooled"]["vol_ratio"],
                     "vs_mkt": vm["diff"], "vs_mkt_t": vm["t"],
                     "vs_lin": vl["diff"], "vs_lin_t": vl["t"],
                     "detect": vl["detectable"], "win_vs_lin": vl["win_frac"],
                     "turnover": r["fold"]["turnover"].mean(),
                     "pos_sd": r["pooled"]["pos_sd"],
                     "pinned": r["pooled"]["pinned"],
                     "active_frac": r["active"]})
    tab = pd.DataFrame(rows).sort_values("pooled_adj", ascending=False)
    print(tab.round(3).to_string(index=False))
    print(f"\nmarket pooled adj = {M.pooled(g, M.market(g))['adj']:.3f}")

    # ---- transaction costs ---------------------------------------------
    print(f"\n{'=' * 96}\nWITH TRANSACTION COSTS -- the only axis these maps still "
          f"differ on\n{'=' * 96}")
    print("The metric has no costs in it, so 0bps is what gets scored and the "
          "rest is a\nrobustness read. Costs cannot make a worse map better; "
          "they only order the ties\nby how much book each one turns over.")
    fr, rf = g["fr"].to_numpy(float), g["rf"].to_numpy(float)
    print(f"\n   {'map':<16}{'turnover':>10}" + "".join(f"{b}bps".rjust(9) for b in COST_BPS))
    cost_rows = []
    for name, r in list(res.items()) + [("market", {"w": np.ones(len(g))})]:
        w = r["w"]
        vals = [adj_after_costs(w, fr, rf, b) for b in COST_BPS]
        cost_rows.append({"map": name, "turnover": float(np.abs(np.diff(w)).mean()),
                          **{f"bps_{b}": v for b, v in zip(COST_BPS, vals)}})
        print(f"   {name:<16}{np.abs(np.diff(w)).mean():>10.3f}"
              + "".join(f"{v:>9.3f}" for v in vals))
    ct = pd.DataFrame(cost_rows)
    for b in (COST_BPS[len(COST_BPS) // 2], COST_BPS[-1]):
        order = ct.sort_values(f"bps_{b}", ascending=False)["map"].tolist()
        print(f"   ranking at {b}bps: " + " > ".join(order))

    best = tab.iloc[0]
    print(f"\nbest is {best['map']} at {best['pooled_adj']:.3f}, "
          f"{best['vs_lin']:+.3f} against linear (detect {best['detect']:.3f}). ", end="")
    print("Believable." if best["vs_lin"] > BELIEVE else
          "Below the pre-registered threshold, so linear stands.")
    return {"table": tab.to_dict("records"), "costs": ct.to_dict("records")}


def study_parameters(g, s, x, m, mkt) -> dict:
    mkt_pooled = M.pooled(g, M.market(g))["adj"]
    mkt_pooled = M.pooled(g, M.market(g))["adj"]

    p0 = percentile(s)
    z0 = np.sqrt(3.0) * (2.0 * np.nan_to_num(p0, nan=0.5) - 1.0)
    w0, pl0, f0, t0 = score(g, z0, m)
    base = f0["adj"]

    print(f"{len(g)} rows, {g['fold'].nunique()} periods.  "
          f"baseline = detrend {MAP.MU_WINDOW} / rank {MAP.WINDOW} / "
          f"wbar {MAP.WBAR} / budget {MAP.BUDGET}")
    print(f"baseline pooled adj {pl0['adj']:.3f}, market {mkt_pooled:.3f}, "
          f"tau_mean {t0:.3f}")
    print(f"pre-registered: a move under {BELIEVE:+.2f} against baseline is noise\n")

    out: Dict[str, List[dict]] = {}

    # ---- 1. BUDGET ------------------------------------------------------
    print("=" * 100)
    print("1. BUDGET -- the penalty past the kink is LINEAR, so overshooting may pay")
    print("=" * 100)
    rows = []
    for b in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0):
        w, pl, f, tm = score(g, z0, m, budget=b)
        rows.append(row(f"budget={b}", g, mkt, base, w, pl, f, tm))
    for ft in (0.2, 0.4, 0.6, 0.8):
        w, pl, f, tm = score(g, z0, m, fixed_tau=ft)
        rows.append(row(f"[control] fixed tau={ft}", g, mkt, base, w, pl, f, tm))
    t1 = pd.DataFrame(rows)
    print(t1.round(3).to_string(index=False))
    out["budget"] = t1.to_dict("records")

    # ---- 2. MIN_HIST ----------------------------------------------------
    print(f"\n{'=' * 100}\n2. MIN_HIST -- the percentile's warm-up bar\n{'=' * 100}")
    rows = []
    for mh in (20, 40, 60, 100, 150):
        p = percentile(s, min_hist=mh)
        z = np.sqrt(3.0) * (2.0 * np.nan_to_num(p, nan=0.5) - 1.0)
        w, pl, f, tm = score(g, z, m)
        r = row(f"min_hist={mh}", g, mkt, base, w, pl, f, tm)
        r["rank_avail"] = float(np.isfinite(p).mean())
        rows.append(r)
    t2 = pd.DataFrame(rows)
    print(t2.round(3).to_string(index=False))
    out["min_hist"] = t2.to_dict("records")

    # ---- 3. TAU_MIN_HIST and TAU_FALLBACK -------------------------------
    print(f"\n{'=' * 100}\n3. TAU_MIN_HIST / TAU_FALLBACK -- assumed negligible, "
          f"now measured\n{'=' * 100}")
    rows = []
    for mh in (60, 120, 252):
        w, pl, f, tm = score(g, z0, m, min_hist=mh)
        rows.append(row(f"tau_min_hist={mh}", g, mkt, base, w, pl, f, tm))
    for fb in (0.0, 0.3, 0.6, 1.0):
        w, pl, f, tm = score(g, z0, m, fallback=fb)
        rows.append(row(f"tau_fallback={fb}", g, mkt, base, w, pl, f, tm))
    t3 = pd.DataFrame(rows)
    print(t3.round(3).to_string(index=False))
    out["tau_warmup"] = t3.to_dict("records")

    # ---- 4. tau_source --------------------------------------------------
    print(f"\n{'=' * 100}\n4. tau_source -- rolling across folds (deployment) vs "
          f"a 302-row block at the fold start (walk-forward)\n{'=' * 100}")
    rows = []
    for win in (252, 302, 504, 756):
        w, pl, f, tm = score(g, z0, m, window=win)
        rows.append(row(f"rolling {win}, refit daily", g, mkt, base, w, pl, f, tm))
    # The two walk-forward arms. Both see only a 302-row block ending at the
    # fold start -- the same length and position as the real calibration block
    # -- and they differ only in how often tau is re-solved inside the fold.
    # That cadence is the whole of make_mapper's tau_source="cal", and holding
    # tau for all 84 days lets the realised ratio drift off the kink.
    idx_of = g.groupby("fold").indices
    folds = sorted(g["fold"].unique())

    def block_tau(z, series_m, daily: bool, nrows: int = 302):
        t = np.full(len(z), np.nan)
        for fd in folds:
            ii = idx_of[fd]
            i0, lo = int(ii[0]), max(0, int(ii[0]) - nrows)
            if daily:
                for i in ii:
                    if i - lo >= MAP.TAU_MIN_HIST:
                        t[i] = MAP.solve_tau(z[lo:i], series_m[lo:i], MAP.WBAR, MAP.BUDGET)
            elif i0 - lo >= MAP.TAU_MIN_HIST:
                t[ii] = MAP.solve_tau(z[lo:i0], series_m[lo:i0], MAP.WBAR, MAP.BUDGET)
        return pd.Series(t).ffill().fillna(MAP.TAU_FALLBACK).to_numpy()

    # z0 is the contiguous reconstruction; z_cal is the cached p_trail_dm,
    # which is warm-started from each fold's own s_cal the way make_mapper is
    z_cal = np.sqrt(3.0) * (2.0 * g["p"].to_numpy(float) - 1.0)
    for z_in, zlab in ((z0, "contiguous p"), (z_cal, "s_cal-warmed p")):
        for daily, dlab in ((False, "one tau per fold"), (True, "tau re-solved daily")):
            tb = block_tau(z_in, m, daily)
            w = np.clip(MAP.WBAR + tb * z_in, 0.0, 2.0)
            lab = f"block 302, {zlab}, {dlab}"
            if z_in is z_cal and daily:
                lab += "  (= make_mapper)"
            rows.append(row(lab, g, mkt, base, w, M.pooled(g, w), M.per_fold(g, w),
                            float(np.nanmean(tb))))
    t4 = pd.DataFrame(rows)
    print(t4.round(3).to_string(index=False))
    print("""
  READ vol_pen, NOT vs_base, ON THESE ROWS. Holding tau for a whole fold is a
  tie on the per-fold paired contrast and still loses on the pooled score,
  because the entire cost is volatility drifting past the 1.2 kink -- which
  per-fold scoring cannot see and the real metric can. The cadence is the
  finding here; the block length barely matters.""")
    out["tau_source"] = t4.to_dict("records")

    # ---- 5. seed aggregation --------------------------------------------
    print(f"\n{'=' * 100}\n5. seed aggregation -- mean of percentiles vs "
          f"percentile of the mean\n{'=' * 100}")
    per_seed = {sd: pd.read_csv(G.sig_file(G.REF_W, 117, sd)).sort_values(["fold", "pos"])
                for sd in SEEDS}
    rows = []
    # (a) percentile of the mean signal -- the baseline
    rows.append({**row("p(mean s)  [baseline]", g, mkt, base, w0, pl0, f0, t0)})
    # (b) mean of the per-seed percentiles, each computed on its own series
    ps = [percentile(d["s_raw"].to_numpy(float)) for d in per_seed.values()]
    p_mean = np.nanmean(np.vstack(ps), axis=0)
    z = np.sqrt(3.0) * (2.0 * np.nan_to_num(p_mean, nan=0.5) - 1.0)
    w, pl, f, tm = score(g, z, m)
    rows.append(row("mean of p(s_i)", g, mkt, base, w, pl, f, tm))
    # (c) the cached p_trail_dm averaged across seeds (warm-started from s_cal)
    z = np.sqrt(3.0) * (2.0 * g["p"].to_numpy(float) - 1.0)
    w, pl, f, tm = score(g, z, m)
    rows.append(row("mean of cached p_trail_dm (s_cal warm-up)", g, mkt, base, w, pl, f, tm))
    # (d) single seeds, to show the spread bagging is removing
    for sd in SEEDS:
        p = percentile(per_seed[sd]["s_raw"].to_numpy(float))
        z = np.sqrt(3.0) * (2.0 * np.nan_to_num(p, nan=0.5) - 1.0)
        w, pl, f, tm = score(g, z, m)
        rows.append(row(f"[single] seed {sd}", g, mkt, base, w, pl, f, tm))
    t5 = pd.DataFrame(rows)
    print(t5.round(3).to_string(index=False))
    out["seed_agg"] = t5.to_dict("records")

    # ---- verdict --------------------------------------------------------
    allr = pd.concat([t1, t2, t3, t4, t5], ignore_index=True)
    allr = allr[allr["vs_base"].notna()]
    best = allr.loc[allr["vs_base"].idxmax()]
    print(f"\n{'=' * 100}\nVERDICT\n{'=' * 100}")
    print(f"  baseline pooled adj {pl0['adj']:.3f}  (market {mkt_pooled:.3f})")
    print(f"  best move: {best['config']}  {best['vs_base']:+.3f} "
          f"(t={best['vs_base_t']:+.2f}, detect {best['detect']:.3f})")
    print("  " + ("ABOVE the pre-registered threshold -- worth acting on."
                  if best["vs_base"] > BELIEVE else
                  "below the pre-registered threshold -- the mapping stands as frozen."))
    return {"baseline": pl0, **out}


def detrended(s: np.ndarray, mu_window: Optional[int]) -> np.ndarray:
    """s minus its own trailing mean; mu_window=None leaves it alone."""
    z = pd.Series(np.asarray(s, float))
    if mu_window is None:
        return z.to_numpy()
    lvl = z.rolling(mu_window, min_periods=max(10, mu_window // 3)).mean().shift(1)
    return (z - lvl).to_numpy()

def rank_z(s: np.ndarray, mu_window: Optional[int], window: Optional[int],
           min_hist: int = MIN_HIST) -> np.ndarray:
    """
    The mapping's input: sqrt(3) * (2p - 1), unit variance for a uniform p.

    window=None ranks against everything seen so far. Every window is strictly
    trailing, so row t never enters its own reference distribution.
    """
    dev = detrended(s, mu_window)
    p = np.full(len(dev), np.nan)
    for t in range(len(dev)):
        h = dev[max(0, t - window):t] if window else dev[:t]
        h = h[np.isfinite(h)]
        if np.isfinite(dev[t]) and len(h) >= min_hist:
            p[t] = (np.sum(h < dev[t]) + 0.5) / (len(h) + 1.0)
    return np.sqrt(3.0) * (2.0 * np.nan_to_num(p, nan=0.5) - 1.0)

def solved_tau(z: np.ndarray, mkt: np.ndarray) -> np.ndarray:
    t = np.array([MAP.solve_tau(z[max(0, i - MAP.TAU_WINDOW):i],
                                mkt[max(0, i - MAP.TAU_WINDOW):i])
                  if i >= MAP.TAU_MIN_HIST else np.nan for i in range(len(z))])
    return pd.Series(t).ffill().fillna(MAP.TAU_FALLBACK).to_numpy()

def evaluate(g, mkt_fold, z, tau):
    w = np.clip(MAP.WBAR + tau * z, MAP.MIN_POS, MAP.MAX_POS)
    pooled, fold = M.pooled(g, w), M.per_fold(g, w)
    d = M.paired(mkt_fold["adj"], fold["adj"])
    return {"pooled_adj": pooled["adj"], "vs_mkt": d["diff"], "t": d["t"],
            "volr": pooled["vol_ratio"], "vol_pen": pooled["vol_pen"],
            "turnover": fold["turnover"].mean(), "expo": pooled["exposure"],
            "tau_mean": float(np.nanmean(tau)), "_fold": fold}

def _lab(v) -> str:
    return "none/expanding" if v is None else str(v)

def run_grid(g, s, m, mkt_fold) -> pd.DataFrame:
    print("=" * 104)
    print("1 + 2.  detrend window x ranking window")
    print("        cell = pooled_adj / per-fold vs market / t")
    print("=" * 104)
    header = "   %-14s" % "detrend\\rank" + "".join("%-22s" % _lab(w) for w in RANK_WINDOWS)
    print(header)
    rows = []
    for mu in MU_WINDOWS:
        line = "   %-14s" % _lab(mu)
        for win in RANK_WINDOWS:
            z = rank_z(s, mu, win)
            r = evaluate(g, mkt_fold, z, solved_tau(z, m))
            rows.append({"mu_window": _lab(mu), "rank_window": _lab(win),
                         **{k: v for k, v in r.items() if not k.startswith("_")}})
            line += "%-22s" % ("%.3f / %+.3f / %+.2f" % (r["pooled_adj"], r["vs_mkt"], r["t"]))
        print(line)
    print("""
  THE STRUCTURE IS ANTI-DIAGONAL, which is the finding. No detrend works only
  with a short ranking window; a long detrend works only with a short window
  too; and the usable plateau is a short detrend with a ranking window of 126
  or more. Zero removals is not significant. Two removals -- short detrend AND
  short window -- is the worst usable cell, because the second pass takes the
  signal out with the level.""")
    return pd.DataFrame(rows)

def run_coef(g, s, m, mkt_fold) -> pd.DataFrame:
    print(f"\n{'=' * 104}")
    print(f"3.  the linear coefficient, fixed values against the solver")
    print(f"    z from the frozen detrend {MAP.MU_WINDOW} / rank {MAP.WINDOW}")
    print("=" * 104)
    z = rank_z(s, MAP.MU_WINDOW, MAP.WINDOW)
    rows = []
    print("   %-26s %10s %10s %8s %8s %9s" %
          ("c  (w = 1 + c*z)", "pooled", "vs_mkt", "t", "volr", "turnover"))
    for c in COEFFS:
        r = evaluate(g, mkt_fold, z, np.full(len(z), float(c)))
        rows.append({"coef": c, "solved": False,
                     **{k: v for k, v in r.items() if not k.startswith("_")}})
        print("   %-26s %10.3f %+10.3f %+8.2f %8.3f %9.3f" %
              (f"fixed {c}", r["pooled_adj"], r["vs_mkt"], r["t"], r["volr"], r["turnover"]))
    tau = solved_tau(z, m)
    r = evaluate(g, mkt_fold, z, tau)
    rows.append({"coef": float(np.nanmean(tau)), "solved": True,
                 **{k: v for k, v in r.items() if not k.startswith("_")}})
    print("   %-26s %10.3f %+10.3f %+8.2f %8.3f %9.3f   <- FROZEN" %
          (f"solved (mean {np.nanmean(tau):.3f})", r["pooled_adj"], r["vs_mkt"],
           r["t"], r["volr"], r["turnover"]))
    t = pd.DataFrame(rows)
    fixed = t[~t["solved"]]
    best = fixed.loc[fixed["pooled_adj"].idxmax()]
    print(f"""
  THE SOLVER SHOULD BEAT EVERY CONSTANT, and if it does the answer to "what is
  the coefficient" is not a number. Its mean sits between the two best fixed
  values, so any gain is adaptivity rather than level. Best fixed here is
  c={best['coef']} at {best['pooled_adj']:.3f}; solved is {r['pooled_adj']:.3f}.

  t FALLS AND volr RISES MONOTONICALLY IN c. Those are the same fact: a larger
  coefficient buys more point estimate and more variance with it. Choosing on
  the pooled level alone would walk straight up that trade-off.""")
    return t

def run_paired(g, s, m, mkt_fold) -> pd.DataFrame:
    print(f"\n{'=' * 104}")
    print(f"PAIRED: detrend {PAIRED_MUS} against the frozen {MAP.MU_WINDOW}, "
          f"at matched ranking windows")
    print("=" * 104)
    folds = {(mu, w): evaluate(g, mkt_fold, rank_z(s, mu, w),
                               solved_tau(rank_z(s, mu, w), m))["_fold"]
             for mu in (MAP.MU_WINDOW,) + tuple(PAIRED_MUS) for w in PAIRED_WINS}
    rows = []
    print("   %-14s" % "rank window" + "".join("%-26s" % f"{mu} - {MAP.MU_WINDOW}"
                                               for mu in PAIRED_MUS))
    for w in PAIRED_WINS:
        line = "   %-14s" % _lab(w)
        for mu in PAIRED_MUS:
            d = M.paired(folds[(MAP.MU_WINDOW, w)]["adj"], folds[(mu, w)]["adj"])
            rows.append({"mu_window": mu, "rank_window": _lab(w), **d})
            line += "%-26s" % ("%+.3f  t=%+.2f  dt=%.3f" %
                               (d["diff"], d["t"], d["detectable"]))
        print(line)

    print("\n   sign test across the four ranking windows:")
    for mu in PAIRED_MUS:
        sg = [np.sign(r["diff"]) for r in rows if r["mu_window"] == mu]
        sub = [r["diff"] for r in rows if r["mu_window"] == mu]
        print(f"      mu={mu}:  {sum(1 for v in sg if v > 0)}/{len(sg)} positive, "
              f"mean {np.mean(sub):+.3f}")
    print(f"""
  EVERY CONTRAST INSIDE ITS OWN dt IS THE EXPECTED RESULT and it is what keeps
  {MAP.MU_WINDOW} frozen. A consistent sign across the four windows is NOT
  independent evidence -- the four share the same folds and the same data and
  differ only in how the rank was taken, so they are four looks at one thing.
  Adopt a different detrend only on a contrast that clears {BELIEVE:+.2f} on
  its own.""")
    return pd.DataFrame(rows)

def study_axes(g, s, m, mkt) -> dict:
    """
    docs/METHOD.md 2.6 from the dev side: the detrend window and the ranking
    window as a grid, the linear coefficient against the solver, and the paired
    contrasts that keep 63 frozen on this span.

    THE POINT OF THE GRID IS THAT THE TWO WINDOWS DO THE SAME JOB. Both remove
    the slow component of the signal's level -- detrending subtracts it, and a
    short ranking window absorbs it, because inside a short window the level has
    barely moved. Sweeping them one at a time would attribute the whole effect to
    whichever was swept first.

    AND THE POINT OF THE PAIRED SECTION IS A NEGATIVE RESULT. Dev puts 21 at the
    top of every ranking-window column, and the contrast against 63 is inside its
    own detectable by a factor of twenty. Dev cannot separate the short lengths;
    detrend_sweep.py shows that valid can.
    """
    return {"grid": run_grid(g, s, m, mkt).to_dict("records"),
            "coef": run_coef(g, s, m, mkt).to_dict("records"),
            "paired": run_paired(g, s, m, mkt).to_dict("records")}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    flags = {"curves", "params", "axes"} & {a.lstrip("-") for a in sys.argv[1:]}
    want = flags or {"curves", "params", "axes"}

    g = M.load_signals()
    s, x = g["s_raw"].to_numpy(float), g["x"].to_numpy(float)
    m = (g["fr"] - g["rf"]).to_numpy(float)
    mkt = M.per_fold(g, M.market(g))
    print(f"{len(g)} contiguous rows, {g['fold'].nunique()} periods, "
          f"market pooled adj {M.pooled(g, M.market(g))['adj']:.3f}")
    print(f"frozen: detrend {MU_WINDOW} / rank {WINDOW} / wbar {WBAR} / "
          f"budget {MAP.BUDGET}")
    print(f"pre-registered: a move under {BELIEVE:+.2f} is noise")
    print(f"sections {sorted(want)}")
    print()

    out: Dict[str, object] = {"believe": BELIEVE, "wbar": WBAR,
                              "mu_window": MU_WINDOW, "window": WINDOW,
                              "market_pooled": M.pooled(g, M.market(g))["adj"]}
    if "curves" in want:
        print("=" * 100)
        print("A.  THE CURVE -- seven forms at a matched risk budget")
        print("=" * 100)
        out["curves"] = study_curves(g, s, x, m, mkt)
    if "params" in want:
        print()
        print("=" * 100)
        print("B.  THE AMPLITUDE -- the budget, and every number in the solver")
        print("=" * 100)
        out["parameters"] = study_parameters(g, s, x, m, mkt)
    if "axes" in want:
        print()
        print("=" * 100)
        print("C.  THE TWO WINDOWS -- detrend x ranking, and the coefficient")
        print("=" * 100)
        out["axes"] = study_axes(g, s, m, mkt)

    with open(os.path.join(OUT, "mapping_form.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print()
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
