"""
ridge_signal.py -- fit the Ridge signal once, then say what kind of signal it is.

ONE CACHE, WRITTEN ONCE, READ BY EVERYTHING. The fit and the diagnosis live in
the same file so that the thing diagnosed is the thing shipped. Nothing else in
the project fits a Ridge for its signal -- `pipeline.py` fits its own because it
is deliberately self-contained, and `audit_config.py` checks the two agree on
alpha, preprocessing and folds.

    kaggle_hull/probe/ridge_signal/ridge_{span}.csv     columns: pos, s

`pos` is the row in the feature table, so the three files concatenate in order
into one contiguous out-of-sample series -- which is what a trailing percentile
needs -- without any span's rows appearing twice.

THE MODEL. Ridge(alpha=1000) on all 1132 columns, rolling 756-row window, refit
every 84 rows, purge 1 and embargo 0. Features come from the span's own slice via
split_data.load_span, so fitting `train` cannot see `valid` and fitting `valid`
cannot see `test`.

THE DIAGNOSIS, on development folds only. The whole position rule keeps the
model's ORDER and throws its MAGNITUDE away, which is only correct if the
magnitude is genuinely broken. Three tests, and none of them uses a mapping
parameter, because none has been derived at this point in the chain:

    A  the level. R2_oos against zero and against the trailing mean; the
       Mincer-Zarnowitz slope with Newey-West errors, whose reciprocal is how
       much the prediction overstates itself; then the same slope re-estimated
       from earlier rows only and applied forward, which is the shrinkage a
       deployed system could have had.
    B  the order. Per-block Spearman with its t, and the decile table.
    C  does the magnitude add anything to the order? Inside each 84-row block --
       a unit the fold geometry fixes, not a tuned window -- `r` is the normal
       score of the within-block rank, which is the order and nothing else, and
       `m` is (s - block mean) / block sd, which keeps the shape. `y` is
       regressed on each and on both. The coefficient on `m` in the JOINT model
       is the answer; the two solo rows settle nothing, because r and m are
       strongly collinear by construction.

The LightGBM is scored beside it at the identical fold geometry, so the two
models can be compared without the older grid signal_diagnostics.py used.

    python ridge_signal.py                    # plan
    python ridge_signal.py --commit           # fit what is missing, then diagnose
    python ridge_signal.py --commit --refit   # discard the cache and fit again
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
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge

import paths
import split_data as SD
from hull_probe import TARGET_COL, _prep_linear, feature_cols

OUT = os.path.join(paths.PROBE_DIR, "ridge_signal")
TREE_DIR = os.path.join(paths.PROBE_DIR, "embargo_capacity", "signals")
TREE_SEEDS = (0, 1)
TREE_CAPACITY = 117

RIDGE_ALPHA = 1000.0      # 1132 columns on 756 rows needs real shrinkage
MZ_LAGS = 5               # Newey-West lags; the label overlaps by one day
RESCALE_MIN = 504         # rows before a causal slope may be estimated
TRAIL_MEAN = 252          # the Campbell-Thompson benchmark's window
SPANS = ("dev", "valid", "test")
SPAN_TO_SLICE = {"dev": "train", "valid": "valid", "test": "test"}


# ------------------------------------------------------------------ signal
def sig_file(span: str) -> str:
    return os.path.join(OUT, f"ridge_{span}.csv")


def fit_span(span: str) -> pd.DataFrame:
    """
    Walk-forward Ridge on one span, from that span's own slice.

    _prep_linear guards the standardisation: a training column that is nearly
    constant otherwise divides by a near-zero scale and sends test predictions
    into the billions.
    """
    slice_name = SPAN_TO_SLICE[span]
    df, folds = SD.load_span(slice_name)
    lo = SD.span_offset(slice_name)
    cols = feature_cols(df)
    y = (df["forward_returns"] - df["risk_free_rate"]).to_numpy(float)

    pos, out = [], []
    for f in folds:
        Xtr, Xte = _prep_linear(df.iloc[f.train][cols], df.iloc[f.test][cols])
        ok = np.isfinite(y[f.train])
        out.append(Ridge(alpha=RIDGE_ALPHA).fit(Xtr[ok], y[f.train][ok]).predict(Xte))
        pos.append(f.test + lo)
    return pd.DataFrame({"pos": np.concatenate(pos), "s": np.concatenate(out)})


def load_signal(span: str) -> pd.DataFrame:
    """The cached predictions for one span. This is the only Ridge cache."""
    p = sig_file(span)
    if not os.path.exists(p):
        raise SystemExit(f"not found: {p}. Run ridge_signal.py --commit.")
    return pd.read_csv(p)


def load_series(spans=SPANS):
    """
    The spans' predictions as one contiguous series, in time order.

    Returns (pos, s, span_of). The caches hold scored rows only, so nothing is
    duplicated where the underlying slices overlap.
    """
    parts = [load_signal(k) for k in spans]
    pos = np.concatenate([d["pos"].to_numpy(int) for d in parts])
    assert (np.diff(pos) > 0).all(), "the cached spans are not in time order"
    return (pos, np.concatenate([d["s"].to_numpy(float) for d in parts]),
            np.concatenate([np.full(len(d), k) for k, d in zip(spans, parts)]))


def load_tree_series(spans=SPANS):
    """
    The bagged LightGBM predictions, averaged over seeds, as one contiguous
    series over `spans`. Returns None when the cache is absent.

    The tree is not the shipped model -- docs/METHOD.md 2.4 drops it -- but the
    reason it is dropped is a contrast measured against it, so the comparison
    arm has to be loadable. embargo_capacity.py writes this cache.
    """
    files = {k: [os.path.join(TREE_DIR, f"e0_c{TREE_CAPACITY}_{k}_seed{j}.csv")
                 for j in TREE_SEEDS] for k in spans}
    if not all(os.path.exists(q) for k in spans for q in files[k]):
        return None
    return np.concatenate([
        np.mean([pd.read_csv(q)["s"].to_numpy(float) for q in files[k]], axis=0)
        for k in spans])


# ------------------------------------------------------------------- level
def r2_oos(y, pred, bench) -> float:
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(bench)
    sse = float(np.sum((y[ok] - pred[ok]) ** 2))
    sst = float(np.sum((y[ok] - bench[ok]) ** 2))
    return float(1.0 - sse / sst) if sst > 0 else np.nan


def _nw(y: np.ndarray, X: np.ndarray, lags: int = MZ_LAGS) -> dict:
    """OLS with Newey-West errors; a coefficient and t for each column of X."""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    yy, XX = y[ok], np.column_stack([np.ones(int(ok.sum())), X[ok]])
    xtx_inv = np.linalg.inv(XX.T @ XX)
    beta = xtx_inv @ (XX.T @ yy)
    e = yy - XX @ beta
    S = (XX * e[:, None]).T @ (XX * e[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        A = (XX[L:] * e[L:, None]).T @ (XX[:-L] * e[:-L, None])
        S += w * (A + A.T)
    V = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.diag(V))
    sst = float(np.sum((yy - yy.mean()) ** 2))
    return {"n": int(ok.sum()),
            "r2": float(1.0 - np.sum(e ** 2) / sst) if sst > 0 else np.nan,
            "beta": beta[1:].tolist(), "t": (beta[1:] / se[1:]).tolist()}


def causal_slope(y, s, min_hist: int = RESCALE_MIN) -> np.ndarray:
    """
    The Mincer-Zarnowitz slope re-estimated at every row from earlier rows only.

    The shrinkage a deployed system could have applied. Using the full-sample
    slope instead would be a look-ahead correction and would prove nothing.
    """
    out = np.full(len(s), np.nan)
    ok = np.isfinite(y) & np.isfinite(s)
    cy, cs = np.where(ok, y, 0.0), np.where(ok, s, 0.0)
    k_ = np.cumsum(ok)
    sx, sy = np.cumsum(cs), np.cumsum(cy)
    sxx, sxy = np.cumsum(cs * cs), np.cumsum(cs * cy)
    for t in range(min_hist, len(s)):
        k = k_[t - 1]
        if k < min_hist:
            continue
        vx = sxx[t - 1] - sx[t - 1] ** 2 / k
        if vx > 0:
            out[t] = (sxy[t - 1] - sx[t - 1] * sy[t - 1] / k) / vx
    return out


# ---------------------------------------------------------------- ordering
def block_ic(block, v, x) -> dict:
    ok = np.isfinite(v) & np.isfinite(x)
    per = []
    for b in np.unique(block):
        m = (block == b) & ok
        if m.sum() > 10 and np.ptp(v[m]) > 0:
            per.append(spearmanr(v[m], x[m]).correlation)
    per = np.array([q for q in per if np.isfinite(q)])
    se = per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else np.nan
    return {"blocks": int(len(per)), "block_ic": float(per.mean()),
            "t": float(per.mean() / se) if se else np.nan,
            "pos_frac": float((per > 0).mean()),
            "pooled_ic": float(spearmanr(v[ok], x[ok]).correlation)}


def deciles(s, x, k: int = 10) -> pd.DataFrame:
    ok = np.isfinite(s) & np.isfinite(x)
    q = pd.qcut(pd.Series(s[ok]).rank(method="first"), k, labels=False)
    g = pd.DataFrame({"decile": q + 1, "x": x[ok]}).groupby("decile")["x"]
    return pd.DataFrame({"n": g.size(), "mean_excess_ann_%": g.mean() * 252 * 100,
                         "hit_rate": g.apply(lambda v: float((v > 0).mean()))})


def within_block_encodings(s, blk) -> Dict[str, np.ndarray]:
    """
    The same signal encoded twice inside each block: order only, and order plus
    magnitude.

    `r` is invariant to any monotone transform of s -- the order and nothing
    else. `m` keeps the shape, standardised by the block's own spread. Neither is
    causal; both use the whole block. This measures what the signal CONTAINS, not
    what a rule could capture.
    """
    r = np.full(len(s), np.nan)
    m = np.full(len(s), np.nan)
    for b in np.unique(blk):
        i = np.where(blk == b)[0]
        v = s[i]
        ok = np.isfinite(v)
        if ok.sum() < 10:
            continue
        q = pd.Series(v[ok]).rank(method="average").to_numpy()
        r[i[ok]] = norm.ppf((q - 0.5) / len(q))
        sd = float(np.nanstd(v[ok]))
        if sd > 0:
            m[i[ok]] = (v[ok] - float(np.nanmean(v[ok]))) / sd
    return {"r": r, "m": m}


# ------------------------------------------------------------------- main
def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    commit = "--commit" in sys.argv
    refit = "--refit" in sys.argv

    sf = SD.scoring_folds()
    print("=" * 104)
    print("RIDGE SIGNAL -- fit once, then say what kind of signal it is")
    print("=" * 104)
    print(f"   Ridge(alpha={RIDGE_ALPHA:.0f}), rolling 756, refit every 84, "
          f"purge 1, embargo 0")
    for k in SPANS:
        have = os.path.exists(sig_file(k))
        print(f"   {k:<6} {len(sf[k]):>3} folds, "
              f"{sum(len(f.test) for f in sf[k]):>5} scored rows, "
              f"from {SPAN_TO_SLICE[k]}_processed.csv"
              f"   {'cached' if have and not refit else 'to fit'}")
    if not commit:
        print("\n   closed-form fits, a few minutes for all three."
              "\n   --plan only. Re-run with --commit.")
        return

    for k in SPANS:
        if os.path.exists(sig_file(k)) and not refit:
            continue
        t0 = time.time()
        fit_span(k).to_csv(sig_file(k), index=False)
        print(f"   fitted {k} in {time.time() - t0:.0f}s")

    # ---- the diagnosis, development only ------------------------------
    pos, s_ridge, _ = load_series(("dev",))
    blk = np.concatenate([np.full(len(f.test), int(f.test[0])) for f in sf["dev"]])
    df_dev = SD.load_span("train")[0]
    lo = SD.span_offset("train")
    fr = df_dev["forward_returns"].to_numpy(float)[pos - lo]
    rf = df_dev["risk_free_rate"].to_numpy(float)[pos - lo]
    y = fr - rf

    sigs: Dict[str, np.ndarray] = {"ridge": s_ridge}
    t_sig = load_tree_series(("dev",))
    if t_sig is not None:
        if len(t_sig) == len(pos):
            sigs["tree"] = t_sig
        else:
            print(f"   tree signal is {len(t_sig)} rows against {len(pos)}; skipped")

    print(f"\n   diagnosis on development only: {len(sf['dev'])} blocks, "
          f"{len(pos)} rows, table rows {pos.min()}..{pos.max()}")

    trail = pd.Series(y).rolling(TRAIL_MEAN, min_periods=60).mean().shift(1).to_numpy()
    zero = np.zeros(len(y))
    lvl, ordr, inc = [], [], []
    for name, s in sigs.items():
        mz = _nw(y, s[:, None])
        b = causal_slope(y, s)
        lvl.append({"signal": name, "sd_pred": float(np.nanstd(s)),
                    "sd_truth": float(np.nanstd(y)),
                    "R2_vs_zero": r2_oos(y, s, zero),
                    "R2_vs_trailing_mean": r2_oos(y, s, trail),
                    "MZ_slope": mz["beta"][0], "MZ_t_NW": mz["t"][0],
                    "overconfidence_1_over_b": 1.0 / mz["beta"][0] if mz["beta"][0] else np.nan,
                    "causal_slope_median": float(np.nanmedian(b)),
                    "R2_after_causal_rescale": r2_oos(y, b * s, zero)})
        ordr.append({"signal": name, **block_ic(blk, s, y)})
        enc = within_block_encodings(s, blk)
        for lab, X, names in (("r only", enc["r"][:, None], ["r"]),
                              ("m only", enc["m"][:, None], ["m"]),
                              ("both", np.column_stack([enc["r"], enc["m"]]), ["r", "m"])):
            f = _nw(y, X)
            row = {"signal": name, "model": lab, "n": f["n"], "r2": f["r2"]}
            for j, nm in enumerate(names):
                row[f"beta_{nm}"] = f["beta"][j]
                row[f"t_{nm}"] = f["t"][j]
            inc.append(row)

    print(f"\n{'=' * 104}\nA.  THE LEVEL\n{'=' * 104}")
    lv = pd.DataFrame(lvl)
    print(lv[["signal", "sd_pred", "sd_truth", "R2_vs_zero",
              "R2_vs_trailing_mean"]].round(6).to_string(index=False))
    print()
    print(lv[["signal", "MZ_slope", "MZ_t_NW", "overconfidence_1_over_b",
              "causal_slope_median", "R2_after_causal_rescale"]]
          .round(4).to_string(index=False))

    print(f"\n{'=' * 104}\nB.  THE ORDER\n{'=' * 104}")
    print(pd.DataFrame(ordr).round(4).to_string(index=False))
    for name, s in sigs.items():
        print(f"\n   --- {name}: deciles of the raw prediction ---")
        print("   " + deciles(s, y).round(2).to_string().replace("\n", "\n   "))

    print(f"\n{'=' * 104}")
    print("C.  DOES THE MAGNITUDE ADD ANYTHING TO THE ORDER?")
    print("=" * 104)
    print("   r = normal score of the within-block rank (order only)")
    print("   m = (s - block mean) / block sd          (order and magnitude)")
    print("   no mapping parameter is used; the 84-row block is fold geometry\n")
    ic = pd.DataFrame(inc)
    print(ic.reindex(columns=["signal", "model", "n", "r2", "beta_r", "t_r",
                             "beta_m", "t_m"])
          .to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    print(f"""
Reading it.

  A IS READ AS A PAIR. R2_vs_zero negative says the magnitude is worse than
  predicting nothing; MZ_slope strictly between 0 and 1 says there is a linear
  signal underneath that the prediction overstates by 1/b. Neither alone says
  "the order is informative and the level is not".

  R2_after_causal_rescale IS THE CONFIRMATION. Shrinking by a slope estimated
  only from earlier rows should move R2 from clearly negative to near or above
  zero. If it does, the information was in the direction all along and the scale
  was the whole problem.

  t_m IN THE "both" ROW IS WHAT JUSTIFIES THE DESIGN. r and m encode the same
  signal and are strongly collinear, so neither solo row settles anything. A
  small t on m in the joint model says the magnitude adds nothing once the order
  is known, and the rest of the mapping is then free to be built on the rank.

  THE TREE IS BESIDE IT at the identical fold geometry. If the two behave the
  same way, the design transfers to the model that ships; where they differ, the
  difference is itself evidence about which model to ship.

written to """ + OUT)

    with open(os.path.join(OUT, "ridge_signal.json"), "w") as f:
        json.dump({"alpha": RIDGE_ALPHA, "blocks": len(sf["dev"]), "rows": len(pos),
                   "level": lvl, "order": ordr, "increment": inc},
                  f, indent=2, default=float)


if __name__ == "__main__":
    main()
