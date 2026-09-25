"""
level_refit.py -- can a model read its own output level at the moment it is refit?

docs/METHOD.md 2.6. The offset a fit carries is a property of that fit, and the
fit exists the moment the model is refitted -- so the obvious repair is to read
the level off then and subtract it. This asks whether anything available at that
moment correlates with the level the predictions actually take on the block
ahead.

ONE SCRIPT, TWO MODELS. `--model tree` and `--model ridge` run the identical
experiment; only the fit differs. That is the point: 2.4 uses the contrast between
them to justify shipping the Ridge, and a reader can confirm nothing else changed
between the two columns.

THE ESTIMATORS, each correlated against the realised level of the next block:

    cal_current      fit train[:60%] -> predict train[60%:]. What ships today:
                     the calibration block comes from a DIFFERENT fit than the
                     test predictions, because capacity_data.bagged_signal splits
                     the window at INNER_FRAC.
    cal_matched      fit train[:-84] -> predict train[-84:]. The repair that
                     defect invites: 89% of the window, so very nearly the same
                     fit, with a short tail held out.
    insample_all     the model predicting its own training rows.
    insample_recent  the same, restricted to the last 84. ZERO fit difference --
                     the same model, the same parameters -- which makes it the
                     discriminator. If even this is flat, no reading taken from
                     the model can recover the level and the line is closed.
    y_train_mean     mean of the training labels. No fit.
    y_recent_mean    mean of the last 84 training labels. No fit.
    prev_te          the previous block's realised level. No fit.

WHERE THE TWO MODELS CAN DIFFER. A 91-leaf tree fits its own training rows
closely, so its in-sample mean is forced onto its training labels' mean -- on the
tree they agreed to three decimals with identical dispersion. A Ridge at
alpha=1000 does not, so if those two separate, the in-sample route carries
something for a linear model that it could not carry for a tree.

EVERY ESTIMATE IS DEPLOYED THROUGH AN EXPANDING CALIBRATION, never raw. An
estimate correlated at r must be shrunk by about r before being subtracted or it
injects more level noise than it removes; at fold k the slope and intercept come
from folds 1..k-1 only, so the shrinkage is not borrowed knowledge. The raw
subtraction is reported beside it to show what skipping that step costs.

Development folds only, 69 of them, so a correlation has a standard error near
0.120. The tree needs about nineteen minutes; the Ridge is closed form and needs
a few.

    python level_refit.py --model ridge            # plan
    python level_refit.py --model ridge --commit
    python level_refit.py --model tree --commit --bags 3   # cheaper, noisier
    python level_refit.py --model ridge --score-only
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
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

import capacity_data as cd
import paths
import ranking as RK
import scoring as SC
import split_data as SD
from hull_probe import INNER_FRAC, TEST_WINDOW, TRAIN_WINDOW, _prep_linear, feature_cols

OUT_ROOT = os.path.join(paths.PROBE_DIR, "level_refit")
CAL_TAIL = TEST_WINDOW     # the calibration model gives up exactly one block
CAPACITY = 117             # frozen; only the tree uses it
N_BAG = 7                  # only the tree uses it
SEED = 0
RIDGE_ALPHA = 1000.0
ESTIMATORS = ("cal_current", "cal_matched", "insample_all", "insample_recent",
              "y_train_mean", "y_recent_mean", "prev_te")
REFERENCE = ("frozen_flat", "center_causal", "center_TRUE", "p_window_LOOKAHEAD")


def out_dir(model: str) -> str:
    return os.path.join(OUT_ROOT, model)


# ------------------------------------------------------------------- fits
def fit_predict(model: str, df, cols, y, train_idx, predict_sets,
                n_bag: int) -> List[np.ndarray]:
    """
    Fit on `train_idx` ONCE, then predict each array in `predict_sets`.

    One fit serving several prediction sets is not an optimisation detail: the
    in-sample level and the test-block level have to come from the same model, or
    the comparison between them means nothing.
    """
    ok = np.isfinite(y[train_idx])
    tr = train_idx[ok]
    if model == "tree":
        p = cd.tree_params(len(tr), CAPACITY)
        X, yy = df.iloc[tr][cols], y[tr]
        ms = [LGBMRegressor(**{**p, "random_state": SEED * 1000 + b}).fit(X, yy)
              for b in range(n_bag)]
        return [np.mean([m.predict(df.iloc[idx][cols]) for m in ms], axis=0)
                for idx in predict_sets]
    # Ridge: one closed-form fit, and _prep_linear wants the prediction rows
    # alongside the training rows, so they go in as one block and come apart after
    flat = np.concatenate(predict_sets)
    Xtr, Xpr = _prep_linear(df.iloc[train_idx][cols], df.iloc[flat][cols])
    pred = Ridge(alpha=RIDGE_ALPHA).fit(Xtr[ok], y[train_idx][ok]).predict(Xpr)
    out, at = [], 0
    for idx in predict_sets:
        out.append(pred[at:at + len(idx)])
        at += len(idx)
    return out


def fit_fold(model: str, df, cols, y, f, n_bag: int) -> dict:
    """One fold: the full model, the two calibration models, and every level."""
    tr, te = f.train, f.test
    cut = int(len(tr) * INNER_FRAC)
    cur_tr, cur_cal = tr[:cut], tr[cut:]
    mat_tr, mat_cal = tr[:-CAL_TAIL], tr[-CAL_TAIL:]

    s_te, s_in = fit_predict(model, df, cols, y, tr, [te, tr], n_bag)
    s_cur, = fit_predict(model, df, cols, y, cur_tr, [cur_cal], n_bag)
    s_mat, = fit_predict(model, df, cols, y, mat_tr, [mat_cal], n_bag)

    return {"fold": int(te[0]), "s_te": s_te,
            "cal_current": float(np.nanmean(s_cur)),
            "cal_matched": float(np.nanmean(s_mat)),
            "insample_all": float(np.nanmean(s_in)),
            "insample_recent": float(np.nanmean(s_in[-CAL_TAIL:])),
            "y_train_mean": float(np.nanmean(y[tr])),
            "y_recent_mean": float(np.nanmean(y[tr[-CAL_TAIL:]])),
            "n_cur_train": int(len(cur_tr)), "n_mat_train": int(len(mat_tr)),
            "n_full_train": int(len(tr))}


# ------------------------------------------------------------- deployment
def expanding_calibrate(est: np.ndarray, truth: np.ndarray,
                        min_hist: int = 12) -> np.ndarray:
    """
    Shrink each estimate using ONLY earlier folds.

    At fold k, least squares of truth on est over folds 0..k-1 gives a slope and
    an intercept, applied to fold k. An estimator correlated at r produces a
    slope near r once the scales are matched, which is where the shrinkage comes
    from -- and because the fit never sees fold k, the shrinkage is not borrowed.
    Folds before min_hist fall back to the running mean of truth, which centres
    nothing on average and is the honest neutral choice.
    """
    out = np.full(len(est), np.nan)
    for k in range(len(est)):
        h = np.arange(k)
        ok = h[np.isfinite(est[h]) & np.isfinite(truth[h])]
        if len(ok) < min_hist or not np.isfinite(est[k]):
            out[k] = float(np.nanmean(truth[ok])) if len(ok) else 0.0
            continue
        a, b = np.polyfit(est[ok], truth[ok], 1)
        out[k] = a * est[k] + b
    return out


def corr(a, b) -> dict:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.ptp(a[ok]) == 0:
        return {"n": int(ok.sum()), "pearson": np.nan, "r2": np.nan}
    r = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return {"n": int(ok.sum()), "pearson": r, "r2": r * r}


# ------------------------------------------------------------------- main
def main() -> None:
    argv = sys.argv
    model = argv[argv.index("--model") + 1] if "--model" in argv else "ridge"
    if model not in ("tree", "ridge"):
        raise SystemExit("--model must be tree or ridge")
    commit, score_only = "--commit" in argv, "--score-only" in argv
    n_bag = int(argv[argv.index("--bags") + 1]) if "--bags" in argv else N_BAG
    OUT = out_dir(model)
    cache = os.path.join(OUT, "fold_levels.csv")
    os.makedirs(OUT, exist_ok=True)

    df, folds = SD.load_span("train")
    lo = SD.span_offset("train")          # caches carry FEATURE TABLE rows
    cols = feature_cols(df)
    y = (df["forward_returns"] - df["risk_free_rate"]).to_numpy(float)

    print("=" * 106)
    print(f"LEVEL REFIT ({model.upper()}) -- can the model read its own level?")
    print("=" * 106)
    print(f"   development only: {len(folds)} folds x {TEST_WINDOW} rows, "
          f"{len(cols)} columns, window {TRAIN_WINDOW}")
    print(f"   fit: " + (f"LightGBM, {CAPACITY} leaves, bag {n_bag}, seed {SEED}"
                         if model == "tree" else f"Ridge(alpha={RIDGE_ALPHA:.0f})"))
    print(f"   calibration split: current fits {int(TRAIN_WINDOW * INNER_FRAC)} "
          f"rows and predicts {TRAIN_WINDOW - int(TRAIN_WINDOW * INNER_FRAC)}; "
          f"matched fits {TRAIN_WINDOW - CAL_TAIL} and predicts {CAL_TAIL}")
    print(f"   standard error on a {len(folds)}-fold correlation is about "
          f"{1 / np.sqrt(len(folds)):.3f}")
    have = os.path.exists(cache)
    if not (commit or score_only):
        print(f"\n   {len(folds)} folds x 3 fits"
              + (f" x {n_bag} bags; about nineteen minutes" if model == "tree"
                 else "; closed form, a few minutes"))
        print(f"   cache {'EXISTS' if have else 'absent'}")
        print("\n   --plan only. Re-run with --commit.")
        return

    if score_only or have:
        if not have:
            raise SystemExit(f"--score-only but {cache} is missing")
        lv = pd.read_csv(cache)
        sig = pd.read_csv(os.path.join(OUT, "signal.csv"))
        print(f"\n   cache: {len(lv)} folds, {len(sig)} signal rows")
    else:
        recs, parts = [], []
        t0 = time.time()
        for j, f in enumerate(folds):
            r = fit_fold(model, df, cols, y, f, n_bag)
            s = r.pop("s_te")
            parts.append(pd.DataFrame({"fold": r["fold"] + lo,
                                       "pos": f.test + lo, "s": s}))
            r["te_mean"] = float(np.nanmean(s))
            recs.append(r)
            if j % 20 == 0 or j == len(folds) - 1:
                el = time.time() - t0
                print(f"   fold {j + 1}/{len(folds)}  {el / 60:.1f} min elapsed, "
                      f"{el / (j + 1) * (len(folds) - j - 1) / 60:.1f} left")
        lv, sig = pd.DataFrame(recs), pd.concat(parts, ignore_index=True)
        lv.to_csv(cache, index=False)
        sig.to_csv(os.path.join(OUT, "signal.csv"), index=False)

    sig = sig.sort_values("pos")
    idx, blk = sig["pos"].to_numpy(int), sig["fold"].to_numpy(int)
    s = sig["s"].to_numpy(float)
    x = y[idx - lo]
    lv = lv.sort_values("fold").reset_index(drop=True)
    lv["prev_te"] = lv["te_mean"].shift(1)
    truth = lv["te_mean"].to_numpy(float)

    print(f"\n{'=' * 106}")
    print("A.  does the estimate track the realised block level?")
    print("=" * 106)
    cr = [{"estimator": e, **corr(lv[e].to_numpy(float), truth),
           "est_sd": float(np.nanstd(lv[e].to_numpy(float))),
           "truth_sd": float(np.nanstd(truth))} for e in ESTIMATORS]
    print(pd.DataFrame(cr).round(4).to_string(index=False))

    fold_pos = {int(f): k for k, f in enumerate(lv["fold"].to_numpy(int))}
    order = np.array([fold_pos[b] for b in blk])
    bm = RK.block_stat(s, blk, np.nanmean)
    ranks: Dict[str, np.ndarray] = {
        "frozen_flat": RK.trailing_rank(RK.detrend(s)),
        "center_causal": RK.trailing_rank(s - RK.block_expanding_mean(s, blk)),
        "center_TRUE": RK.trailing_rank(s - bm),
        "p_window_LOOKAHEAD": RK.window_rank_lookahead(s, blk),
    }
    for e in ESTIMATORS:
        raw = lv[e].to_numpy(float)
        ranks[f"{e}__calibrated"] = RK.trailing_rank(
            s - expanding_calibrate(raw, truth)[order])
        ranks[f"{e}__raw"] = RK.trailing_rank(s - raw[order])

    ic_rows = []
    for rname, p in ranks.items():
        ok = np.isfinite(p)
        if ok.sum() < 200:
            continue
        v, tt, pf = SC.block_ic(blk[ok], p[ok], x[ok])
        ic_rows.append({"rank": rname, "avail": float(ok.mean()),
                        "block_ic": v, "t": tt, "pos_frac": pf})
    ic = pd.DataFrame(ic_rows)
    ref = {r["rank"]: r["block_ic"] for r in ic_rows if r["rank"] in REFERENCE}
    floor, ceil_ = ref.get("frozen_flat", np.nan), ref.get("center_TRUE", np.nan)
    ic["vs_causal"] = ic["block_ic"] - ref.get("center_causal", np.nan)
    ic["frac_of_ceiling"] = (ic["block_ic"] - floor) / (ceil_ - floor)

    print(f"\n{'=' * 106}")
    print("B.  block_ic after recentring -- the bar is center_causal, not the floor")
    print("=" * 106)
    head = ic[ic["rank"].isin(REFERENCE)].set_index("rank").reindex(
        [r for r in REFERENCE if r in set(ic["rank"])]).reset_index()
    body = ic[~ic["rank"].isin(REFERENCE)].sort_values("block_ic", ascending=False)
    print(pd.concat([head, body]).round(4).to_string(index=False))

    print(f"""
Reading it.

  cal_current AGAINST cal_matched IS THE EXPERIMENT. They differ in one thing,
  the fraction of the window the calibration model is fitted on. If cal_matched
  correlates while cal_current stays flat, the wrong-fit defect is real and the
  repair works; if both are flat, repairing it changes nothing.

  insample_recent IS THE DISCRIMINATOR. Zero fit difference, so if it is flat, no
  reading taken from the model can recover the level and this line is closed for
  this model.

  insample_all AGAINST y_train_mean IS WHERE THE MODELS CAN DIFFER. A 91-leaf
  tree's in-sample mean is its training labels' mean; a Ridge at alpha=1000 is
  shrunk far enough that it need not be.

  est_sd AGAINST truth_sd MATTERS AS MUCH AS THE CORRELATION. An estimator
  pointing the right way at a fifth of the amplitude removes a fifth of what it
  should, and a correlation of r captures only r-squared of the level variance.

  frac_of_ceiling PUTS EVERY ROW ON ONE SCALE: 0 is no centring, 1 is the exact
  block mean.

written to """ + OUT)

    with open(os.path.join(OUT, "level_refit.json"), "w") as f:
        json.dump({"model": model,
                   "config": {"folds": len(folds), "inner_frac": INNER_FRAC,
                              "cal_tail": CAL_TAIL, "n_bag": n_bag,
                              "capacity": CAPACITY, "alpha": RIDGE_ALPHA},
                   "correlations": cr, "block_ic": ic.to_dict("records"),
                   "fold_levels": lv.to_dict("records")}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
