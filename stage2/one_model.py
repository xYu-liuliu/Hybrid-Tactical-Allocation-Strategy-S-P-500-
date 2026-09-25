"""
one_model.py -- does the output level belong to the fit, or to the period?

docs/METHOD.md 2.4 and 2.5. Hold the mapping completely fixed and change only
how the signal is produced, over identical rows:

    walk_forward         refit every 84 rows, 756-row window    many fits
    train_once_756        one fit,            756-row window    one fit, little data
    train_once_all_dev    one fit,          ~6600-row window    one fit, much data

walk_forward against train_once_756 isolates REFITTING with the window held
fixed. train_once_756 against train_once_all_dev isolates TRAINING DATA with the
number of fits held at one. Nothing about the mapping moves, so anything that
moves is the signal.

TWO RANKINGS, AND THE SECOND IS THE LOAD-BEARING ONE:

    frozen_flat   a 63-day mean removed, then a trailing 252-day percentile.
                  Deployable, and it spans several fits.
    p_window      the full within-block rank. LOOK-AHEAD, and immune to the
                  block's level BY CONSTRUCTION -- it ranks inside the block. So
                  if IT rises as the number of fits falls, the model is ordering
                  better and the mapping has nothing to do with it.

AND THE PERSISTENCE CONTRAST, which is what 2.5 rests on: a block's realised
level against the previous block's, within one fit and across a refit. Same rows,
same period, same algorithm; the only difference is whether a refit sat between
them.

ONE SCRIPT, TWO MODELS, ONE GEOMETRY. `--model tree` and `--model ridge` differ
only in the fit. This matters more here than elsewhere: before the merge the two
models were measured on DIFFERENT fold grids -- the tree on the holdout cache at
embargo 10, the Ridge at embargo 0 -- so 2.4's contrast was between two
geometries as well as two models. Both arms now read the same splits at embargo
0 and fit fresh.

SCORED ON valid ONLY. 2.4 and 2.5 are valid-span findings and test is not touched
here. The one-shot arms train on rows strictly before valid's first scored row,
which is why the frame reaches back into development -- `load_through` says so
out loud.

    python one_model.py --model ridge            # plan
    python one_model.py --model ridge --commit    # closed form, a minute
    python one_model.py --model tree --commit     # bagged LightGBM, a few minutes
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
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

import capacity_data as cd
import paths
import ranking as RK
import scoring as SC
import split_data as SD
from hull_probe import (
    START_AT, TEST_WINDOW, TRAIN_WINDOW, _prep_linear, feature_cols,
)

OUT_ROOT = os.path.join(paths.PROBE_DIR, "one_model")
SPAN = "valid"
PURGE, EMBARGO = 1, 0
CAPACITY = 117            # frozen; only the tree uses it
N_BAG = 7                 # only the tree uses it
SEED = 0
RIDGE_ALPHA = 1000.0
PROTOCOLS = ("walk_forward", "train_once_756", "train_once_all_dev")


def out_dir(model: str) -> str:
    return os.path.join(OUT_ROOT, model)


def fit_predict(model: str, df, cols, y, train_idx, test_idx,
                n_bag: int) -> np.ndarray:
    """One fit on train_idx, predictions on test_idx. Bagged for the tree."""
    ok = np.isfinite(y[train_idx])
    if model == "tree":
        tr = train_idx[ok]
        p = cd.tree_params(len(tr), CAPACITY)
        X, yy = df.iloc[tr][cols], y[tr]
        return np.mean([LGBMRegressor(**{**p, "random_state": SEED * 1000 + b})
                        .fit(X, yy).predict(df.iloc[test_idx][cols])
                        for b in range(n_bag)], axis=0)
    Xtr, Xte = _prep_linear(df.iloc[train_idx][cols], df.iloc[test_idx][cols])
    return Ridge(alpha=RIDGE_ALPHA).fit(Xtr[ok], y[train_idx][ok]).predict(Xte)


def drift(s: np.ndarray, blk: np.ndarray) -> dict:
    """Between-block level spread against within-block spread, and its persistence."""
    g = pd.Series(s).groupby(blk)
    means = g.mean().to_numpy()
    within = float(g.std(ddof=1).mean())
    out = {"blocks": int(len(means)),
           "sd_of_block_means": float(np.std(means, ddof=1)),
           "mean_within_block_sd": within,
           "drift_ratio": float(np.std(means, ddof=1) / within) if within > 0 else np.nan}
    if len(means) > 3:
        out["lag1_pearson"] = float(np.corrcoef(means[:-1], means[1:])[0, 1])
        out["lag1_spearman"] = float(spearmanr(means[:-1], means[1:]).correlation)
        out["n_pairs"] = int(len(means) - 1)
    return out


def main() -> None:
    argv = sys.argv
    model = argv[argv.index("--model") + 1] if "--model" in argv else "ridge"
    if model not in ("tree", "ridge"):
        raise SystemExit("--model must be tree or ridge")
    commit = "--commit" in argv
    n_bag = int(argv[argv.index("--bags") + 1]) if "--bags" in argv else N_BAG
    OUT = out_dir(model)
    os.makedirs(OUT, exist_ok=True)

    df, folds, lo = SD.load_through(SPAN)
    cols = feature_cols(df)
    y = (df["forward_returns"] - df["risk_free_rate"]).to_numpy(float)
    idx = np.concatenate([f.test for f in folds])
    blk = np.concatenate([np.full(len(f.test), int(f.test[0])) for f in folds])
    x = y[idx]

    # The two one-shot arms train once, ending where walk_forward's first fold
    # ends, so the three protocols differ in ONE thing: how much history the fit
    # sees and whether it is ever refreshed.
    tr_end = int(idx[0]) - (PURGE + EMBARGO)
    trains = {"train_once_756": np.arange(max(0, tr_end - TRAIN_WINDOW), tr_end),
              "train_once_all_dev": np.arange(START_AT - lo, tr_end)}

    print("=" * 100)
    print(f"ONE MODEL ({model.upper()}) -- is the level the fit's, or the period's?")
    print("=" * 100)
    print(f"   span {SPAN}: {len(folds)} blocks, {len(idx)} scored rows, "
          f"feature-table rows {idx.min() + lo}..{idx.max() + lo}")
    print(f"   fit: " + (f"LightGBM, {CAPACITY} leaves, bag {n_bag}"
                         if model == "tree" else f"Ridge(alpha={RIDGE_ALPHA:.0f})")
          + f", {len(cols)} columns, purge {PURGE}, embargo {EMBARGO}")
    print(f"   walk_forward       {len(folds):>3} fits, {TRAIN_WINDOW} train rows each")
    for k, tr in trains.items():
        print(f"   {k:<18} {1:>3} fit,  {len(tr)} train rows, ending at row {tr_end + lo}")
    assert all(not (set(t.tolist()) & set(idx.tolist())) for t in trains.values()), \
        "one-shot training overlaps scored rows"
    print(f"   leakage check: training ends at row {tr_end + lo}, first scored row "
          f"is {int(idx[0]) + lo}")
    if not commit:
        n = len(folds) + 2
        print(f"\n   {n} fits" + (f" x {n_bag} bags" if model == "tree" else
                                  ", closed form")
              + ".\n   --plan only. Re-run with --commit.")
        return

    sig: Dict[str, np.ndarray] = {}
    t0 = time.time()
    sig["walk_forward"] = np.concatenate(
        [fit_predict(model, df, cols, y, f.train, f.test, n_bag) for f in folds])
    for k, tr in trains.items():
        sig[k] = fit_predict(model, df, cols, y, tr, idx, n_bag)
    print(f"\n   fitted in {time.time() - t0:.0f}s")

    ic_rows, dr_rows = [], []
    for name in PROTOCOLS:
        s = sig[name]
        for rname, p in (("frozen_flat", RK.trailing_rank(RK.detrend(s))),
                         ("p_window_LOOKAHEAD", RK.window_rank_lookahead(s, blk))):
            ok = np.isfinite(p)
            v, tt, pf = SC.block_ic(blk[ok], p[ok], x[ok])
            ic_rows.append({"protocol": name, "rank": rname,
                            "avail": float(ok.mean()), "blocks": len(folds),
                            "block_ic": v, "t": tt, "pos_frac": pf})
        dr_rows.append({"protocol": name, **drift(s, blk)})

    ic = pd.DataFrame(ic_rows)
    print(f"\n{'=' * 100}\nA.  block_ic on {SPAN}\n{'=' * 100}")
    print(ic.pivot_table(index="protocol", columns="rank", values="block_ic")
          .reindex(PROTOCOLS)[["frozen_flat", "p_window_LOOKAHEAD"]]
          .round(4).to_string())
    print("\n   detail")
    print("   " + ic.round(4).to_string(index=False).replace("\n", "\n   "))

    print(f"\n{'=' * 100}")
    print("B.  the block level: how far it moves, and whether it persists")
    print("=" * 100)
    print(pd.DataFrame(dr_rows).round(4).to_string(index=False))

    print(f"""
Reading it.

  A IS THE LADDER. Both columns should move together as the number of fits falls.
  The right-hand one is load-bearing: a within-block rank cannot be helped by
  removing a block level it never sees, so a rise there is the model ordering
  better, not the mapping working better.

  B IS THE CONTRAST 2.5 RESTS ON. lag1 is the previous block's realised level
  against the next one's. Under walk_forward a refit sits between them; under
  either one-shot arm it does not. A positive lag1 on one fit against a flat one
  under refitting says the level belongs to the fit.

  n_pairs IS SMALL and should be read as such: {len(folds)} blocks give
  {len(folds) - 1} pairs, a standard error near {1 / np.sqrt(max(1, len(folds) - 1)):.2f}.
  The ladder in A is the stronger evidence; the persistence contrast corroborates
  it rather than carrying it.

  drift_ratio IS sd(block means) OVER mean(within-block sd). Near 1 means the
  between-block component is as large as everything happening inside a block,
  which is why removing the block level matters at all.

written to """ + OUT)

    pd.DataFrame({"pos": idx + lo, **{k: v for k, v in sig.items()}}) \
        .to_csv(os.path.join(OUT, "signals.csv"), index=False)
    with open(os.path.join(OUT, "one_model.json"), "w") as f:
        json.dump({"model": model, "span": SPAN, "blocks": len(folds),
                   "rows": len(idx), "train_end_row": int(tr_end + lo),
                   "first_scored_row": int(idx[0] + lo),
                   "config": {"capacity": CAPACITY, "n_bag": n_bag,
                              "alpha": RIDGE_ALPHA, "embargo": EMBARGO},
                   "block_ic": ic_rows, "drift": dr_rows}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
