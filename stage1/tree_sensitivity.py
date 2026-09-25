"""
tree_sensitivity.py -- the TREE parameters that capacity and window did not cover.

WHAT IS ALREADY SETTLED. num_leaves and min_child_samples were swept jointly as
capacity and the window was swept with them: the frozen point is rolling 756 at
target 117, about 92 actual leaves. n_estimators is settled too, by staged
predictions from a single 1500-tree fit over all 69 periods:

    n_trees      50     100     250     500    1000    1500
    spread_ann 1.236   1.228   1.229   1.260   1.260   1.255
    vs 250     +0.008  -0.001     -    +0.032  +0.031  +0.027
    t          +0.07   -0.02      -    +0.70   +0.56   +0.50

Flat across a 30x range -- even 50 trees matches 250. The boosting dimension is
saturated, unlike leaves where 4 -> 48 was worth +0.442.

WHAT IS LEFT is learning_rate, colsample_bytree, reg_lambda and subsample.

THIS IS A SENSITIVITY SWEEP, NOT A SEARCH, and the reason is arithmetic rather
than taste. A single evaluation has a standard error near 0.167 annualised, so
the best of N configurations sits above the truth by +0.26 at N=10 and +0.38 at
N=50 -- the latter being 85% of the largest real effect this project has found.
One parameter moves at a time, every value is reported, and the reading is the
SHAPE:

    flat        that parameter is not where anything is. Leave it.
    monotone with the default at an edge
                the one actionable shape: extend the grid that way.
    peaked      the peak is grid noise. The median is the estimate.

SUBSAMPLE IS NOT INDEPENDENT of capacity: min_child_samples is derived from
subsample * W / target, so moving it can move the realised leaf count and turn
a subsample test into a capacity test. At target 117 it happens not to -- the
realised count stays 117 from subsample 0.5 to 1.0 -- and main() prints the
realised leaves for every row so the check is visible rather than assumed.

PRECISION. The default is one bag and two seeds, about 7 minutes per
configuration against 74 for the grid's settings. Bagging was measured not to
matter for this statistic (per-period variance 1.003e-05 at bag=1 against
1.117e-05 at bag=15, inside the standard error of a variance on 48 periods),
but two seeds instead of three does cost resolution. Read detectable_ann on
every row: a null wider than the effect you would care about is not a null.
Anything that does move is worth re-running at the grid's bag=7 and three seeds
before it is believed.

    python tree_sensitivity.py                     # all four parameters
    python tree_sensitivity.py learning_rate       # one of them
    python tree_sensitivity.py --plan              # cost, then stop
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
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import paths
import capacity_data as cd
import capacity_window_grid as G
import split_data as SD
from hull_probe import (
    EMBARGO, HOLDOUT_FRAC, HOLDOUT_GAP, INNER_FRAC, START_AT, TARGET_COL,
    TEST_WINDOW, TREE, feature_cols,
)
from wfo import make_folds
from lightgbm import LGBMRegressor

WINDOW, TARGET = 756, 117            # frozen by capacity_window_grid
N_BAG = 1
SEEDS = (0, 1)
OUT = os.path.join(paths.PROBE_DIR, "tree_sensitivity")

# one parameter at a time, defaults in the middle where the grid allows
GRID: Dict[str, List] = {
    "learning_rate":    [0.01, 0.02, 0.05, 0.10, 0.20],
    "colsample_bytree": [0.3, 0.5, 0.7, 0.9, 1.0],
    "reg_lambda":       [0.1, 1.0, 5.0, 20.0, 100.0],
    "subsample":        [0.5, 0.7, 0.85, 1.0],
    "n_estimators":     [50, 250, 500],     # settled already; cheap to confirm
}
DEFAULTS = {k: TREE[k] for k in GRID}


def params_for(param: str, value) -> dict:
    """
    Frozen capacity, one parameter overridden.

    min_child_samples has to be recomputed from the overridden subsample, or
    the cell would silently change its realised leaf count and the row would be
    testing capacity under a subsample label.
    """
    sub = value if param == "subsample" else TREE["subsample"]
    per_tree = max(1, int(WINDOW * sub))
    p = {**TREE, param: value, "subsample": sub,
         "num_leaves": TARGET,
         "min_child_samples": max(G.MIN_CHILD_FLOOR, per_tree // TARGET)}
    return p


def realised_for(param: str, value) -> int:
    p = params_for(param, value)
    per_tree = int(WINDOW * p["subsample"])
    return min(p["num_leaves"], per_tree // p["min_child_samples"])


def folds_for():
    """
    The development fold grid, positioned on train_processed.csv.

    make_folds is called with dev_end as the series length, so no fold can reach
    the holdout; SD.remap_rows then shifts the result onto the slice.
    """
    n = SD.table_rows()
    dev_end = n - int(round(n * HOLDOUT_FRAC)) - HOLDOUT_GAP
    return SD.remap_rows(
        make_folds(dev_end, WINDOW, TEST_WINDOW, embargo=EMBARGO,
                   start_at=START_AT), "train")


def signal(cols: Sequence[str], p: dict, n_bag: int = N_BAG):
    cols = list(cols)

    def f(df, train_idx, test_idx, seed):
        y = df[TARGET_COL].to_numpy(float)
        cut = int(len(train_idx) * INNER_FRAC)
        inner_tr, inner_cal = train_idx[:cut], train_idx[cut:]
        ok_i, ok_t = np.isfinite(y[inner_tr]), np.isfinite(y[train_idx])

        X_in, y_in = df.iloc[inner_tr[ok_i]][cols], y[inner_tr[ok_i]]
        X_cal = df.iloc[inner_cal][cols]
        X_fu, y_fu = df.iloc[train_idx[ok_t]][cols], y[train_idx[ok_t]]
        X_te = df.iloc[test_idx][cols]

        # the inner model gets the same complexity on its own smaller slice
        sub = p["subsample"]
        p_in = {**p, "min_child_samples": max(G.MIN_CHILD_FLOOR,
                                              max(1, int(len(X_in) * sub)) // TARGET)}
        cals, preds = [], []
        for b in range(n_bag):
            rs = int(seed) * 1000 + b
            cals.append(LGBMRegressor(**{**p_in, "random_state": rs}).fit(X_in, y_in).predict(X_cal))
            preds.append(LGBMRegressor(**{**p, "random_state": rs}).fit(X_fu, y_fu).predict(X_te))
        return np.mean(cals, axis=0), np.mean(preds, axis=0)

    return f


def sig_file(param: str, value, seed: int) -> str:
    tag = str(value).replace(".", "p")
    return os.path.join(OUT, f"ts_{param}_{tag}_bag{N_BAG}_seed{seed}.csv")


def shape(v: List[float], default_idx: int) -> str:
    a = np.asarray(v, float)
    if not np.isfinite(a).all() or len(a) < 3:
        return "-"
    rng, mid = a.max() - a.min(), float(np.nanmedian(np.abs(a)))
    if mid > 0 and rng / mid < 0.15:
        return "flat"
    d = np.diff(a)
    if np.all(d >= -1e-12) or np.all(d <= 1e-12):
        return "monotone (default at an edge)" if default_idx in (0, len(a) - 1) \
            else "monotone (default inside)"
    return f"peaked at index {int(np.nanargmax(a))}"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    chosen = args or list(GRID)
    bad = [p for p in chosen if p not in GRID]
    if bad:
        raise SystemExit(f"unknown parameter(s) {bad}; known: {list(GRID)}")
    os.makedirs(OUT, exist_ok=True)

    df = paths.load_split("train")
    cols = feature_cols(df)
    folds = folds_for()
    rows_total = sum(len(f.train) for f in folds)
    base_p = params_for("learning_rate", TREE["learning_rate"])   # == frozen defaults

    print(f"{len(df)} rows, {len(cols)} columns, rolling {WINDOW}, {len(folds)} periods")
    print(f"frozen: target {TARGET} -> num_leaves={base_p['num_leaves']}, "
          f"min_child_samples={base_p['min_child_samples']}, "
          f"~{int(G.realised(WINDOW, TARGET) * G.LEAF_FACTOR)} actual leaves")
    print(f"sweep at bag={N_BAG}, seeds {list(SEEDS)}\n")

    jobs = [(p, v) for p in chosen for v in GRID[p]]
    est = 0.0
    print(f"{'parameter':<18}{'value':>9}{'min_child':>11}{'realised':>10}{'est min':>9}")
    for p, v in jobs:
        pp = params_for(p, v)
        c = (G.cost(WINDOW, TARGET) * (N_BAG / cd.N_BAG) * (len(SEEDS) / 3)
             * (pp["n_estimators"] / TREE["n_estimators"]))
        if not all(os.path.exists(sig_file(p, v, s)) for s in SEEDS):
            est += c
        star = "  <- default" if v == DEFAULTS[p] else ""
        print(f"{p:<18}{str(v):>9}{pp['min_child_samples']:>11}"
              f"{realised_for(p, v):>10}{c:>9.1f}{star}")
    print(f"\nestimated {est:.0f} min ({est / 60:.1f} h) for the missing configurations")
    if "--plan" in sys.argv:
        return

    runs: Dict[tuple, Dict[int, pd.DataFrame]] = {}
    for p, v in jobs:
        per_seed = {}
        for s in SEEDS:
            f = sig_file(p, v, s)
            if os.path.exists(f):
                per_seed[s] = pd.read_csv(f)
                continue
            t0 = time.time()
            d = cd.collect(df, folds, signal(cols, params_for(p, v)), s,
                           offset=SD.span_offset("train"))
            d.to_csv(f, index=False)
            per_seed[s] = d
            print(f"  {p}={v} seed={s}  {time.time() - t0:.0f}s")
        runs[(p, v)] = per_seed

    out: Dict[str, object] = {"window": WINDOW, "target": TARGET,
                              "n_bag": N_BAG, "seeds": list(SEEDS),
                              "defaults": DEFAULTS}

    for pcol in ("p_window", cd.PRIMARY):
        print(f"\n{'=' * 78}\n{pcol}\n{'=' * 78}")
        for p in chosen:
            base = cd.spreads(runs[(p, DEFAULTS[p])], pcol)
            rows = []
            for v in GRID[p]:
                sp = cd.spreads(runs[(p, v)], pcol)
                r = cd.cross_fold(sp["spread"]) if len(sp) else {}
                rec = {"value": v, "is_default": v == DEFAULTS[p],
                       "realised": realised_for(p, v), "usable": len(sp),
                       "spread_ann": r.get("mean_ann"), "t": r.get("t")}
                if v != DEFAULTS[p]:
                    rec.update({f"vs_def_{k}": val for k, val in cd.paired(base, sp).items()})
                rows.append(rec)
            tab = pd.DataFrame(rows)
            di = int(tab.index[tab["is_default"]][0])
            print(f"\n--- {p} (default {DEFAULTS[p]}) ---")
            show = ["value", "is_default", "realised", "usable", "spread_ann", "t",
                    "vs_def_diff_ann", "vs_def_t", "vs_def_detectable_ann"]
            print(tab[[c for c in show if c in tab.columns]].round(3).to_string(index=False))
            print(f"  shape: {shape(tab['spread_ann'].tolist(), di)}   "
                  f"median {tab['spread_ann'].median():.3f}   "
                  f"range {tab['spread_ann'].max() - tab['spread_ann'].min():.3f}")
            out[f"{pcol}::{p}"] = tab.to_dict("records")

    with open(os.path.join(OUT, "tree_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"""
Reading it, on p_window (all {len(folds)} periods usable, look-ahead identical
across rows so the paired contrasts are valid; the level itself is inflated):

  FLAT EVERYWHERE is the expected result and it is worth having: it says the
  remaining TREE parameters are not where anything is, which is what lets the
  configuration be frozen instead of searched. Combined with n_estimators being
  flat across 30x, the whole model-side of this problem would then be settled
  at (rolling 756, target 117), and the only untouched layer left is the
  position mapping.

  A MOVE larger than its own detectable_ann is worth re-running at the grid's
  bag=7 and three seeds before acting on it. This sweep trades resolution for
  breadth on purpose.

  CHECK THE realised COLUMN on the subsample rows. If it is not constant, that
  row changed capacity as well and is not a subsample test.

written to {OUT}""")


if __name__ == "__main__":
    main()
