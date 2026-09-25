"""
capacity_data.py -- separate "more history" from "a bigger model", 4 x 2.

TWO THINGS MOVE AT ONCE UNDER THE FIXED TREE SETTINGS. min_child_samples=100 is
an absolute row count, so the leaves a tree can grow scale with the window:

    rolling 756   ->  529 rows/tree  ->  5 leaves   (min_child_samples binds)
    expanding     -> 4527 rows/tree  -> 31 leaves   (num_leaves binds)

"expanding vs rolling at TREE" therefore changes data volume AND capacity
together, and the two readings demand opposite actions: more history is worth
having, or the model was starved at 756 rows and rolling should just be given
more capacity.

A THIRD THING MOVES TOO, which a rolling/expanding binary cannot see. Expanding
is not only more data, it is OLDER data. At the last fold it trains on 6468
rows of which the 756 rolling would have used are 11.7%, and the oldest row is
25.7 years before the test window. More observations lower variance; stale
observations raise bias if the relationship is not stationary. They pull
opposite ways, so a binary contrast can read zero while both effects are large.

The fix is a ladder, not a binary. 756 -> 1512 -> 3024 -> expanding increases
volume and staleness together, so the SHAPE is the answer:

    rising          volume wins, staleness has not bitten yet in this range
    peaked          the peak is the memory length of the relationship, which is
                    the most useful single number this experiment can produce
    falling         staleness dominates and 756 may already be too long

Crossed with capacity, that is 4 x 2:

                    cap 5      cap 62
    rolling 756      A1         A2
    rolling 1512     B1         B2
    rolling 3024     C1         C2
    expanding        D1         D2

CAPACITY IS A TARGET LEAF COUNT, not a parameter set. num_leaves is set to the
target and min_child_samples is derived from each fold's own training size, so
a cell holds the same model complexity whether it fits 756 rows or 6468.
Without that, "expanding at capacity 5" cannot be expressed at all. At capacity
62 on 756 rows a leaf holds about 8 rows, which is aggressive for data this
noisy -- that is the point, since only a starved model gains from the move.

FOLDS ARE KEYED BY TEST-WINDOW START, NOT BY INDEX. A longer window needs more
warm-up, so rolling 3024 has 42 folds against rolling 756's 69, and their fold
indices refer to different periods (fold 0 tests row 3539 against row 1271).
Every window's test windows sit on the same 84-row grid, so the first test row
identifies a period across all of them; that is what cells are paired on. Each
contrast reports how many periods it actually shared.

READ THE LADDER AS A CURVE. Eight cells generate many pairwise contrasts and
the largest of them is biased upward by construction -- the trap
diagnostics.run_baselines warns about. The shape and the median are the
estimate; a single winning rung is not.

    python capacity_data.py                 # all eight
    python capacity_data.py A1 A2           # a subset
    python capacity_data.py --plan          # sizes and cost, then stop
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import paths
from wfo import Fold, make_folds
from hull_probe import (
    EMBARGO, HOLDOUT_FRAC, HOLDOUT_GAP, INNER_FRAC, RANK_MAP, START_AT,
    TARGET_COL, TEST_WINDOW, TREE, feature_cols, load,
)
from step2_tails import MIN_TAIL_N, SEEDS, cross_fold, fold_table, trailing_percentile_detrended

try:
    from scipy import stats as _st
except ImportError:
    _st = None

# 7, not the 15 the other experiments use. Measured on the cached step2 run,
# where bag=1 and bag=15 exist for the same folds and seeds: the per-period
# spread had variance 1.003e-05 at bag=1 against 1.117e-05 at bag=15 (inside the
# ~20% standard error of a variance on 48 periods), and the effect itself moved
# +0.012 annualised against a detectable threshold of 0.165. Once a spread has
# been averaged over the ~10 days in a tail and then over three seeds inside the
# period, model randomness is gone; what is left is genuine period-to-period
# variation, which bagging does not touch.
N_BAG = 7
TAIL = 0.10
PRIMARY = "p_trail_dm"
OUT = os.path.join(paths.PROBE_DIR, "capacity_data")

WINDOWS: Dict[str, Optional[int]] = {"A": 756, "B": 1512, "C": 3024, "D": None}  # None = expanding
CAPS: Dict[str, int] = {"1": 5, "2": 62}
CELLS = [f"{w}{c}" for w in WINDOWS for c in CAPS]

# seconds per training row per fit, measured on this machine at both capacities
RATE = {5: 1.375e-4, 62: 1.177e-3}


def label(cell: str) -> str:
    w, c = WINDOWS[cell[0]], CAPS[cell[1]]
    return f"{'expanding' if w is None else f'roll {w}'} x cap{c}"


MIN_CHILD_FLOOR = 5      # capacity_window_grid.py passes 3 to reach higher targets


def tree_params(n_rows: int, target_leaves: int,
                min_child_floor: int = MIN_CHILD_FLOOR) -> dict:
    """
    Same complexity whatever the training size.

    With n_rows*subsample rows going into a tree, asking for `target_leaves`
    means each may hold about n_rows*subsample/target_leaves. num_leaves is set
    to the target as well, so the two constraints agree instead of one silently
    overriding the other the way they do in TREE.

    min_child_floor is the only thing capacity_window_grid.py needed to change,
    so it is a parameter rather than a second copy of this function.
    """
    per_tree = max(1, int(n_rows * TREE["subsample"]))
    return {**TREE, "num_leaves": target_leaves,
            "min_child_samples": max(min_child_floor, per_tree // target_leaves)}


def realised_leaves(n_rows: int, target_leaves: int) -> int:
    p = tree_params(n_rows, target_leaves)
    return min(p["num_leaves"], int(n_rows * TREE["subsample"]) // p["min_child_samples"])


def folds_for(window: Optional[int], dev_end: int) -> List[Fold]:
    return make_folds(dev_end, window or 756, TEST_WINDOW, embargo=EMBARGO,
                      start_at=START_AT, expanding=window is None)


def bagged_signal(cols: Sequence[str], target_leaves: int, n_bag: int = N_BAG,
                  min_child_floor: int = MIN_CHILD_FLOOR):
    cols = list(cols)

    def f(df, train_idx, test_idx, seed):
        y = df[TARGET_COL].to_numpy(float)
        cut = int(len(train_idx) * INNER_FRAC)
        inner_tr, inner_cal = train_idx[:cut], train_idx[cut:]
        ok_i, ok_t = np.isfinite(y[inner_tr]), np.isfinite(y[train_idx])

        X_inner, y_inner = df.iloc[inner_tr[ok_i]][cols], y[inner_tr[ok_i]]
        X_cal = df.iloc[inner_cal][cols]
        X_full, y_full = df.iloc[train_idx[ok_t]][cols], y[train_idx[ok_t]]
        X_test = df.iloc[test_idx][cols]

        # each model gets the capacity of the set it is actually fitted on
        p_inner = tree_params(len(X_inner), target_leaves, min_child_floor)
        p_full = tree_params(len(X_full), target_leaves, min_child_floor)

        cals, preds = [], []
        for b in range(n_bag):
            rs = int(seed) * 1000 + b
            cals.append(LGBMRegressor(**{**p_inner, "random_state": rs})
                        .fit(X_inner, y_inner).predict(X_cal))
            preds.append(LGBMRegressor(**{**p_full, "random_state": rs})
                         .fit(X_full, y_full).predict(X_test))
        return np.mean(cals, axis=0), np.mean(preds, axis=0)

    return f


def collect(df, folds, signal_fn, seed, offset: int = 0) -> pd.DataFrame:
    """
    `fold` is the test window's FIRST ROW, not the fold index.

    Window lengths produce different numbers of folds starting in different
    places, so index 0 means a different period in each. The first test row is
    the same identifier everywhere, which is what makes the cells pairable.

    `offset` is where `df` begins in the feature table. Callers that pass a
    split slice pass its offset, so `fold` and `pos` are written in FEATURE
    TABLE rows whatever frame produced them -- otherwise a cache built from a
    slice could not be compared with one built from the whole table.
    """
    fwd = df["forward_returns"].to_numpy(float)
    rf = df["risk_free_rate"].to_numpy(float)
    w, mh = RANK_MAP["window"], RANK_MAP["min_hist"]
    parts = []
    for f in folds:
        s_cal, s_te = signal_fn(df, f.train, f.test, seed)
        s = pd.Series(s_te)
        parts.append(pd.DataFrame({
            "fold": int(f.test[0]) + offset, "pos": f.test + offset,
            "n_train": len(f.train), "s_raw": s_te,
            "p_trail_dm": trailing_percentile_detrended(s_cal, s_te, w, mh),
            "p_window": (s.rank(method="first") - 0.5).to_numpy() / len(s),
            "x": fwd[f.test] - rf[f.test],
        }))
    return pd.concat(parts, ignore_index=True)


def sig_file(cell: str, seed: int) -> str:
    return os.path.join(OUT, f"cap_{cell}_bag{N_BAG}_seed{seed}.csv")


def spreads(per_seed: Dict[int, pd.DataFrame], pcol: str) -> pd.DataFrame:
    t = fold_table(per_seed, pcol, TAIL, MIN_TAIL_N)
    return t[["fold", "spread"]] if len(t) else pd.DataFrame(columns=["fold", "spread"])


def paired(a: pd.DataFrame, b: pd.DataFrame, keep: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """
    b - a over the periods both measured; `keep` restricts to a subset of them.

    Always returns the same keys. A short-circuit that dropped them left callers
    building a DataFrame whose columns depended on whether any pair happened to
    share three periods, which is exactly the kind of thing that fails at the
    reporting step after the fits have already been paid for.
    """
    empty = {"n": 0, "diff_ann": np.nan, "t": np.nan,
             "win": np.nan, "detectable_ann": np.nan}
    if not len(a) or not len(b):
        return empty
    m = a.merge(b, on="fold", suffixes=("_a", "_b"))
    if keep is not None:
        m = m[m["fold"].isin(list(keep))]
    if len(m) < 3:
        return {**empty, "n": len(m)}
    d = (m["spread_b"] - m["spread_a"]).to_numpy()
    n, sd = len(d), d.std(ddof=1)
    tc = float(_st.t.ppf(0.975, n - 1)) if _st is not None else 1.96 + 2.4 / max(n - 1, 1)
    return {"n": n, "diff_ann": float(d.mean() * 252),
            "t": float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan,
            "win": float((d > 0).mean()),
            "detectable_ann": float(tc * sd / np.sqrt(n) * 252)}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    chosen = args or list(CELLS)
    bad = [c for c in chosen if c not in CELLS]
    if bad:
        raise SystemExit(f"unknown cell(s) {bad}; known: {CELLS}")
    os.makedirs(OUT, exist_ok=True)

    df = load()
    cols = feature_cols(df)
    n = len(df)
    dev_end = n - int(round(n * HOLDOUT_FRAC)) - HOLDOUT_GAP
    fsets = {w: folds_for(win, dev_end) for w, win in WINDOWS.items()}

    grids = {w: {int(f.test[0]) for f in fs} for w, fs in fsets.items()}
    common = set.intersection(*grids.values())
    print(f"{n} rows, {len(cols)} columns, bag={N_BAG}, seeds {list(SEEDS)}")
    print(f"periods shared by every window: {len(common)}  "
          f"(each contrast uses its own pair's overlap, reported per row)")
    print()

    print(f"{'cell':<5}{'config':<22}{'folds':>6}{'train rows':>22}{'leaves':>10}{'est min':>9}")
    est_total = 0.0
    for c in CELLS:
        cap = CAPS[c[1]]
        fs = fsets[c[0]]
        sizes = np.array([len(f.train) for f in fs])
        lv = [realised_leaves(s, cap) for s in sizes]
        est = RATE[cap] * sizes.sum() * len(SEEDS) * N_BAG * 2 / 60
        if c in chosen:
            est_total += est
        print(f"{c:<5}{label(c):<22}{len(fs):>6}"
              f"{f'{sizes.min()}-{sizes.max()} (mean {int(sizes.mean())})':>22}"
              f"{f'{min(lv)}-{max(lv)}':>10}{est:>9.0f}")

    todo = [(c, s) for c in chosen for s in SEEDS if not os.path.exists(sig_file(c, s))]
    print(f"\n{len(todo)} (cell, seed) runs to do   estimated {est_total:.0f} min "
          f"({est_total / 60:.1f} h) for the {len(chosen)} chosen cells")
    if "--plan" in sys.argv:
        return

    runs: Dict[str, Dict[int, pd.DataFrame]] = {}
    for c in chosen:
        cap = CAPS[c[1]]
        per_seed = {}
        for s in SEEDS:
            p = sig_file(c, s)
            if os.path.exists(p):
                per_seed[s] = pd.read_csv(p)
                print(f"  {c} ({label(c)}) seed={s}  (cached)")
                continue
            t0 = time.time()
            d = collect(df, fsets[c[0]], bagged_signal(cols, cap), s)
            d.to_csv(p, index=False)
            per_seed[s] = d
            print(f"  {c} ({label(c)}) seed={s}  {time.time() - t0:.0f}s")
        runs[c] = per_seed

    srt = sorted(common)
    early, late = srt[: len(srt) // 2], srt[len(srt) // 2:]
    out: Dict[str, object] = {"n_bag": N_BAG, "tail": TAIL, "primary": PRIMARY,
                              "cells": {c: label(c) for c in chosen},
                              "n_common_periods": len(common)}

    for pcol in (PRIMARY, "p_window"):
        sp = {c: spreads(runs[c], pcol) for c in chosen}

        lvl = pd.DataFrame([{
            "cell": c, "config": label(c), "usable": len(sp[c]),
            **{k: v for k, v in (cross_fold(sp[c]["spread"]) if len(sp[c]) else {}).items()
               if k in ("mean_ann", "t", "pos_frac")}} for c in chosen])
        print(f"\n=== {pcol}: levels (each on its own periods) ===")
        print(lvl.round(3).to_string(index=False))

        rows = []
        for cap in CAPS:                       # the ladder, inside one capacity
            rungs = [w + cap for w in WINDOWS if w + cap in chosen]
            for lo, hi in zip(rungs[:-1], rungs[1:]):
                rows.append({"kind": "ladder", "contrast": f"{hi} - {lo}",
                             "what": f"{label(hi)}  vs  {label(lo)}", **paired(sp[lo], sp[hi])})
            if len(rungs) > 2:                 # the two ends of the ladder
                rows.append({"kind": "ladder", "contrast": f"{rungs[-1]} - {rungs[0]}",
                             "what": f"{label(rungs[-1])}  vs  {label(rungs[0])}",
                             **paired(sp[rungs[0]], sp[rungs[-1]])})
        for w in WINDOWS:                      # capacity, at one window length
            lo, hi = w + "1", w + "2"
            if lo in chosen and hi in chosen:
                rows.append({"kind": "capacity", "contrast": f"{hi} - {lo}",
                             "what": f"cap62 vs cap5 @ {label(lo).split(' x ')[0]}",
                             **paired(sp[lo], sp[hi])})
        con = pd.DataFrame(rows)
        if len(con):
            print(f"\n=== {pcol}: contrasts (paired on shared periods) ===")
            print(con.round(3).to_string(index=False))

        # staleness placebo: expanding only diverges from rolling 756 late on
        if "A1" in chosen and "D1" in chosen:
            print(f"\n=== {pcol}: expanding vs roll756 @ cap5, split in time ===")
            ph = pd.DataFrame([{"periods": tag, **paired(sp["A1"], sp["D1"], k)}
                               for tag, k in (("all", None), ("early", early), ("late", late))])
            print(ph.round(3).to_string(index=False))
            out[f"{pcol}_placebo"] = ph.to_dict("records")

        out[pcol] = {"levels": lvl.to_dict("records"),
                     "contrasts": con.to_dict("records") if len(con) else []}

    with open(os.path.join(OUT, "capacity_data.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"""
Reading it, on {PRIMARY}, each contrast against its OWN detectable_ann:

  THE LADDER is the point. Read A1 -> B1 -> C1 -> D1 as a curve, not as four
  chances to find a winner. Rising means the extra history pays and staleness
  has not bitten. Peaked locates the memory length of the relationship. Falling
  means 756 is already too long. Flat closes the question.

  CAPACITY is the A2-A1 / B2-B1 / C2-C1 / D2-D1 rows. If it is large at rolling
  756, then every result so far -- the five null ablations, the feature-set
  invariance where 128 changed columns moved the signal by 1.5% -- was measured
  on a 5-leaf model too small to express a feature difference, and has to be
  redone before it is believed.

  THE PLACEBO is the last table. Expanding's first fold trains on the same 756
  rows as rolling 756 and only diverges later, so a genuine volume-or-staleness
  effect must be near zero early and present late. One already present in the
  early half is neither.

  Everything below its own detectable_ann means the fixed rolling-756-at-5-
  leaves setup costs nothing measurable at this resolution, and the question is
  closed.

written to {OUT}""")


if __name__ == "__main__":
    main()
