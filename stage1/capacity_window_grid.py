"""
capacity_window_grid.py -- the (window, capacity) surface, replacing two ladders.

WHY A GRID AND NOT TWO LADDERS. The 4 x 2 measured an interaction directly: the
capacity gain from 5 to 62 leaves was +0.442 at rolling 756, +0.435 at 1512,
+0.245 at 3024 and +0.124 at expanding. One-factor-at-a-time is only valid
without interaction, so two orthogonal lines through (756, 62) cannot find an
optimum that sits off both of them.

Worse, the two axes are not independent even in their ranges. A window supports
only so many leaves, so "cap62" is not the same thing at both ends: it is 59%
of what rolling 756 can support and 12% of what expanding can. The headline
reading -- capacity matters less as data grows -- has a second explanation the
ladders cannot exclude: the long windows were never given enough capacity to
show anything.

THE FLOOR MOVED FROM 5 TO 3. min_child_samples caps the reachable leaf count at
subsample*W/floor, and at floor=5 that ceiling (105 targets at W=756) was a
number I picked, not one the data chose. With the capacity curve still rising
at 62 there was a real chance of running into my own parameter and calling it a
result. At floor=3 the ceiling is 176 and the sweep can find its own end.

This is widening an axis, not tuning: the direction is one the data already
points in, since 8 rows per leaf (target 62 at W=756) beat 105 rows per leaf
(target 5) by +0.442.

subsample stays at 0.7. It is a third knob on the same quantity, and lowering
the floor buys more headroom (176 vs 151 at W=756) without the 43% cost of
putting every row back into every tree.

TARGETS ARE NOT LEAF COUNTS. num_leaves is an upper bound; leaf-wise growth
stops early when leaves go pure or gain runs out. Measured on a real 756-row
window:

    target      5   20   35   62  105  150
    realised    4   16   27   50   77   77      (median over 250 trees)

The ratio is 0.77-0.80 and -- this is what makes the axis usable -- it is the
same at every window: at target 62 the realised median was 49 / 48 / 48 / 48
for W = 756 / 1512 / 3024 / 6000. So the 4 x 2 did hold capacity constant
across windows; only the labels were 22% high. Both target and realised are
reported here.

THE GRID. Levels run until one clips to the window's ceiling, so each window's
top cell IS its frontier and no two cells of a window are the same model:

    W=252  ceiling  58    targets  5  20  62*         (* clips to 58)
    W=504  ceiling 117    targets  5  20  62  117
    W=756  ceiling 176    targets  5  20  62  117  176

(756, 5) and (756, 62) come from the 4 x 2's A1 and A2. The floor change does
not touch them -- at those targets min_child_samples is 105 and 8, both above
either floor -- and main() asserts it rather than assuming it.

Windows are 756 minus a multiple of the test window, or their fold grids
interleave and never share a period. Periods are keyed by the test window's
first row.

READ IT AS A SURFACE. Twelve cells generate many contrasts and the largest is
biased upward: with a per-cell standard error of about 0.167 annualised, the
best of 50 configurations sits +0.38 above the truth, which is 85% of the
largest real effect found so far. That is why this is a small enumerated grid
read for its shape and not a search.

    python capacity_window_grid.py            # fit what is missing, then report
    python capacity_window_grid.py --plan     # sizes and cost, then stop
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
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import paths
import capacity_data as cd
import split_data as SD
from hull_probe import (
    EMBARGO, HOLDOUT_FRAC, HOLDOUT_GAP, INNER_FRAC, RANK_MAP, START_AT,
    TARGET_COL, TEST_WINDOW, TREE, feature_cols,
)
from step2_tails import SEEDS
from wfo import make_folds

MIN_CHILD_FLOOR = 3       # passed to capacity_data.tree_params, which owns the definition          # was 5; see the header
WINDOWS = (252, 504, 756)
TARGETS = (5, 20, 62, 117, 176)
REF_W, REF_L = 756, 62       # the 4 x 2's best cell, every contrast's reference
OUT = os.path.join(paths.PROBE_DIR, "capacity_window_grid")
ADOPT = {(756, 5): "A1", (756, 62): "A2"}
LEAF_FACTOR = 0.78           # measured realised/target, stable across windows

# rate ~ a * leaves**b, fitted to the 4 x 2's two timed points
_B = float(np.log(cd.RATE[62] / cd.RATE[5]) / np.log(62 / 5))
_A = cd.RATE[5] / 5.0 ** _B




def cd_tree_params(n_rows: int, target: int) -> dict:
    """capacity_data's, at this file's lower min_child_samples floor."""
    return cd.tree_params(n_rows, target, MIN_CHILD_FLOOR)


def cd_bagged_signal(cols, target: int, n_bag: int = None):
    """capacity_data's, at this file's lower min_child_samples floor."""
    return cd.bagged_signal(cols, target, n_bag or cd.N_BAG, MIN_CHILD_FLOOR)


def realised(n_rows: int, target: int) -> int:
    """Formula upper bound. Multiply by LEAF_FACTOR for what LightGBM actually grows."""
    p = cd_tree_params(n_rows, target)
    return min(p["num_leaves"], int(n_rows * TREE["subsample"]) // p["min_child_samples"])


def ceiling(n_rows: int) -> int:
    return int(n_rows * TREE["subsample"]) // MIN_CHILD_FLOOR


def cells_for(w: int) -> List[int]:
    """
    Targets up to this window's frontier, with no two targets giving one model.

    Two stops are needed, not one. A target above the ceiling clips down to it,
    so at W=504 both 117 and 176 build a 117-leaf model: checking only for
    clipping would fit the same configuration twice and call it two rungs.
    """
    out: List[int] = []
    seen: set[int] = set()
    for t in TARGETS:
        lv = realised(w, t)
        if lv in seen:              # identical to a cheaper target already taken
            break
        out.append(t)
        seen.add(lv)
        if lv < t:                  # clipped: this is the frontier
            break
    return out


def folds_for(w: int):
    """
    The development fold grid at window `w`, positioned on train_processed.csv.

    make_folds is called with dev_end as the series length, so no fold can reach
    the holdout whatever the window; SD.remap then shifts the result onto the
    slice split_data.py wrote. The grid keeps its own embargo -- the feature-layer
    results were all measured at 10 -- so these folds are not the span's own.
    """
    n = SD.table_rows()
    dev_end = n - int(round(n * HOLDOUT_FRAC)) - HOLDOUT_GAP
    raw = make_folds(dev_end, w, TEST_WINDOW, embargo=EMBARGO,
                     start_at=START_AT, min_train=min(504, w))
    return SD.remap_rows(raw, "train")


def sig_file(w: int, t: int, seed: int) -> str:
    if (w, t) in ADOPT:
        return cd.sig_file(ADOPT[(w, t)], seed)
    return os.path.join(OUT, f"g_W{w}_T{t}_bag{cd.N_BAG}_seed{seed}.csv")




def cost(w: int, t: int) -> float:
    lv = realised(w, t)
    return _A * lv ** _B * sum(len(f.train) for f in folds_for(w)) * len(SEEDS) * cd.N_BAG * 2 / 60


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    off = [w for w in WINDOWS if (REF_W - w) % TEST_WINDOW]
    if off:
        raise SystemExit(f"windows {off} are not {REF_W} minus a multiple of {TEST_WINDOW}; "
                         f"their fold grids would never share a period with W{REF_W}")

    # the adopted cells must be the identical computation under the new floor
    for (w, t), cell in ADOPT.items():
        old = {**TREE, "num_leaves": t, "min_child_samples": max(5, int(w * TREE["subsample"]) // t)}
        new = cd_tree_params(w, t)
        if old != new:
            raise SystemExit(f"{cell} was fitted at min_child_samples="
                             f"{old['min_child_samples']} but this file would use "
                             f"{new['min_child_samples']}; it cannot be adopted")

    df = paths.load_split("train")
    cols = feature_cols(df)
    fsets = {w: folds_for(w) for w in WINDOWS}
    plan = [(w, t) for w in WINDOWS for t in cells_for(w)]

    print(f"{len(df)} rows, {len(cols)} columns, bag={cd.N_BAG}, seeds {list(SEEDS)}")
    print(f"min_child_samples floor {MIN_CHILD_FLOOR} (was 5), subsample {TREE['subsample']}\n")
    print(f"{'W':>5}{'ceiling':>9}{'target':>8}{'min_child':>11}{'bound':>7}"
          f"{'~actual':>9}{'folds':>7}{'source':>10}{'est min':>9}")
    todo, est = [], 0.0
    for w, t in plan:
        p, lv = cd_tree_params(w, t), realised(w, t)
        have = all(os.path.exists(sig_file(w, t, s)) for s in SEEDS)
        src = ADOPT.get((w, t), "cached" if have else "to fit")
        c = cost(w, t)
        if not have:
            todo.append((w, t))
            est += c
        clip = "  <- frontier" if t == cells_for(w)[-1] else ""
        print(f"{w:>5}{ceiling(w):>9}{t:>8}{p['min_child_samples']:>11}{lv:>7}"
              f"{int(lv * LEAF_FACTOR):>9}{len(fsets[w]):>7}{src:>10}{c:>9.0f}{clip}")

    print(f"\n{len(todo) * len(SEEDS)} (cell, seed) runs to do   "
          f"estimated {est:.0f} min ({est / 60:.1f} h)")
    print("cells are fitted cheapest first, and every one is resumable, so an "
          "early stop still leaves a readable surface")
    if "--plan" in sys.argv:
        return

    runs: Dict[Tuple[int, int], Dict[int, pd.DataFrame]] = {}
    for w, t in sorted(plan, key=lambda wt: cost(*wt)):
        per_seed = {}
        for s in SEEDS:
            p = sig_file(w, t, s)
            if os.path.exists(p):
                per_seed[s] = pd.read_csv(p)
                print(f"  W{w} T{t} seed={s}  ({ADOPT.get((w, t), 'cached')})")
                continue
            t0 = time.time()
            d = cd.collect(df, fsets[w], cd_bagged_signal(cols, t), s,
                           offset=SD.span_offset("train"))
            d.to_csv(p, index=False)
            per_seed[s] = d
            print(f"  W{w} T{t} seed={s}  {time.time() - t0:.0f}s")
        runs[(w, t)] = per_seed

    out: Dict[str, object] = {"floor": MIN_CHILD_FLOOR, "windows": list(WINDOWS),
                              "targets": list(TARGETS), "n_bag": cd.N_BAG,
                              "cells": [{"w": w, "t": t, "bound": realised(w, t)} for w, t in plan]}

    for pcol in (cd.PRIMARY, "p_window"):
        sp = {k: cd.spreads(runs[k], pcol) for k in plan}

        surf = pd.DataFrame(
            [{"W": w, **{f"T{t}": (cd.cross_fold(sp[(w, t)]["spread"])["mean_ann"]
                                   if (w, t) in sp and len(sp[(w, t)]) else np.nan)
                         for t in TARGETS}} for w in WINDOWS])
        print(f"\n=== {pcol}: spread_ann surface ===")
        print(surf.round(3).to_string(index=False))

        lvl = pd.DataFrame([{
            "W": w, "target": t, "bound": realised(w, t), "usable": len(sp[(w, t)]),
            **{k: v for k, v in (cd.cross_fold(sp[(w, t)]["spread"]) if len(sp[(w, t)]) else {}).items()
               if k in ("mean_ann", "t", "pos_frac")}} for w, t in plan])
        print(f"\n=== {pcol}: levels ===")
        print(lvl.round(3).to_string(index=False))

        rows = []
        for w in WINDOWS:                                   # capacity curve within a window
            ts = cells_for(w)
            for t in ts[1:]:
                rows.append({"kind": f"capacity @ W{w}", "contrast": f"T{t} - T{ts[0]}",
                             **cd.paired(sp[(w, ts[0])], sp[(w, t)])})
        for t in TARGETS:                                   # window curve at a shared capacity
            ws = [w for w in WINDOWS if (w, t) in sp]
            if REF_W in ws:
                for w in ws:
                    if w != REF_W:
                        rows.append({"kind": f"window @ T{t}", "contrast": f"W{w} - W{REF_W}",
                                     **cd.paired(sp[(REF_W, t)], sp[(w, t)])})
        for w in WINDOWS:                                   # frontier vs the 4x2's best cell
            top = cells_for(w)[-1]
            if (w, top) != (REF_W, REF_L):
                rows.append({"kind": "frontier", "contrast": f"W{w}T{top} - W{REF_W}T{REF_L}",
                             **cd.paired(sp[(REF_W, REF_L)], sp[(w, top)])})
        con = pd.DataFrame(rows)
        print(f"\n=== {pcol}: contrasts (paired on shared periods) ===")
        print(con.round(3).to_string(index=False))

        out[pcol] = {"levels": lvl.to_dict("records"), "contrasts": con.to_dict("records")}

    with open(os.path.join(OUT, "capacity_window_grid.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"""
Reading it, on p_window (every period usable, look-ahead identical across cells
so paired contrasts are valid; the level itself is inflated):

  THE CAPACITY CURVE inside each window is the first thing. If W{REF_W}'s curve
  is still rising at target 176 -- its own frontier now, not my old floor of 5 --
  then capacity is limited by the window and the next lever is n_estimators or
  learning_rate, which are the other coordinates of the same complexity. If it
  turns over, that peak is the capacity this problem supports.

  THE FRONTIER ROW answers the question the ladders could not: does a short
  window at ITS OWN maximum capacity beat (756, 62)? W252 at 58 leaves and W504
  at 117 are each the best their window can do. If either wins, recency beats
  volume more strongly than the 4 x 2 suggested.

  THE WINDOW ROWS at a shared target say whether the 4 x 2's window effect
  survives below 756. Read them beside the capacity rows: the whole point of a
  grid is that these two are not independent.

  NOT COVERED: long windows at high capacity. W=1512 at target 117 would cost
  about two hours and W=3024 at 176 about five, since cost grows as W * L**0.85.
  The prior against it is that extra history was already worthless at target 5
  (-0.020, t=-0.14) before capacity entered, and the signal is strongest at a
  one-day horizon and gone by 63 days. That is a judgement, not evidence, and it
  stays a documented gap.

written to {OUT}""")


if __name__ == "__main__":
    main()
