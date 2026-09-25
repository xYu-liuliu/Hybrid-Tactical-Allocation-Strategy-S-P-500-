"""
embargo_capacity.py -- re-derive capacity now that the embargo is zero.

THE DECISION THIS FOLLOWS FROM. Deployment trains through yesterday and trades
today; there is no embargo in production. And in a strictly forward walk there
is no backward leakage to prevent -- purging and embargoes exist for k-fold
schemes where a test block can precede its training data. So embargo = 0 is the
protocol, and the project's 10 was a function default in wfo.py that nobody
justified and that the sensitivity check in wfo.py's own docstring never got.

WHY CAPACITY HAS TO BE RE-DERIVED. The frozen 117 leaves was chosen under
embargo = 10, where the nearest training row sits eleven days away and a tree
cannot reach it. At embargo = 0 the nearest training row is adjacent, its
rolling features are near-identical, and a tree at 4.5 rows per leaf can simply
return that row's LABEL -- which is yesterday's return. Daily returns revert at
short horizons, so that produces an inverted signal, and the first weekly run
measured exactly that: block IC -0.0616 (t=-2.22) for the tree, while Ridge at
alpha 1000 stayed positive because shrinkage prevents it from memorising.

If that reading is right, the +0.442 that capacity bought over 4 leaves was
partly an artefact of a protocol that hid the memorisation. The sweep settles
it: at embargo = 0, small trees should hold their ordering while large ones
invert, and the crossover is where capacity should sit.

BOTH EMBARGOES ARE RUN, not because 10 will be adopted, but because the
difference between the two columns at each capacity IS the memorisation
effect, measured. A capacity where the two agree is one the embargo was never
doing anything for.

CADENCE IS HELD AT 84, deliberately. Weekly refitting is the other question and
running both at once would confound them again, as the first weekly run did by
changing cadence and embargo together. Fix capacity here, cheaply, then take it
to the weekly cadence.

    python embargo_capacity.py            # plan and cost
    python embargo_capacity.py --commit
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

import capacity_data as cd
import capacity_window_grid as G
import mapping as MAP
import paths
import ridge_signal as RS
import split_data as SD
from hull_probe import (
    HOLDOUT_FRAC, HOLDOUT_GAP, START_AT, TEST_WINDOW, TRAIN_WINDOW,
    feature_cols, kaggle_adjusted_sharpe,
)
from ranking import frozen_z, tau_daily
from scoring import block_ic
from wfo import make_folds

OUT = os.path.join(paths.PROBE_DIR, "embargo_capacity")
SIG_DIR = os.path.join(OUT, "signals")
CADENCE = TEST_WINDOW          # 84, held fixed
EMBARGOES = (0, 10)
CAPACITIES = (5, 20, 62, 117)
SEEDS = (0, 1)
N_BAG = 1


def drift(block: np.ndarray, s: np.ndarray) -> float:
    d = pd.DataFrame({"b": block, "s": s})
    m, w = d.groupby("b")["s"].mean(), d.groupby("b")["s"].std(ddof=1)
    return float(m.std(ddof=1) / w.mean())


def spans_for(n: int, emb: int):
    """
    The three spans at an arbitrary embargo, on the feature table's index.

    split_data.fold_sets is the same thing at the protocol's embargo of 0. This
    sweep needs 10 as well -- the gap between the two columns IS the
    memorisation effect -- so the grid is rebuilt here and mapped onto each
    span's slice by load_through.
    """
    dev_end = n - int(round(n * HOLDOUT_FRAC)) - HOLDOUT_GAP
    start = dev_end + HOLDOUT_GAP
    every = make_folds(n, TRAIN_WINDOW, CADENCE, embargo=emb, start_at=START_AT)
    hold = [f for f in every if int(f.test[0]) >= start]
    cut = len(hold) // 2
    return {"dev": [f for f in every if int(f.test[-1]) < dev_end],
            "valid": hold[:cut], "test": hold[cut:]}


def main() -> None:
    os.makedirs(SIG_DIR, exist_ok=True)
    n = SD.table_rows()

    print("=" * 104)
    print(f"EMBARGO x CAPACITY -- cadence {CADENCE} held fixed, window {TRAIN_WINDOW}")
    print("=" * 104)
    per_fit = G.cost(TRAIN_WINDOW, 117) / (69 * 3 * 7)
    est = 0.0
    for emb in EMBARGOES:
        sp = spans_for(n, emb)
        nf = sum(len(v) for v in sp.values())
        for tgt in CAPACITIES:
            scale = G.realised(TRAIN_WINDOW, tgt) / G.realised(TRAIN_WINDOW, 117)
            c = per_fit * nf * len(SEEDS) * N_BAG * 2 * scale
            have = all(os.path.exists(os.path.join(
                SIG_DIR, f"e{emb}_c{tgt}_{k}_seed{s}.csv"))
                for k in sp for s in SEEDS)
            if not have:
                est += c
            if emb == EMBARGOES[0]:
                print(f"   capacity {tgt:>4}: ~{int(G.realised(TRAIN_WINDOW, tgt) * G.LEAF_FACTOR):>3}"
                      f" leaves, {int(TRAIN_WINDOW * 0.7) / max(1, G.realised(TRAIN_WINDOW, tgt)):>5.1f}"
                      f" rows/leaf, {c:>5.1f} min per embargo")
    print(f"   {len(EMBARGOES)} embargoes x {len(CAPACITIES)} capacities x 3 spans")
    print(f"   estimated {est:.0f} min for what is missing")
    if "--commit" not in sys.argv:
        print()
        print("   --plan only. Re-run with --commit.")
        return

    rows: List[dict] = []
    for emb in EMBARGOES:
        grid = spans_for(n, emb)
        for name, fl_all in grid.items():
            # Read exactly as far as this grid reaches and no further.
            # load_through reaches back for the history the fits need and never
            # forward, so at the protocol's embargo of 0 a span is fitted from
            # its own slice. At embargo 10 every block moves, so a span's blocks
            # can land past the end of the slice that nominally owns them, and
            # slice_covering asks for the smallest slice that holds them. That
            # arm is a diagnostic contrast, not a selection -- the conclusion is
            # read off the embargo-0 column.
            slice_name = SD.slice_covering(int(max(f.test.max() for f in fl_all)))
            df, _, lo = SD.load_through(slice_name)
            cols = feature_cols(df)
            fl = [type(f)(f.k, f.train - lo, f.test - lo) for f in fl_all]
            bad = [f.k for f in fl if f.train.min() < 0 or f.test.max() >= len(df)]
            if bad:
                raise SystemExit(
                    f"embargo {emb}, span {name}: folds {bad[:5]} reach outside "
                    f"rows {lo}..{lo + len(df) - 1}")

            idx = np.concatenate([f.test for f in fl])          # slice positions
            pos = idx + lo                                      # feature-table rows
            blk = np.concatenate([np.full(len(f.test), int(f.test[0]) + lo) for f in fl])
            fr = df["forward_returns"].to_numpy(float)[idx]
            rf = df["risk_free_rate"].to_numpy(float)[idx]
            x = fr - rf
            bh = kaggle_adjusted_sharpe(np.ones(len(idx)), fr, rf)

            for tgt in CAPACITIES:
                preds = []
                for sd in SEEDS:
                    p_ = os.path.join(SIG_DIR, f"e{emb}_c{tgt}_{name}_seed{sd}.csv")
                    if os.path.exists(p_):
                        preds.append(pd.read_csv(p_)["s"].to_numpy(float))
                        continue
                    t0 = time.time()
                    sig = cd.bagged_signal(cols, tgt, n_bag=N_BAG)
                    pr = np.concatenate([sig(df, f.train, f.test, sd)[1] for f in fl])
                    pd.DataFrame({"pos": pos, "s": pr}).to_csv(p_, index=False)
                    preds.append(pr)
                    print(f"   emb{emb} cap{tgt} {name} seed{sd}: {time.time() - t0:.0f}s")
                s = np.mean(preds, axis=0)

                ic, tt, pf = block_ic(blk, s, x)
                tau, zz = tau_daily(frozen_z(s), x)
                w = np.clip(MAP.WBAR + tau * zz, 0.0, 2.0)
                adj, c = kaggle_adjusted_sharpe(w, fr, rf, return_components=True)
                rows.append({"embargo": emb, "capacity": tgt, "span": name,
                             "leaves": int(G.realised(TRAIN_WINDOW, tgt) * G.LEAF_FACTOR),
                             "block_ic": ic, "t": tt, "pos_frac": pf,
                             "drift": drift(blk, s), "adj": adj, "vs_bh": adj - bh,
                             "vol_ratio": (c["strategy_vol_annual"]
                                           / c["market_vol_annual"]) if c else np.nan,
                             "exposure": float(w.mean()),
                             "turnover": float(np.abs(np.diff(w)).mean())})

    t = pd.DataFrame(rows)
    for col in ("block_ic", "t", "vs_bh", "drift"):
        print(f"\n{'=' * 104}\n{col}\n{'=' * 104}")
        print(t.pivot_table(index=["span", "capacity"], columns="embargo",
                            values=col).round(4).to_string())

    print(f"\n{'=' * 104}\nDETAIL\n{'=' * 104}")
    print(t.round(3).to_string(index=False))

    print(f"""
Reading it.

  THE embargo=0 COLUMN IS THE ONE THAT COUNTS. The other is there so the gap
  between them can be read as the memorisation effect at each capacity, not
  because 10 is going to be adopted.

  block_ic RISING AS LEAVES FALL, at embargo 0, is the prediction. If 5 leaves
  holds its ordering while 117 inverts, capacity is the whole story and the
  frozen cell was chosen under a protocol that concealed it. Pick the crossover
  and take it to the weekly cadence.

  THE TWO COLUMNS AGREEING AT SOME CAPACITY means the embargo was doing nothing
  there -- the tree was never close enough to the boundary to memorise it. That
  capacity is safe under either protocol and is the conservative choice.

  IF 117 DOES NOT INVERT ON dev, the weekly run's -0.0616 came from the
  cadence rather than the embargo after all, and the two need separating again.

  EVERY NUMBER HERE IS AT bag {N_BAG} AND {len(SEEDS)} SEEDS, against the grid's
  bag 7 and 3. That is enough to locate a crossover and not enough to report a
  level; re-run the chosen cell at the grid's settings before building on it.

written to {OUT}""")

    with open(os.path.join(OUT, "embargo_capacity.json"), "w") as f:
        json.dump({"cadence": CADENCE, "seeds": list(SEEDS), "n_bag": N_BAG,
                   "rows": t.to_dict("records")}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
