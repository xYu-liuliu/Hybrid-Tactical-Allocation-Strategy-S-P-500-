"""
ablate_blocks.py (v2) -- remove one structural block at a time, at a model big
enough to notice.

v1 ran these five blocks at TREE's settings and every one came back null. That
result was about the model, not the features. Measured on a real 756-row
window, TREE grows 4 leaves:

                  splits/tree   distinct columns/tree   columns ever used
    4 leaves            3               2.9             267 of 1132 (24%)
    92 leaves          91              65.7             666 of 1132 (59%)

A tree making three splits cannot express a difference between feature
transforms whether or not one exists. Removing 261 columns from a model that
touches 24% of the column space, three at a time, was never going to register.

The grid then established that the starvation was real and costly: capacity
from 4 to 48 leaves was worth +0.442 annualised at rolling 756 (t=2.77), and
the plateau runs from target 62 to 176. (756, target 117) is the frozen
configuration -- 92 actual leaves, best stability of the twelve cells (1.160),
best q10 and best worst period.

SO THIS IS NOT MAINLY ABOUT WHETHER TO CUT COLUMNS. Cutting buys nothing:
973 columns and 1114 columns both fitted in 9 seconds, and the n_select sweep
already put k=all above k=300 by 0.274 (t=-2.64). The question is whether
build_features contributes at all, which three separate results said no to --
the 128-column rebuild moving the signal by 1.5%, the k sweep, and v1's five
nulls -- and all three were measured on the 4-leaf model.

    still null at 92 leaves   the 1132 columns really are redundant
                              re-encodings of ~93 underlying series, adding
                              more transforms of them is provably pointless,
                              and the only feature-side lever left is new
                              underlying data, which the competition does not
                              provide.
    something bites           the feature engineering does work and v1 was
                              blind to it.

BASELINE is the grid's W756/T117 cell, adopted rather than refitted: same
window, capacity, bag count, seeds and periods, the identical computation.

v1's numbers are printed alongside where they exist. They are NOT paired with
these -- different capacity, different bag count, and v1 keyed periods by fold
index while everything since keys them by the test window's first row -- so
read them as context, not as a contrast.

    python ablate_blocks.py                      # the five blocks
    python ablate_blocks.py drop_D drop_gate     # a subset
    python ablate_blocks.py --list               # block sizes and overlaps
    python ablate_blocks.py --plan               # cost, then stop
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
import re
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import paths
import capacity_data as cd
import capacity_window_grid as G
import split_data as SD
from hull_probe import (
    EMBARGO, HOLDOUT_FRAC, HOLDOUT_GAP, START_AT, TEST_WINDOW,
    feature_cols,
)
from step2_tails import MIN_TAIL_N, SEEDS
from wfo import make_folds
from Hull_Tactical_feature_engineering import group_of

try:
    from scipy import stats as _st
except ImportError:
    _st = None

WINDOW, TARGET = G.REF_W, 117          # the frozen configuration
N_BAG = cd.N_BAG
TAIL = cd.TAIL
PRIMARY = cd.PRIMARY
OUT = os.path.join(paths.PROBE_DIR, f"ablation_T{TARGET}")
V1_SUMMARY = os.path.join(paths.PROBE_DIR, "ablation", "ablation_summary.csv")

DEFAULT_BLOCKS = ("drop_D", "drop_mean_long", "drop_pos", "drop_z_long", "drop_gate")


def blocks(cols: Sequence[str]) -> Dict[str, List[str]]:
    """
    The columns each named block removes.

    lagged_* is excluded from the _mean and _z blocks on purpose: the rolling
    statistics of the lagged labels are a different family and the two highest
    lifts in the audit (lagged _std 2.39, lagged _mean 2.25). Lumping them in
    would test two opposite things at once.

    _pc1_z126 and _pc1_z252 are excluded from drop_z_long for the same reason
    (pc1 _z lift 1.18) and because they are the gate sources -- removing them
    would silently mix a gate change into a z change.

    drop_z_long is the long windows only: the audit puts _z5 at lift 1.268 and
    _z10 at 1.187 against _z126 0.845 and _z252 0.778, so removing the family
    whole would test whether z works at all rather than whether its long
    windows are dead weight -- the same split as drop_mean_long.

    drop_D and drop_mean_long OVERLAP on the 9 D*_mean63 columns: the D group's
    only derivative IS a long-window mean. Each is still a clean single-block
    test against the same baseline, but their effects cannot be added.
    """
    lag = lambda c: c.startswith("lagged_")
    pc1 = lambda c: "_pc1" in c
    return {
        "drop_D":         [c for c in cols if group_of(c) == "D"],
        "drop_mean_long": [c for c in cols if re.search(r"_mean(63|126|252)$", c) and not lag(c)],
        "drop_pos":       [c for c in cols if re.search(r"_pos\d+$", c)],
        "drop_z_long":    [c for c in cols if re.search(r"_z(126|252)$", c) and not lag(c) and not pc1(c)],
        "drop_gate":      [c for c in cols if re.search(r"_g(?:V|E|Vstrong)$", c)],
    }


def folds_for():
    """
    The development fold grid, positioned on train_processed.csv.

    make_folds is called with dev_end as the series length, so no fold can reach
    the holdout; SD.remap_rows then shifts the result onto the slice
    split_data.py wrote. The embargo is this file's own, kept at what the
    feature-layer results were measured at.
    """
    n = SD.table_rows()
    dev_end = n - int(round(n * HOLDOUT_FRAC)) - HOLDOUT_GAP
    return SD.remap_rows(
        make_folds(dev_end, WINDOW, TEST_WINDOW, embargo=EMBARGO,
                   start_at=START_AT), "train")


def sig_file(block: str, seed: int) -> str:
    return os.path.join(OUT, f"ablate_{block}_W{WINDOW}_T{TARGET}_bag{N_BAG}_seed{seed}.csv")


def base_file(seed: int) -> str:
    return G.sig_file(WINDOW, TARGET, seed)


def spreads(per_seed: Dict[int, pd.DataFrame], pcol: str) -> pd.DataFrame:
    return cd.spreads(per_seed, pcol)


def v1_context() -> Dict[str, float]:
    """v1's diff_ann per block on p_trail_dm, for the printout. Context only."""
    if not os.path.exists(V1_SUMMARY):
        return {}
    d = pd.read_csv(V1_SUMMARY)
    d = d[(d["ranking"] == PRIMARY) & (d["variant"] != "baseline")]
    return dict(zip(d["variant"], d.get("vs_base_mean_diff_ann", pd.Series(dtype=float))))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    os.makedirs(OUT, exist_ok=True)

    df = paths.load_split("train")
    cols = feature_cols(df)
    reg = blocks(cols)

    if "--list" in sys.argv:
        print(f"{len(cols)} feature columns\n")
        for k, v in reg.items():
            print(f"  {k:<16} {len(v):>4} cols   e.g. {', '.join(v[:4])}")
        names = list(reg)
        print()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ov = set(reg[a]) & set(reg[b])
                if ov:
                    print(f"  overlap: {a} & {b} share {len(ov)} cols "
                          f"({', '.join(sorted(ov)[:3])}...)")
        return

    chosen = args or list(DEFAULT_BLOCKS)
    unknown = [b for b in chosen if b not in reg]
    if unknown:
        raise SystemExit(f"unknown block(s) {unknown}; known: {list(reg)}")

    missing = [p for s in SEEDS if not os.path.exists(p := base_file(s))]
    if missing:
        raise SystemExit(f"baseline missing: {missing[0]}\n"
                         f"Run capacity_window_grid.py first; the W{WINDOW}/T{TARGET} "
                         f"cell is this file's baseline and is not refitted here.")

    folds = folds_for()
    p = G.cd_tree_params(WINDOW, TARGET)
    full_cost = G.cost(WINDOW, TARGET)

    print(f"{len(df)} rows, {len(cols)} columns, {len(folds)} periods, bag={N_BAG}, "
          f"seeds {list(SEEDS)}")
    print(f"frozen config: rolling {WINDOW}, target {TARGET} "
          f"(num_leaves={p['num_leaves']}, min_child_samples={p['min_child_samples']}, "
          f"~{int(G.realised(WINDOW, TARGET) * G.LEAF_FACTOR)} actual leaves)")
    print(f"baseline: grid cell W{WINDOW}/T{TARGET}, adopted not refitted\n")

    est = 0.0
    for b in chosen:
        keep = len(cols) - len(reg[b])
        c = full_cost * keep / len(cols)          # histogram cost tracks column count
        if not all(os.path.exists(sig_file(b, s)) for s in SEEDS):
            est += c
        print(f"  {b:<16} removes {len(reg[b]):>4} -> {keep:>5} remain   ~{c:>4.0f} min")
    print(f"\nestimated {est:.0f} min ({est / 60:.1f} h) for the missing cells")
    if "--plan" in sys.argv:
        return

    runs: Dict[str, Dict[int, pd.DataFrame]] = {}
    for b in chosen:
        per_seed = {}
        for s in SEEDS:
            f = sig_file(b, s)
            if os.path.exists(f):
                per_seed[s] = pd.read_csv(f)
                print(f"  {b} seed={s}  (cached)")
                continue
            keep = [c for c in cols if c not in set(reg[b])]
            t0 = time.time()
            d = cd.collect(df, folds, G.cd_bagged_signal(keep, TARGET), s,
                           offset=SD.span_offset("train"))
            d.to_csv(f, index=False)
            per_seed[s] = d
            print(f"  {b} seed={s}  {time.time() - t0:.0f}s")
        runs[b] = per_seed

    base = {s: pd.read_csv(base_file(s)) for s in SEEDS}
    v1 = v1_context()
    out: Dict[str, object] = {"window": WINDOW, "target": TARGET, "n_bag": N_BAG,
                              "blocks": {b: len(reg[b]) for b in chosen}}

    for pcol in (PRIMARY, "p_window"):
        bt = spreads(base, pcol)
        rb = cd.cross_fold(bt["spread"]) if len(bt) else {}
        rows = [{"variant": "baseline", "removed": 0, "usable": len(bt),
                 "spread_ann": rb.get("mean_ann"), "t": rb.get("t")}]
        for b in chosen:
            vt = spreads(runs[b], pcol)
            rv = cd.cross_fold(vt["spread"]) if len(vt) else {}
            rec = {"variant": b, "removed": len(reg[b]), "usable": len(vt),
                   "spread_ann": rv.get("mean_ann"), "t": rv.get("t")}
            rec.update({f"vs_base_{k}": v for k, v in cd.paired(bt, vt).items()})
            if pcol == PRIMARY and b in v1:
                rec["v1_at_4_leaves"] = v1[b]
            rows.append(rec)
        res = pd.DataFrame(rows)

        print(f"\n=== {pcol} ===")
        show = ["variant", "removed", "usable", "spread_ann", "t",
                "vs_base_diff_ann", "vs_base_t", "vs_base_win", "vs_base_detectable_ann"]
        if "v1_at_4_leaves" in res.columns:
            show.append("v1_at_4_leaves")
        print(res[[c for c in show if c in res.columns]].round(3).to_string(index=False))
        out[pcol] = res.to_dict("records")

    res_p = pd.DataFrame(out[PRIMARY])
    hit = res_p[(res_p["variant"] != "baseline")
                & (res_p["vs_base_diff_ann"].abs() > res_p["vs_base_detectable_ann"])]
    with open(os.path.join(OUT, "ablation.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"""
Reading it, each contrast against its OWN detectable_ann:

  ALL FIVE NULL AGAIN, now on a model that uses 59% of the column space and
  66 columns per tree instead of 24% and 3, is the strong version of the
  result v1 could not support: the 1132 columns are redundant re-encodings of
  ~93 underlying series, and more transforms of those series cannot help.
  That closes the feature-engineering question rather than leaving it open,
  and it says the remaining leverage is the position mapping, which is still
  untouched and still known broken -- position noise doubling strategy
  variance, 28.3% of days pinned at 0 or 2, a one-sided crash rule firing on
  12.7%.

  A BLOCK THAT NOW BITES means build_features does work and v1 was blind to
  it. Read the sign: positive means removal helped and the block was diluting;
  negative means it was carrying signal that a 4-leaf model could not reach.

  usable moving with a block is its own result -- that block changed the
  signal's drift, not just its accuracy.

  {len(hit)} of {len(chosen)} blocks cleared their threshold on {PRIMARY}.

  These are five SEPARATE one-block tests and their effects are not additive;
  drop_D and drop_mean_long even share 9 columns. A combined variant has to be
  run as its own variant before it can be claimed.

written to {OUT}""")


if __name__ == "__main__":
    main()
