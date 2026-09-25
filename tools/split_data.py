"""
split_data.py -- Stage 2: cut the feature table into train / valid / test, once.

WHY THIS RUNS AFTER THE FEATURE BUILDER, not before. The builder drops the
leading warm-up rows -- `date_id` 0 to 71, where the longest rolling windows have
no history -- so the raw series has 8990 rows and the feature table has 8918. The
fold grid is positional on the feature table, so a split computed on the raw
index lands 58 rows away from where the models actually cut. The split therefore
has to be taken on the table the models read.

WHY EVERY LATER SCRIPT READS THESE FILES AND NOT THE FEATURE TABLE. `test` is
then unreachable unless a script names it: `paths.load_split("train")` and
`paths.load_split("valid")` are the only calls anything before the final scoring
needs to make.

WHAT EACH FILE CONTAINS. A span's model trains on rows that come before the span
starts, so each file carries its own scored rows **plus the history those models
need** -- the training window, the purge and the embargo. The three files
therefore overlap by 757 rows each, and any one of them is enough to run the full
walk-forward for its span without seeing the others.

NO COLUMNS ARE ADDED. Each file is an exact slice of the feature table, so
`feature_cols` and everything downstream sees the column set it always saw. Where
each slice begins is recorded in `manifest.json`, and `load_span` uses that one
offset to put the folds back where they belong.

THE GEOMETRY IS DERIVED, NOT WRITTEN DOWN. It comes from `wfo.make_folds` under
the frozen constants, so it cannot drift from what the models do:

    train    every fold whose last scored row is before dev_end
    holdout  every fold starting at or after dev_end + HOLDOUT_GAP
    valid    the first half of the holdout folds, by fold ordinal
    test     the second half

    python split_data.py            # plan: the ranges, nothing written
    python split_data.py --commit   # write the three files and the manifest
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

import paths
from hull_probe import (
    HOLDOUT_FRAC, HOLDOUT_GAP, START_AT, TEST_WINDOW, TRAIN_WINDOW,
)
from wfo import make_folds

PURGE = 1
EMBARGO = 0                # the protocol in force; see docs/METHOD.md


def fold_sets(n: int) -> Dict[str, list]:
    """The three spans as lists of folds. One definition, used everywhere."""
    dev_end = n - int(round(n * HOLDOUT_FRAC)) - HOLDOUT_GAP
    start = dev_end + HOLDOUT_GAP
    every = make_folds(n, TRAIN_WINDOW, TEST_WINDOW, embargo=EMBARGO,
                       purge=PURGE, start_at=START_AT)
    hold = [f for f in every if int(f.test[0]) >= start]
    cut = len(hold) // 2
    return {"train": [f for f in every if int(f.test[-1]) < dev_end],
            "valid": hold[:cut], "test": hold[cut:]}


def ranges(n: int) -> Dict[str, dict]:
    """
    For each span: the rows it needs, and which of them it scores.

    `lo` is the earliest row any of its folds trains on. `hi` is the last row the
    span owns: the development boundary for train, the end of the table for test,
    and its own last scored row for valid.
    """
    dev_end = n - int(round(n * HOLDOUT_FRAC)) - HOLDOUT_GAP
    out = {}
    for k, folds in fold_sets(n).items():
        scored = np.concatenate([f.test for f in folds])
        # train runs to the development boundary, not merely to its own last
        # scored row: other scripts build their own fold grids on this slice --
        # a shorter window, the older embargo -- and those score further into the
        # development span. Every row below dev_end is a development row anyway.
        # test runs to the end so no row is silently dropped.
        hi = {"train": dev_end - 1, "test": n - 1}.get(k, int(scored.max()))
        out[k] = {"lo": int(min(int(f.train[0]) for f in folds)), "hi": hi,
                  "n_folds": len(folds),
                  "scored_lo": int(scored.min()), "scored_hi": int(scored.max()),
                  "n_scored": int(len(scored)), "scored_rows": scored.tolist()}
        out[k]["n_rows"] = out[k]["hi"] - out[k]["lo"] + 1
    return out


def span_offset(span: str) -> int:
    """Where a span's slice begins in the feature table."""
    return int(_manifest()["spans"][span]["lo"])


def scoring_folds() -> Dict[str, list]:
    """
    The three spans' folds on the FULL table index, under the names the document
    uses: `dev` is the training span.

    For scripts that score all three spans as one contiguous series -- a trailing
    percentile needs the rows either side of a span boundary -- rather than
    working inside one slice. They read cached signals, not features, so they
    never touch another span's rows.
    """
    fs = fold_sets(table_rows())
    return {"dev": fs["train"], "valid": fs["valid"], "test": fs["test"]}


def load_through(span: str):
    """
    Every row from the first training row through the end of `span`, deduplicated,
    with that span's folds indexed into it.

    For protocols that train on an EARLIER span and predict this one -- fitting
    once on all of development and predicting valid, for instance, which needs
    6636 training rows while valid's own slice carries 757. It reaches back
    deliberately and says so; the folds returned still score only `span`'s own
    rows, and nothing later than `span` is read.

        df, folds, lo = load_through("valid")
    """
    from wfo import Fold

    order = list(paths.SPANS)
    upto = order[:order.index(span) + 1]
    frames = [paths.load_split(k) for k in upto]
    man = _manifest()
    los = [man["spans"][k]["lo"] for k in upto]
    lo = min(los)

    rows, keep = set(), []
    for k, d, l in zip(upto, frames, los):
        idx = np.arange(l, l + len(d))
        m = ~np.isin(idx, list(rows))
        rows.update(idx[m].tolist())
        keep.append(d.iloc[np.where(m)[0]])
    df = pd.concat(keep, ignore_index=True)
    assert len(df) == max(rows) - lo + 1, "the concatenated spans are not contiguous"

    folds = [Fold(f.k, f.train - lo, f.test - lo) for f in fold_sets(man["rows"])[span]]
    bad = [f.k for f in folds if f.train.min() < 0 or f.test.max() >= len(df)]
    if bad:
        raise SystemExit(f"folds {bad[:5]} reach outside rows {lo}..{max(rows)}")
    return df, folds, lo


def slice_covering(row: int) -> str:
    """
    The earliest span whose `load_through` reaches `row`.

    For scripts whose own fold grid does not line up with the protocol's. At the
    protocol's embargo of 0 every span's blocks land inside its own slice and
    this returns that slice; a grid built at another embargo moves every block,
    so a span's blocks can fall past the end of the slice that nominally owns
    them. Naming the row and asking for the smallest slice that holds it reads
    exactly as far as the grid requires and no further.
    """
    man = _manifest()
    for k in paths.SPANS:
        if row <= man["spans"][k]["hi"]:
            return k
    raise SystemExit(f"row {row} is past the end of the feature table "
                     f"({man['rows']} rows)")


def load_span(span: str):
    """
    A span's rows and its folds, with the folds indexed into the frame returned.

    This is the entry point every script after Stage 2 uses. The frame is an
    exact slice of the feature table -- no extra columns -- holding the span's
    scored rows and the history its models need. The folds are the ones
    `wfo.make_folds` produces on the whole table, shifted by where the slice
    begins, so `f.train` and `f.test` index straight into it.

        df, folds = load_span("train")
        for f in folds:
            fit(df.iloc[f.train]); predict(df.iloc[f.test])
    """
    from wfo import Fold

    man = _manifest()
    lo, expect = man["spans"][span]["lo"], man["spans"][span]["n_rows"]

    df = paths.load_split(span).reset_index(drop=True)
    if len(df) != expect:
        raise SystemExit(
            f"{paths.split_processed(span).name} has {len(df)} rows, manifest "
            f"says {expect}; re-run split_data.py --commit")

    folds = [Fold(f.k, f.train - lo, f.test - lo)
             for f in fold_sets(man["rows"])[span]]
    assert all(f.train.min() >= 0 and f.test.max() < len(df) for f in folds),         f"a fold reaches outside {span}_processed.csv"
    return df, folds


def _manifest() -> dict:
    p = paths.SPLIT_DIR / "manifest.json"
    if not p.exists():
        raise SystemExit(f"not found: {p}. Run split_data.py --commit.")
    with open(p) as fh:
        return json.load(fh)


def table_rows() -> int:
    """How many rows the whole feature table has, without reading it."""
    return int(_manifest()["rows"])


def remap_rows(folds, span: str):
    """
    Put folds built on the whole feature table onto positions in a span's slice.

    For scripts that build their own fold grid -- a different window length, a
    different embargo -- rather than taking the span's own from load_span. The
    bounds come from the manifest, so the slice does not have to be in memory.
    """
    from wfo import Fold

    sp = _manifest()["spans"][span]
    lo, n = sp["lo"], sp["n_rows"]
    out = [Fold(f.k, f.train - lo, f.test - lo) for f in folds]
    bad = [f.k for f in out if f.train.min() < 0 or f.test.max() >= n]
    if bad:
        raise SystemExit(
            f"folds {bad[:5]} reach outside {span}_processed.csv, which covers "
            f"rows {lo}..{lo + n - 1}; that slice does not carry enough history")
    return out


def main() -> None:
    commit = "--commit" in sys.argv
    if not paths.FEATURE_PATH.exists():
        raise SystemExit(f"not found: {paths.FEATURE_PATH}\n"
                         f"Run Hull_Tactical_feature_engineering.py first.")
    feat = pd.read_csv(paths.FEATURE_PATH)
    n = len(feat)
    rg = ranges(n)

    print("=" * 94)
    print("SPLIT -- train fits, valid selects, test is held out")
    print("=" * 94)
    print(f"   {paths.FEATURE_PATH.name}: {n} rows x {feat.shape[1]} columns, "
          f"date_id {int(feat['date_id'].iloc[0])}..{int(feat['date_id'].iloc[-1])}")
    print(f"   window {TRAIN_WINDOW}, block {TEST_WINDOW}, purge {PURGE}, "
          f"embargo {EMBARGO}, start_at {START_AT}")
    print(f"   dev_end {n - int(round(n * HOLDOUT_FRAC)) - HOLDOUT_GAP}, "
          f"holdout gap {HOLDOUT_GAP}\n")
    print(f"   {'span':<8}{'rows kept':>15}{'scored':>19}{'folds':>7}{'history':>9}")
    for k in paths.SPANS:
        r = rg[k]
        print(f"   {k:<8}{r['lo']:>6}-{r['hi']:<8}{r['scored_lo']:>10}-"
              f"{r['scored_hi']:<8}{r['n_folds']:>7}{r['scored_lo'] - r['lo']:>9}")
    total = sum(rg[k]["n_rows"] for k in paths.SPANS)
    print(f"\n   {total} rows written against {n} in the table ({total / n:.2f}x); "
          f"the overlap is the history each span's models need")

    scored_all = np.concatenate([rg[k]["scored_rows"] for k in paths.SPANS])
    assert len(set(scored_all.tolist())) == len(scored_all), \
        "a row is scored by two spans"
    covered = set()
    for k in paths.SPANS:
        covered |= set(range(rg[k]["lo"], rg[k]["hi"] + 1))
    assert covered == set(range(rg["train"]["lo"], n)), \
        "the three ranges leave a gap between the first training row and the end"
    print(f"   checks: {len(scored_all)} scored rows, none scored twice, "
          f"no gap in coverage")

    if not commit:
        print("\n   --plan only. Re-run with --commit.")
        return

    os.makedirs(paths.SPLIT_DIR, exist_ok=True)
    for k in paths.SPANS:
        r = rg[k]
        d = feat.iloc[r["lo"]:r["hi"] + 1]
        p = paths.split_processed(k)
        d.to_csv(p, index=False)
        print(f"   wrote {p.name:<24} {len(d):>5} rows, "
              f"{r['n_scored']:>5} scored, {d.shape[1]} columns, "
              f"{p.stat().st_size / 1048576:.0f} MB")

    with open(paths.SPLIT_DIR / "manifest.json", "w") as f:
        json.dump({"feature_table": paths.FEATURE_PATH.name, "rows": n,
                   "constants": {"train_window": TRAIN_WINDOW,
                                 "test_window": TEST_WINDOW, "purge": PURGE,
                                 "embargo": EMBARGO, "start_at": START_AT,
                                 "holdout_frac": HOLDOUT_FRAC,
                                 "holdout_gap": HOLDOUT_GAP},
                   "spans": {k: {kk: vv for kk, vv in rg[k].items()
                                 if kk != "scored_rows"} for k in paths.SPANS}},
                  f, indent=2)
    print("   wrote manifest.json")
    print("""
   From here every script reads one of these through paths.load_split(span).
   Nothing before the final scoring needs to name "test".""")


if __name__ == "__main__":
    main()
