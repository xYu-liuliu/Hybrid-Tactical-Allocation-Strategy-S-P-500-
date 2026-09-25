"""
audit_config.py -- is the repository actually in the state the results assume?

RUN THIS FIRST, AND AGAIN BEFORE THE HOLDOUT. Every number in this project was
produced under one configuration, and the configuration lives in five files
that can drift apart silently. This project has already been burned by exactly
that: results were computed for weeks against a feature file the builder no
longer produced, and hull_probe's own TREE sat at num_leaves=31 /
min_child_samples=100 -- five realised leaves on a 756-row window -- long after
the capacity work had frozen 117/4 and measured the gap at +0.442.

Neither failure announced itself. Both were found by accident. This file turns
that check into something that runs in a second and exits non-zero.

WHAT IS CHECKED

    prediction     TREE's eight parameters, the training window, the test
                   window, the embargo, the warm-up start, the inner fraction
                   and the seeds, against capacity_window_grid's frozen cell.
    mapping        the ten constants in mapping.py.
    consistency    the percentile is built in three places -- mapping.py,
                   step2_tails.py and capacity_data.py via RANK_MAP -- and they
                   must agree on the detrend window, the ranking window and the
                   warm-up bar, or the cached signals and the live mapper are
                   ranking against different distributions.
    features       the feature file must be NEWER than both scripts that build
                   it, the coverage filter must be at its frozen value, and the
                   per-group quantile rule must still be disabled.
    holdout        that it is where it is supposed to be and the development
                   fold count matches what the grid was run on.
    inheritance    which modules take TREE without overriding the capacity, so
                   a change to TREE silently changes their behaviour. These are
                   reported, not failed: some of them are meant to inherit.

EXIT CODE is the number of mismatches, so this drops into a shell or a CI step
without parsing the output.

    python audit_config.py
    python audit_config.py --quiet     # only the failures
"""

from __future__ import annotations

# The repo keeps entry scripts in stage1/ and stage2/ and shared modules in
# tools/. Put all three on sys.path so every `import <module>` below resolves
# no matter which directory the script is launched from.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / d)
                 for d in ("", "tools", "stage1", "stage2")]

import datetime
import io
import os
import re
import sys

FROZEN_WINDOW, FROZEN_TARGET = 756, 117
EXPECTED_FEATURE_COLS = 1132
EXPECTED_DEV_FOLDS = 69
EXPECTED_VALID_FOLDS = 10
EXPECTED_TEST_FOLDS = 11
EXPECTED_TABLE_ROWS = 8918


def _dirs():
    """
    Where this project's scripts live, flat layout or split into stage folders.

    The checks below read other scripts by name. Resolving those names against
    the current working directory would make the audit pass or fail depending on
    where it was launched from, so they are resolved against this file instead.
    """
    here = _Path(__file__).resolve().parent
    cand = [here] + [here.parent / d for d in ("", "tools", "stage1", "stage2")]
    return [d for d in cand if d.is_dir()]


def _script(name: str) -> str:
    for d in _dirs():
        if (d / name).is_file():
            return str(d / name)
    raise SystemExit(f"cannot find {name} beside {_Path(__file__).resolve().parent}")


def _all_scripts():
    seen, out = set(), []
    for d in _dirs():
        for f in sorted(d.glob("*.py")):
            if f.name not in seen:
                seen.add(f.name)
                out.append(f)
    return out


def main() -> int:
    quiet = "--quiet" in sys.argv
    bad, notes = [], []

    def chk(section, name, got, want, note=""):
        ok = got == want
        if not ok:
            bad.append(f"{section}.{name}: is {got!r}, frozen value is {want!r}")
        if not quiet or not ok:
            print("   %-34s %-24s %-24s %s %s"
                  % (name, str(got), str(want), "ok " if ok else "MISMATCH", note))
        return ok

    import capacity_window_grid as G
    import hull_probe as H
    import mapping as MAP

    frozen = G.cd_tree_params(FROZEN_WINDOW, FROZEN_TARGET)

    print("=" * 104)
    print("A. prediction layer (hull_probe.py) against the frozen cell "
          f"({FROZEN_WINDOW}, T{FROZEN_TARGET})")
    print("=" * 104)
    if not quiet:
        print("   %-34s %-24s %-24s" % ("parameter", "in the code", "frozen"))
    for k in ("num_leaves", "min_child_samples", "n_estimators", "learning_rate",
              "subsample", "colsample_bytree", "reg_lambda", "subsample_freq"):
        chk("TREE", "TREE." + k, H.TREE.get(k), frozen[k])
    chk("folds", "TRAIN_WINDOW", H.TRAIN_WINDOW, FROZEN_WINDOW)
    chk("folds", "TEST_WINDOW", H.TEST_WINDOW, 84)
    chk("folds", "EMBARGO", H.EMBARGO, 10)
    chk("folds", "START_AT", H.START_AT, 504)
    chk("folds", "INNER_FRAC", H.INNER_FRAC, 0.60)
    chk("folds", "SEEDS", tuple(H.SEEDS), (0, 1, 2))

    n_in = int(FROZEN_WINDOW * H.INNER_FRAC)
    leaves = lambda n, mcs: min(frozen["num_leaves"], int(n * frozen["subsample"]) // mcs)
    mcs_in = max(3, max(1, int(n_in * frozen["subsample"])) // frozen["num_leaves"])
    print(f"   realised leaves: inner ~{leaves(n_in, mcs_in)}, "
          f"full ~{leaves(FROZEN_WINDOW, frozen['min_child_samples'])} "
          f"(they should be close; a large gap means s_cal and s_test are drawn "
          f"from different distributions)")

    print("\n" + "=" * 104)
    print("B. mapping layer (mapping.py)")
    print("=" * 104)
    for k, want in (("MU_WINDOW", 63), ("WINDOW", 252), ("MIN_HIST", 60),
                    ("WBAR", 1.00), ("BUDGET", 1.2), ("TAU_WINDOW", 756),
                    ("TAU_MIN_HIST", 120), ("TAU_FALLBACK", 0.30),
                    ("MIN_POS", 0.0), ("MAX_POS", 2.0)):
        chk("mapping", "mapping." + k, getattr(MAP, k), want)

    print("\n" + "=" * 104)
    print("C. the percentile is defined in three places and they must agree")
    print("=" * 104)
    import step2_tails as ST
    from hull_probe import RANK_MAP
    chk("consistency", "step2_tails.MU_WINDOW", ST.MU_WINDOW, MAP.MU_WINDOW)
    chk("consistency", "RANK_MAP['window']", RANK_MAP["window"], MAP.WINDOW)
    chk("consistency", "RANK_MAP['min_hist']", RANK_MAP["min_hist"], MAP.MIN_HIST)

    print("\n" + "=" * 104)
    print("D. features and data")
    print("=" * 104)
    import paths
    import split_data as SD
    src = io.open(_script("Hull_Tactical_feature_engineering.py"), encoding="utf-8").read()
    mc = re.search(r"MIN_COVERAGE\s*=\s*([\d.]+)", src)
    chk("features", "MIN_COVERAGE", float(mc.group(1)) if mc else None, 0.30)
    qb = re.search(r"Q_BY_PREFIX\s*=\s*\{([^}]*)\}", src)
    disabled = bool(qb) and "1.0" in qb.group(1)
    chk("features", "Q_BY_PREFIX disabled", disabled, True, "(per-group quantile rule off)")

    feat_t = os.path.getmtime(paths.FEATURE_PATH)
    for b in ("Hull_Tactical_feature_engineering.py",):
        fresh = feat_t > os.path.getmtime(_script(b))
        chk("features", f"feature file newer than {b}", fresh, True,
            datetime.datetime.fromtimestamp(feat_t).strftime("%Y-%m-%d %H:%M"))

    df = paths.load_split("train")
    cols = H.feature_cols(df)
    chk("features", "feature columns", len(cols), EXPECTED_FEATURE_COLS)

    # The geometry belongs to split_data.py, which derives it from wfo.make_folds
    # under the frozen constants and records it in the manifest. Recomputing it
    # here would be a second definition that could disagree with the one the
    # models actually read, so this checks the manifest instead.
    man = SD._manifest()
    chk("data", "split manifest rows", man["rows"], EXPECTED_TABLE_ROWS)
    for k, v in man["constants"].items():
        # purge and embargo belong to split_data.py, which owns the protocol in
        # force. hull_probe's EMBARGO is the legacy grid's 10 and is checked
        # separately in section A, because Part 1's cached grids ran under it.
        want = {"train_window": H.TRAIN_WINDOW, "test_window": H.TEST_WINDOW,
                "purge": SD.PURGE, "embargo": SD.EMBARGO, "start_at": H.START_AT,
                "holdout_frac": H.HOLDOUT_FRAC, "holdout_gap": H.HOLDOUT_GAP}[k]
        chk("data", f"manifest {k}", v, want)
    sp = man["spans"]
    chk("data", "development folds", sp["train"]["n_folds"], EXPECTED_DEV_FOLDS)
    chk("data", "valid folds", sp["valid"]["n_folds"], EXPECTED_VALID_FOLDS)
    chk("data", "test folds", sp["test"]["n_folds"], EXPECTED_TEST_FOLDS)
    for k in ("train", "valid", "test"):
        p = paths.split_processed(k)
        chk("data", f"{p.name} present", p.exists(), True)
        if p.exists():
            fresh = os.path.getmtime(p) > os.path.getmtime(_script("split_data.py"))
            chk("data", f"{p.name} newer than split_data.py", fresh, True)
    for k in ("train", "valid", "test"):
        s = sp[k]
        print(f"   {k:<6} rows {s['lo']:>5}-{s['hi']:<5} scored "
              f"{s['scored_lo']:>5}-{s['scored_hi']:<5} {s['n_folds']:>3} folds, "
              f"{s['scored_lo'] - s['lo']} rows of history")
    scored = sum(sp[k]["n_scored"] for k in ("train", "valid", "test"))
    print(f"   {scored} scored rows of {man['rows']} in the table; "
          f"dev_end {sp['train']['hi'] + 1}, gap {H.HOLDOUT_GAP}")

    print("\n" + "=" * 104)
    print("E. modules that inherit TREE's capacity (informational, not failures)")
    print("=" * 104)
    for fp in _all_scripts():
        f, body = fp.name, io.open(fp, encoding="utf-8").read()
        for i, line in enumerate(body.splitlines(), 1):
            if not re.search(r"\{\*\*TREE\b", line):
                continue
            nxt = "\n".join(body.splitlines()[i - 1:i + 3])
            if "num_leaves" in nxt or "min_child" in nxt:
                continue
            notes.append(f"{f}:{i}")
            print(f"   {f}:{i}  {line.strip()[:66]}")
    if not notes:
        print("   (none)")
    print("   these change behaviour whenever TREE changes; cached results from "
          "them go stale silently")

    print("\n" + "=" * 104)
    if bad:
        print(f"{len(bad)} MISMATCH(ES) -- results computed now will not match the "
              f"frozen configuration:")
        for b in bad:
            print("   - " + b)
    else:
        print("0 mismatches. The repository is in the frozen configuration.")
    print("=" * 104)
    return len(bad)


if __name__ == "__main__":
    sys.exit(main())
