# paths.py
# ========
# Single source of truth for every file this project reads or writes.
#
# Everything is resolved from this file's own location, so the scripts run from
# any working directory and nothing machine-specific is left in them.
#
# N_RAW lives here on purpose. The feature file name is derived from it, so a
# run at one window length cannot silently reuse a feature file built at
# another one -- which is exactly how the 5000-row file and the current
# Hull_Tactical_feature_engineering.py drifted apart.

import warnings
from pathlib import Path

def _find_root() -> Path:
    """
    The directory holding kaggle_hull/ and train.csv.

    Found by walking up from this file rather than assumed to be its own parent,
    so the scripts work whether they sit beside the data or two levels down in a
    stage1/ stage2/ tools/ layout. Falls back to this file's parent, which is
    what a flat checkout gives.
    """
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "kaggle_hull").is_dir() or (d / "train.csv").is_file():
            return d
    return here.parent


ROOT = _find_root()


# ============================================================
# Input
# ============================================================

TRAIN_CSV = ROOT / "train.csv"


# ============================================================
# Window length
# ============================================================
# 5000 : the short window used up to now
# 8990 : the whole of train.csv  <- current
#
# Changing this one value moves FEATURE_PATH and RUN_DIR together, so the
# feature file, the model outputs and the row count can never disagree.
#
# The feature filter no longer depends on this value in a surprising way:
# Hull_Tactical_feature_engineering.py now keeps a raw feature when it covers at
# least MIN_COVERAGE = 0.30 of the window, and the per-group quantile rule is
# off. At 5000 that drops nothing (94 raw features); at 8990 it drops only E7.

N_RAW = 8990


# ============================================================
# Outputs
# ============================================================

KAGGLE_HULL = ROOT / "kaggle_hull"

# written by Hull_Tactical_feature_engineering.py, read by everything else
FEATURE_PATH = KAGGLE_HULL / f"all_feature_last_{N_RAW}.csv"

# runs from the earlier feature file, kept for reference; nothing here reads them
RUN_DIR = KAGGLE_HULL / str(N_RAW)
RFF_DIR = RUN_DIR / "rff_check"

# hull_probe.py / select_probe.py / step2_tails.py
#
# NOTE: this directory is NOT scoped by N_RAW, so rerunning the probe at a
# different window length overwrites the 31-fold results already in it. Back it
# up first if you want to keep them for comparison.
PROBE_DIR = KAGGLE_HULL / "probe"

# ============================================================
# Splits
# ============================================================
# split_data.py cuts the feature table into three here. Every script after it
# reads one of these and never the whole table, so `test` cannot be touched by
# accident -- a script has to name it.

SPLIT_DIR = KAGGLE_HULL / "splits"
SPANS = ("train", "valid", "test")


def split_processed(span: str):
    """The same rows with all 1132 engineered columns."""
    if span not in SPANS:
        raise ValueError(f"span must be one of {SPANS}, got {span!r}")
    return SPLIT_DIR / f"{span}_processed.csv"


def load_split(span: str):
    """
    Read one span: an exact slice of the feature table, no columns added.

    Prefer `split_data.load_span(span)`, which also returns the folds already
    shifted onto this slice. Use this only when the folds are not needed.
    """
    import pandas as pd
    p = split_processed(span)
    if not p.exists():
        raise SystemExit(f"not found: {p}. Run "
                         f"Hull_Tactical_feature_engineering.py, then "
                         f"split_data.py --commit.")
    return pd.read_csv(p)



def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for d in (KAGGLE_HULL, RUN_DIR, RFF_DIR, PROBE_DIR, SPLIT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Staleness guard
# ============================================================

def _warn_if_feature_file_stale() -> None:
    """
    Warn when the feature file predates the module that builds it.

    This is the check that was missing: the 5000-row feature file was written
    44 minutes before the last edit to Hull_Tactical_feature_engineering.py, so
    every downstream result was computed on a column set the current code does
    not produce (72 D-group columns instead of 18, no E_pc1 / P_pc1, and the
    gate_E block silently doing nothing because E_pc1_z252 did not exist).

    A warning rather than an exception: it must never break a run, only make
    the mismatch impossible to miss.
    """
    fe = ROOT / "Hull_Tactical_feature_engineering.py"
    if not (FEATURE_PATH.exists() and fe.exists()):
        return
    if fe.stat().st_mtime > FEATURE_PATH.stat().st_mtime:
        warnings.warn(
            f"\n  {FEATURE_PATH.name} is OLDER than {fe.name}.\n"
            f"  The feature file may not match the current build_features().\n"
            f"  Rerun Hull_Tactical_feature_engineering.py before trusting any result.",
            stacklevel=2,
        )


_warn_if_feature_file_stale()
