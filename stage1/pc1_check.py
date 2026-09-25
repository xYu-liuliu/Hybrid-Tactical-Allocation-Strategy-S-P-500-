"""
pc1_check.py -- is the group PCA still producing a factor, or a reshuffle?

add_group_pcas refits PCA at every row on whatever columns are ACTIVE in that
row's window:

    active = ~isnan[a:b].all(axis=0)

While first_valid_by_prefix dropped every late-starting feature, that set was
effectively constant and PC1 behaved like a slow factor. With the filter
loosened to MIN_COVERAGE = 0.30 the late starters are back, so the active set
now grows several times per group as they come online. Each join changes the
covariance the PCA is fitted on, which can move PC1's loadings, its scale, and
-- the one the rolling z-score cannot absorb -- its sign.

That matters beyond the six PC1 columns: V_pc1_z126 and E_pc1_z252 are the
gate sources, so a PC1 that degrades silently degrades ~136 gated columns.

Two tests:

  PERSISTENCE   lag-1 autocorrelation, sign-change rate and jump rate, against
                the previous feature file.

  JOIN SHOCK    at every row where a column enters a group's PCA input, where
                the local |d PC1| sits in that series' own distribution.

                The null is NOT 50. Taking the largest of k neighbouring days
                puts the percentile at k/(k+1) even when the join does nothing,
                so the reading is the p-value 1 - (pct/100)^k, and Fisher's
                combination over the joins of a group. Comparing the raw
                percentile against 50 flags every group.

    python pc1_check.py                 # the development split
    python pc1_check.py <feature.csv>   # any feature file
"""

from __future__ import annotations

# The repo keeps entry scripts in stage1/ and stage2/ and shared modules in
# tools/. Put all three on sys.path so every `import <module>` below resolves
# no matter which directory the script is launched from.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / d)
                 for d in ("", "tools", "stage1", "stage2")]

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import paths
from Hull_Tactical_feature_engineering import PCA_PLAN, group_of

try:
    from scipy import stats as _st
except ImportError:
    _st = None

# The split is not recomputed here. split_data.py wrote train_processed.csv and
# that file is the development span -- features only, never the target, so
# looking past it could not bias a performance estimate, but one definition of
# the split beats two.


# Rows searched around a join. The discontinuity is at the join itself; the
# window only covers the off-by-one in when a column first enters the window.
NEIGHBOURHOOD = (-1, 2)
K = NEIGHBOURHOOD[1] - NEIGHBOURHOOD[0]
NULL_PCT = 100.0 * K / (K + 1.0)


def persistence(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 50:
        return {}
    d = s.diff().dropna()
    return {
        "n": len(s),
        "ac1": float(s.autocorr(1)),
        "sign_flip_pct": float((np.sign(s.values[1:]) != np.sign(s.values[:-1])).mean() * 100),
        "jump_pct": float((d.abs() > 2 * d.std()).mean() * 100),
    }


def join_rows(df: pd.DataFrame, group: str, z_input: int) -> list[tuple[int, str]]:
    """(row, column) for each column that enters a group's PCA input after the first."""
    cols = [c for c in df.columns if group_of(c) == group and c.endswith(f"_z{z_input}")]
    at: dict[int, str] = {}
    for c in cols:
        if df[c].notna().any():
            at.setdefault(int(df[c].first_valid_index()), c)
    rows = sorted(at)
    return [(r, at[r]) for r in rows[1:]]   # the earliest is the baseline set, not a join


def join_shock(pc1: pd.Series, rows: list[tuple[int, str]]) -> tuple[list[float], list[float], float]:
    """(percentiles, per-join p-values, Fisher-combined p) for the local |d PC1|."""
    d = pc1.diff().abs()
    ref = d.dropna().to_numpy()
    if len(ref) < 50 or not rows:
        return [], [], np.nan

    pct, pv = [], []
    for r, _ in rows:
        w = d.iloc[max(r + NEIGHBOURHOOD[0], 0):r + NEIGHBOURHOOD[1]].dropna()
        if not len(w):
            continue
        p = float((ref < w.max()).mean())
        pct.append(p * 100)
        pv.append(max(1.0 - p ** K, 1e-12))

    if not pv:
        return [], [], np.nan
    if _st is not None:
        comb = float(_st.combine_pvalues(pv, method="fisher")[1])
    else:
        comb = float(min(min(pv) * len(pv), 1.0))     # Bonferroni fallback
    return pct, pv, comb


def report(path: Path, base: pd.DataFrame | None) -> None:
    df = pd.read_csv(path) if path else paths.load_split("train")
    n_all = len(df)
    pc1_cols = [c for c in df.columns if c.endswith("_pc1")]
    gated = [c for c in ("V_pc1_z126", "E_pc1_z252") if c in df.columns]

    name = path.name if path else paths.split_processed("train").name
    print()
    print(f"{name}: {n_all} rows "
          f"(the development span; valid and test are separate files), "
          f"{df.shape[1]} columns, {len(pc1_cols)} PCA groups")

    print(f"\n{'column':<13}{'ac1':>8}{'sign flip':>11}{'jumps':>8}   vs previous file")
    for c in pc1_cols + gated:
        p = persistence(df[c])
        if not p:
            print(f"{c:<13}  (too short)")
            continue
        delta = ""
        if base is not None and c in base.columns:
            b = persistence(base[c])
            if b:
                delta = f"   ac1 {b['ac1']:.3f} -> {p['ac1']:.3f}  ({p['ac1'] - b['ac1']:+.3f})"
        flag = "  <-- CHECK" if p["ac1"] < 0.60 else ""
        print(f"{c:<13}{p['ac1']:>8.3f}{p['sign_flip_pct']:>10.1f}%{p['jump_pct']:>7.1f}%{delta}{flag}")

    print(f"\njoin shock: local |d PC1| at rows where a column enters the PCA")
    print(f"  window = {K} rows, so under the null the percentile sits at {NULL_PCT:.0f}, not 50")
    print(f"\n{'group':<7}{'joins':>6}   {'percentiles':<22}{'p per join':<24}{'Fisher p':>9}")
    for g, cfg in PCA_PLAN.items():
        col = f"{g}_pc1"
        if col not in df.columns:
            continue
        rows = join_rows(df, g, cfg["z_input"])
        pct, pv, comb = join_shock(df[col], rows)
        if not pct:
            print(f"{g:<7}{0:>6}   (no joins)")
            continue
        flag = "  <-- CHECK" if np.isfinite(comb) and comb < 0.05 else ""
        who = ", ".join(f"{c.split('_')[0]}@{r}" for r, c in rows[:len(pct)])
        print(f"{g:<7}{len(pct):>6}   {', '.join(f'{p:.0f}' for p in pct):<22}"
              f"{', '.join(f'{p:.2f}' for p in pv):<24}{comb:>9.3f}{flag}")
        print(f"{'':<7}{'':>6}   joined by: {who}")

    print(f"""
Reading it:
  ac1 above ~0.8 and no Fisher p below 0.05
      -> the PCA survives the longer window, nothing to do.
  ac1 collapsing, or a group's Fisher p below 0.05
      -> the changing active set is driving that group's PC1. Simplest fix is
         to feed each group only the columns present over the whole window and
         let the late starters reach the model directly; a second option is to
         carry the active-column count beside each PC1 so the tree can
         condition on it. Settle this before tuning anything else -- V_pc1_z126
         and E_pc1_z252 multiply into ~136 gate columns.

  A single join at the largest move in the whole series gives p = {1 - 0.999 ** K:.3f},
  so one join CAN be significant on its own -- but six groups are tested here,
  so multiply a lone p by 6 before believing it.
""")


def main() -> None:
    # No argument: read the development split, which is what this checks.
    # An argument overrides it, for comparing an arbitrary feature table.
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if path is not None and not path.exists():
        raise SystemExit(f"not found: {path}")

    # an older feature table, if one is lying about, as a before/after baseline
    prev = sorted(
        (q for q in paths.KAGGLE_HULL.glob("all_feature_last_*.csv")
         if q != paths.FEATURE_PATH),
        key=lambda q: q.stat().st_mtime, reverse=True,
    )
    base = None
    if prev:
        keep = [c for c in pd.read_csv(prev[0], nrows=0).columns if "_pc1" in c]
        if keep:
            base = pd.read_csv(prev[0], usecols=keep)
            print(f"baseline: {prev[0].name}")
    report(path, base)


if __name__ == "__main__":
    main()
