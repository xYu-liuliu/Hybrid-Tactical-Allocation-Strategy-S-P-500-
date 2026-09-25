# Hull_Tactical_feature_engineering.py
# ====================================
# Stage 1: build the feature table.
#
# Everything above main() is a pure transformation of a DataFrame; main() is the
# only place that touches disk, and it reads and writes the two paths named in
# paths.py. Run it directly to produce the feature file every other script reads:
#
#     python Hull_Tactical_feature_engineering.py

# The repo keeps entry scripts in stage1/ and stage2/ and shared modules in
# tools/. Put all three on sys.path so every `import <module>` below resolves
# no matter which directory the script is launched from.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / d)
                 for d in ("", "tools", "stage1", "stage2")]

import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA


# ============================================================
# 0. Column groups
# ============================================================
# Raw features are named <group><number>: M1, MOM3, D7, ...
# "MOM" is matched before "M", so momentum columns never fall into the M group.
# Derived columns (M1_z21, MOM3_d1, ...) inherit the group of their base column.
# M_pc1*, XINT_*, lagged_*, labels and meta columns have no group.

GROUPS = ("MOM", "M", "E", "I", "P", "V", "S", "D")
_GROUP_RE = re.compile(r"^(MOM|M|E|I|P|V|S|D)\d+")
_RAW_RE = re.compile(r"(MOM|M|E|I|P|V|S|D)\d+")


def group_of(col: str) -> Optional[str]:
    m = _GROUP_RE.match(col)
    return m.group(1) if m else None


def is_raw_feature(col: str) -> bool:
    return _RAW_RE.fullmatch(col) is not None


# ============================================================
# 1. Lagged block construction
# ============================================================

LAGGED_COLS = (
    "lagged_forward_returns",
    "lagged_risk_free_rate",
    "lagged_market_forward_excess_returns",
)


def build_lagged_block(
    df: pd.DataFrame,
    wins=(2, 3, 5, 10, 21, 63),
    excess_wins=(2, 3, 5, 10),
    make_diff=True,
):
    # A 1-day window is skipped: its sample std is always NaN
    # and its mean is identical to the raw column.
    out = df.copy()

    if {"lagged_forward_returns", "lagged_risk_free_rate"}.issubset(out.columns):
        out["lagged_excess"] = out["lagged_forward_returns"] - out["lagged_risk_free_rate"]

    plan = [(c, wins) for c in LAGGED_COLS if c in out.columns]
    if "lagged_excess" in out.columns:
        plan.append(("lagged_excess", excess_wins))

    new = {}
    for col, wlist in plan:
        s = out[col]
        for w in wlist:
            if w < 2:
                continue
            r = s.rolling(w, min_periods=w)
            mu = r.mean()
            sd = r.std()
            new[f"{col}_mean{w}"] = mu
            new[f"{col}_std{w}"] = sd
            new[f"{col}_z{w}"] = (s - mu) / (sd + 1e-9)
        if make_diff:
            new[f"{col}_d1"] = s.diff()

    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


# ============================================================
# 2. First-valid filtering by group
# ============================================================

PREFIXES = GROUPS
# Per-group quantile rule, disabled: 1.0 makes the cut the group maximum, which
# nothing can exceed. It had no absolute meaning -- it always removed the
# latest-starting ~10% of each group however healthy they were. At the full
# 8990-row window it dropped P6 (82.4% coverage) and P7 (82.7%) while keeping
# P5 (83.2%), a knife edge inside the strongest group in the IC panel.
Q_BY_PREFIX = {p: 1.0 for p in PREFIXES}

# Keep a raw feature when it has data over at least this fraction of the window.
#
# Replaces MAX_START_FRAC = 0.25, which was the same rule written from the other
# end: "drop if it starts after 25% of the window" == "keep if coverage >= 75%".
#
# 0.30 is deliberately loose, because the walk-forward evaluation already makes
# this call per fold and makes it better: LightGBM never splits on a column that
# is all-NaN in its training window, and rank_features_in_window gives it zero
# gain. A construction-time drop removes the feature from EVERY fold -- at the
# 8990-row length P6 and P7 have a FULL training window in 56 of 69 folds, and
# the old rule killed them for all 69.
#
# At 0.30 the only feature that goes is E7 (22.5% coverage; not one of the 69
# folds reaches even 50% training coverage), which is genuinely unusable.
MIN_COVERAGE = 0.30


def first_valid_by_prefix(df: pd.DataFrame):
    """
    Drop grouped raw features that start too late or are entirely NaN.
    Columns without a group (labels, lagged_*, date_id, ...) are always kept.

    Returns (fv_table, per-group summary, cleaned frame).
    """
    rows = []
    for c in df.columns:
        g = group_of(c)
        if g is None:
            continue
        idx = df[c].first_valid_index()
        rows.append({
            "prefix": g,
            "column": c,
            "first_valid_row": int(idx) if idx is not None else np.nan,
        })

    fv = pd.DataFrame(rows)
    if fv.empty:
        return fv, pd.DataFrame(), df.copy()

    cuts = {}
    for p, g in fv.groupby("prefix"):
        vals = g["first_valid_row"].dropna()
        cuts[p] = int(vals.quantile(Q_BY_PREFIX[p])) if len(vals) else np.nan
    fv["cut_q"] = fv["prefix"].map(cuts)

    fv["is_late_q"] = fv["first_valid_row"].notna() & fv["cut_q"].notna() & (
        fv["first_valid_row"] > fv["cut_q"]
    )
    fv["coverage"] = 1.0 - fv["first_valid_row"] / float(len(df))
    fv["is_thin"] = fv["first_valid_row"].notna() & (fv["coverage"] < MIN_COVERAGE)
    fv["is_all_nan"] = fv["first_valid_row"].isna()
    fv["is_late"] = fv["is_late_q"] | fv["is_thin"] | fv["is_all_nan"]

    drop_cols = set(fv.loc[fv["is_late"], "column"])
    data_clean = df[[c for c in df.columns if c not in drop_cols]].copy()

    summary = fv.groupby("prefix")["is_late"].agg(n_cols="size", n_dropped="sum")
    return fv, summary, data_clean


# ============================================================
# 3. Simple deterministic interactions
# ============================================================

INTERACTION_FEATURES = [
    ("I2", "I1", "diff"),
    ("I7", "I1", "diff"),
    ("P1", "E1", "ratio"),
    ("V1", "M1", "ratio"),
    ("S1", "S2", "diff"),
]


def add_simple_interactions(
    df: pd.DataFrame,
    interactions: Iterable[Tuple[str, str, str]],
    add_missing_flags: bool = False,
    eps: float = 1e-6,
    prefix: str = "XINT_",
) -> pd.DataFrame:
    """
    Hand-crafted interactions:
      diff : a minus b
      ratio: a / b, NaN when |b| <= eps
    """
    out = df.copy()

    for a, b, op in interactions:
        if a not in out.columns or b not in out.columns:
            continue

        if op == "diff":
            name = f"{prefix}{a}_{b}_diff"
            out[name] = out[a] - out[b]
            if add_missing_flags:
                out[f"{name}_miss"] = (out[a].isna() | out[b].isna()).astype("int8")

        elif op == "ratio":
            name = f"{prefix}{a}_{b}_ratio"
            num, denom = out[a], out[b]
            safe = (~denom.isna()) & (denom.abs() > eps)
            out[name] = np.where(safe, num / denom, np.nan)
            if add_missing_flags:
                out[f"{name}_denom_small"] = ((~denom.isna()) & (denom.abs() <= eps)).astype("int8")
                out[f"{name}_miss"] = (num.isna() | denom.isna()).astype("int8")

    return out


# ============================================================
# 4. Rolling feature engineering
# ============================================================

WIN = {
    "V":   (5, 21, 63),
    "S":   (5, 10, 21),
    "M":   (5, 10, 21, 63),
    "MOM": (5, 21),
    "E":   (63, 126, 252),
    "P":   (63, 126, 252),
    "I":   (21, 63),
}

# D columns are 0/1 dummies. z-score, range position and d1 carry little
# information for them, so only the raw value and a rolling mean
# (how often the flag was on recently) are kept.
D_MEAN_WINS = (63,)


def pick_base_numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if is_raw_feature(c) and is_numeric_dtype(df[c])]


def add_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    new = {}

    for c in pick_base_numeric_cols(df):
        g = group_of(c)
        s = df[c]

        if g == "D":
            for w in D_MEAN_WINS:
                new[f"{c}_mean{w}"] = s.rolling(w, min_periods=w).mean()
            continue

        for w in WIN[g]:
            r = s.rolling(w, min_periods=w)
            mean = r.mean()
            std = r.std()
            hi = r.max()
            lo = r.min()
            new[f"{c}_mean{w}"] = mean
            new[f"{c}_z{w}"] = (s - mean) / (std + 1e-9)
            new[f"{c}_pos{w}"] = (s - lo) / (hi - lo + 1e-12)

        new[f"{c}_d1"] = s.diff()

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# Groups whose *_mean* rolling columns are removed after step 8.
DROP_MEAN_GROUPS = ("P", "I", "S")


# ============================================================
# 5. Group PCA (causal, winsorized)
# ============================================================

def causal_pc1_onfly_winsor(
    df, cols,
    method="rolling", window=252, warmup=252,
    min_active=5, zscore_window=252, q=0.98,
    prefix_name="G",
):
    n = len(df)
    pc1 = np.full(n, np.nan)
    Xraw = df[cols].to_numpy(dtype=float)
    isnan = np.isnan(Xraw)

    for t in range(n):
        if method == "rolling":
            if t < max(warmup, window):
                continue
            a, b = t - window, t
        else:
            if t < warmup:
                continue
            a, b = 0, t

        # train on [a, t), transform row t: strictly causal
        active = ~isnan[a:b].all(axis=0)
        if active.sum() < min_active:
            continue

        X_tr = Xraw[a:b, active]
        lo = np.nanquantile(X_tr, 1 - q, axis=0)
        hi = np.nanquantile(X_tr, q, axis=0)
        X_tr = np.nan_to_num(np.clip(X_tr, lo, hi))

        x_t = np.nan_to_num(np.clip(Xraw[t:t + 1, active], lo, hi))

        pca = PCA(n_components=1)
        pca.fit(X_tr)
        pc1[t] = pca.transform(x_t)[0, 0]

    s_pc1 = pd.Series(pc1, index=df.index, name=f"{prefix_name}_pc1")

    mu = s_pc1.rolling(zscore_window, min_periods=1).mean()
    sd = s_pc1.rolling(zscore_window, min_periods=1).std().replace(0, np.nan)
    s_pc1_z = (s_pc1 - mu) / sd
    s_pc1_z.name = f"{prefix_name}_pc1_z{zscore_window}"

    return s_pc1, s_pc1_z


# z_input: which rolling z-score window feeds the PCA.
# E and P have no 21-day window in WIN, so they use their shortest one (63).
PCA_PLAN = {
    "V":   dict(method="rolling",   window=126, warmup=126, zwin=126, z_input=21),
    "M":   dict(method="rolling",   window=126, warmup=126, zwin=126, z_input=21),
    "MOM": dict(method="rolling",   window=63,  warmup=63,  zwin=126, z_input=21),
    "E":   dict(method="rolling",   window=252, warmup=252, zwin=252, z_input=63),
    "S":   dict(method="rolling",   window=126, warmup=126, zwin=252, z_input=21),
    "I":   dict(method="expanding",             warmup=504, zwin=252, z_input=21),
    "P":   dict(method="expanding",             warmup=504, zwin=252, z_input=63),
}


def add_group_pcas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for pref, cfg in PCA_PLAN.items():
        suffix = f"_z{cfg['z_input']}"
        cols = [c for c in out.columns if group_of(c) == pref and c.endswith(suffix)]
        if not cols:
            continue

        pc1, pc1_z = causal_pc1_onfly_winsor(
            out, cols=cols,
            method=cfg["method"],
            window=cfg.get("window", 252),
            warmup=cfg["warmup"],
            zscore_window=cfg["zwin"],
            prefix_name=pref,
        )
        pc1_pos = np.clip(pc1_z, 0, None).rename(f"{pref}_pc1_pos")
        out = pd.concat([out, pc1, pc1_z, pc1_pos], axis=1)

    return out


# ============================================================
# 6. Gate interactions
# ============================================================

def make_gate(s: pd.Series, alpha=1.0, clip_hi=3.0) -> pd.Series:
    pos = np.clip(s.fillna(0), 0, clip_hi)
    return 1.0 + alpha * pos


def pick_z_for_gate(df: pd.DataFrame, plan: Dict[str, Tuple[int, ...]]) -> List[str]:
    """
    For each (group, window) in plan, pick raw-derived columns '<group><n>..._z<window>'.
    PCA outputs (V_pc1_z126, ...) have no group and are never picked.
    Each column is returned once.
    """
    picked: List[str] = []
    seen = set()
    for pref, wins in plan.items():
        for w in wins:
            suffix = f"_z{w}"
            for c in df.columns:
                if c not in seen and group_of(c) == pref and c.endswith(suffix):
                    picked.append(c)
                    seen.add(c)
    return picked


def apply_gate(
    df: pd.DataFrame,
    base_cols: List[str],
    gate: pd.Series,
    name: str,
    add_missing_term: bool = False,
    use_availability_mask: bool = False,
    avail_min_obs: int = 21,
    avail_window: int = 63,
) -> pd.DataFrame:
    """
    Gated interactions:
      {col}_g{name}         = col.fillna(0) * gate
      {col}_g{name}_miss    = 1(col is NaN) * gate            (optional)
      {col}_g{name}_unavail = 1(not enough recent history)    (optional)
    """
    g = gate.fillna(1.0)  # neutral gate when NaN
    new = {}

    for c in base_cols:
        if c not in df.columns:
            continue
        gcol = f"{c}_g{name}"
        main = df[c].fillna(0.0) * g

        if use_availability_mask:
            avail_cnt = (~df[c].isna()).rolling(avail_window, min_periods=1).sum()
            mask = (avail_cnt >= avail_min_obs).astype("int8")
            main = main * mask
            new[f"{gcol}_unavail"] = (1 - mask).astype("int8")

        new[gcol] = main

        if add_missing_term:
            new[f"{gcol}_miss"] = df[c].isna().astype("int8") * g

    if not new:
        return df
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# ============================================================
# 7. Master pipeline
# ============================================================

GATE_PLAN = {
    "V":   (21, 63),
    "S":   (21, 63),
    "M":   (21, 63),
    "MOM": (21, 63),
    "E":   (126, 252),
    "P":   (126, 252),
    "I":   (63, 126),
}

# Macro (E) and sentiment (S) news usually affects prices with a delay of
# roughly 10 trading days, so both blocks are lagged by ES_LAG rows.
ES_LAG = 10

LABELS = ("forward_returns", "risk_free_rate", "market_forward_excess_returns")


def build_features(
    df_raw: pd.DataFrame,
    n_raw: int = 5000,
    drop_all_nan: bool = False,
) -> pd.DataFrame:
    """
    Master feature pipeline for Hull Tactical train.csv.

    df_raw       : full raw train.csv
    n_raw        : keep only the last n_raw rows before feature engineering
    drop_all_nan : drop rows that still contain any NaN at the end
                   (default False, tree models handle NaN)
    """
    # 0) keep last n_raw rows
    data = df_raw.iloc[-n_raw:].reset_index(drop=True)

    # 1) delayed-impact assumption for E/S
    es_cols = [c for c in data.columns if group_of(c) in ("E", "S")]
    if es_cols and ES_LAG > 0:
        data[es_cols] = data[es_cols].shift(ES_LAG)
        data = data.iloc[ES_LAG:].reset_index(drop=True)

    # 2) lagged labels, same schema as test.csv
    for c in LABELS:
        if c in data.columns:
            data[f"lagged_{c}"] = data[c].shift(1)

    # 3) rolling block on lagged labels / lagged excess
    data = build_lagged_block(data)

    # 4) excess return used by the metric (forward_returns minus risk_free_rate)
    if {"forward_returns", "risk_free_rate"}.issubset(data.columns):
        data["excess_returns"] = data["forward_returns"] - data["risk_free_rate"]

    # 5) unified burn by the longest lagged window (63)
    BURN = 63 - 1
    data = data.iloc[BURN:].reset_index(drop=True)

    # 6) drop late-starting / all-NaN raw features
    _, _, data = first_valid_by_prefix(data)

    # 7) hand-crafted interactions
    data = add_simple_interactions(data, INTERACTION_FEATURES, add_missing_flags=False)

    # 8) rolling mean / z / pos / d1 on raw features (D: rolling mean only)
    data = add_roll_features(data)

    # 9) drop *_mean* columns of selected groups
    cols_to_drop = [
        c for c in data.columns
        if group_of(c) in DROP_MEAN_GROUPS and "_mean" in c
    ]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)

    # 10) group-level causal PCA
    data = add_group_pcas(data)

    # 11) gate interactions
    # 11.1) large E factor amplifies V z-features
    if "E_pc1_z252" in data.columns:
        gate_E = make_gate(data["E_pc1_z252"], alpha=0.5, clip_hi=3.0)
        Vz_rep = pick_z_for_gate(data, {"V": GATE_PLAN["V"]})
        data = apply_gate(data, base_cols=Vz_rep, gate=gate_E, name="E")

    # 11.2) large V factor amplifies non-V z-features
    if "V_pc1_z126" in data.columns:
        gate_V = make_gate(data["V_pc1_z126"], alpha=0.5, clip_hi=3.0)
        nonV_plan = {k: v for k, v in GATE_PLAN.items() if k != "V"}
        core_z = pick_z_for_gate(data, nonV_plan)
        data = apply_gate(data, base_cols=core_z, gate=gate_V, name="V")

    # 11.3) strong volatility gives extra amplification to S z-features
    if "V_pc1_z126" in data.columns:
        gate_Vstrong = make_gate(data["V_pc1_z126"], alpha=0.8, clip_hi=4.0)
        Sz_rep = pick_z_for_gate(data, {"S": GATE_PLAN["S"]})
        data = apply_gate(data, base_cols=Sz_rep, gate=gate_Vstrong, name="Vstrong")

    # 12) optional: drop rows with any remaining NaN
    if drop_all_nan:
        data = data.dropna(axis=0).reset_index(drop=True)

    return data

# ============================================================
# Entry point
# ============================================================
#
# Reads   <project>/train.csv
# Writes  <project>/kaggle_hull/all_feature_last_<N_RAW>.csv
#
# N_RAW is the number of raw rows kept before feature engineering. It lives in
# paths.py, not here, so the output file name always follows the window length
# and a run at one length cannot silently reuse a file built at another.
#
#     python Hull_Tactical_feature_engineering.py

def main() -> None:
    import os
    from paths import TRAIN_CSV as IN_PATH, FEATURE_PATH as OUT_PATH, N_RAW

    print("Loading raw data...")
    df_raw = pd.read_csv(IN_PATH)

    print(f"Building features from the last {N_RAW} rows...")
    df_feat = build_features(df_raw, n_raw=N_RAW)

    out_dir = os.path.dirname(OUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Saving to {OUT_PATH} ...")
    df_feat.to_csv(OUT_PATH, index=False)
    print(f"Done. Final shape: {df_feat.shape}")


if __name__ == "__main__":
    main()
