# Hull_Tactical_feature_engineering.py
# ------------------------------------
# Feature engineering module for Hull Tactical Kaggle competition
# This file only contains pure data transformation functions
# No file I/O is performed here.

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pandas.api.types import is_numeric_dtype
from typing import Iterable
from typing import Dict, Tuple, List


# ============================================================
# 1. Lagged block construction
# ============================================================

def build_lagged_block(
    df: pd.DataFrame,
    wins=(1, 2, 3, 5, 10, 21, 63),
    excess_wins=(1, 2, 3, 5, 10),
    make_diff=True,
):
    out = df.copy()

    # Construct lagged excess return
    if {'lagged_forward_returns', 'lagged_risk_free_rate'}.issubset(out.columns):
        out['lagged_excess'] = (
            out['lagged_forward_returns'] - out['lagged_risk_free_rate']
        )

    lag_cols = [
        c for c in [
            'lagged_forward_returns',
            'lagged_risk_free_rate',
            'lagged_market_forward_excess_returns'
        ] if c in out.columns
    ]

    def roll_add(col, wlist):
        s = out[col]
        for w in wlist:
            r = s.rolling(w, min_periods=w)
            mu = r.mean()
            sd = r.std()
            out[f'{col}_mean{w}'] = mu
            out[f'{col}_std{w}']  = sd
            out[f'{col}_z{w}']    = (s - mu) / (sd + 1e-9)

        if make_diff:
            out[f'{col}_d1'] = s.diff()

    for col in lag_cols:
        roll_add(col, wins)

    if 'lagged_excess' in out.columns:
        roll_add('lagged_excess', excess_wins)

    return out


# ============================================================
# 2. First-valid filtering by prefix
# ============================================================

PREFIXES = ('E', 'S', 'V', 'M', 'P', 'I', 'MOM', 'D')
EXCLUDE_ALWAYS = {
    'date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns'
}
EXCLUDE_PREFIXES = ('lagged_',)
Q_BY_PREFIX = {p: 0.90 for p in PREFIXES}
MAX_START_ABS = None
MAX_START_FRAC = 0.25


def first_valid_by_prefix(df: pd.DataFrame):
    rows = []

    def is_excluded(col: str) -> bool:
        if col in EXCLUDE_ALWAYS:
            return True
        return any(col.startswith(ep) for ep in EXCLUDE_PREFIXES)

    for p in PREFIXES:
        cols = [c for c in df.columns if c.startswith(p) and not is_excluded(c)]
        for c in cols:
            idx = df[c].first_valid_index()
            row_id = int(idx) if idx is not None else None
            rows.append({'prefix': p, 'column': c, 'first_valid_row': row_id})

    fv = pd.DataFrame(rows)
    if fv.empty:
        return fv, pd.DataFrame(), df.copy()

    cuts = {}
    for p, g in fv.groupby('prefix'):
        vals = g['first_valid_row'].dropna()
        cuts[p] = int(vals.quantile(Q_BY_PREFIX[p])) if len(vals) else None

    fv['cut_q'] = fv['prefix'].map(cuts)

    if MAX_START_ABS is not None:
        hard_cap = MAX_START_ABS
    elif MAX_START_FRAC is not None:
        hard_cap = int(len(df) * float(MAX_START_FRAC))
    else:
        hard_cap = None

    fv['is_late_q'] = fv.apply(
        lambda r: pd.notna(r['first_valid_row']) and
                  pd.notna(r['cut_q']) and
                  r['first_valid_row'] > r['cut_q'],
        axis=1
    )

    fv['is_late_cap'] = fv['first_valid_row'].apply(
        lambda v: hard_cap is not None and pd.notna(v) and v > hard_cap
    )

    fv['is_all_nan'] = fv['first_valid_row'].isna()
    fv['is_late'] = fv['is_late_q'] | fv['is_late_cap'] | fv['is_all_nan']

    drop_cols = fv.loc[fv['is_late'], 'column'].tolist()

    keep_cols = [
        c for c in df.columns
        if (c not in drop_cols)
        or (c in EXCLUDE_ALWAYS)
        or any(c.startswith(ep) for ep in EXCLUDE_PREFIXES)
    ]

    data_clean = df[keep_cols].copy()

    return fv, fv.groupby('prefix').size(), data_clean


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
    interactions: Iterable[tuple[str, str, str]],
    add_missing_flags: bool = False,
    eps: float = 1e-6,
    prefix: str = "XINT_",
) -> pd.DataFrame:
    """
    Add a small set of hand-crafted interaction features:
      - diff:  a - b
      - ratio: a / b (with safe denominator check)

    Parameters
    ----------
    df : pd.DataFrame
        Input feature table.
    interactions : iterable of (a, b, op)
        Column pairs and operation type, e.g. ("I2", "I1", "diff").
    add_missing_flags : bool
        If True, also add *_miss / *_denom_small flags.
    eps : float
        Minimum absolute value for denominator in ratio to be considered safe.
    prefix : str
        Prefix for new interaction feature names.

    Returns
    -------
    pd.DataFrame
        Feature table with interaction columns appended.
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
            denom = out[b]
            num = out[a]

            safe = (~denom.isna()) & (denom.abs() > eps)
            out[name] = np.where(safe, num / denom, np.nan)

            if add_missing_flags:
                out[f"{name}_denom_small"] = (
                    (~denom.isna()) & (denom.abs() <= eps)
                ).astype("int8")
                out[f"{name}_miss"] = (num.isna() | denom.isna()).astype("int8")

    return out


# ============================================================
# 4. Rolling feature construction
# ============================================================

WIN = {
    'V':   (5, 21, 63),
    'S':   (5, 10, 21),
    'M':   (5, 10, 21, 63),
    'MOM': (5, 21),
    'E':   (63, 126, 252),
    'P':   (63, 126, 252),
    'I':   (21, 63),
    'DEFAULT': (21, 63),
}


def prefix_of(col: str):
    for p in ('E','S','V','MOM','M','P','I','D'):
        if col.startswith(p):
            return p
    return 'DEFAULT'


def pick_base_numeric_cols(df: pd.DataFrame):
    exclude_cols = {
        'date_id', 'is_scored',
        'forward_returns', 'risk_free_rate',
        'market_forward_excess_returns', 'excess_returns'
    }

    base = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if c.startswith(('lagged_', 'XINT_')):
            continue
        if c.endswith('_pc1') or '_pc1_' in c:
            continue
        if is_numeric_dtype(df[c]):
            base.append(c)

    return base


def add_roll_features(df: pd.DataFrame):
    out = df.copy()
    base_cols = pick_base_numeric_cols(out)

    for c in base_cols:
        p = prefix_of(c)
        wins = WIN.get(p, WIN['DEFAULT'])
        s = out[c]

        for w in wins:
            r = s.rolling(w, min_periods=w)
            mean = r.mean()
            std = r.std()
            hi = r.max()
            lo = r.min()

            out[f'{c}_mean{w}'] = mean
            out[f'{c}_z{w}']    = (s - mean) / (std + 1e-9)
            out[f'{c}_pos{w}']  = (s - lo) / (hi - lo + 1e-12)

        out[f'{c}_d1'] = s.diff()

    return out


# ============================================================
# 5. Group PCA (causal, winsorized)
# ============================================================

def causal_pc1_onfly_winsor(
    df, cols,
    method='rolling', window=252, warmup=252,
    min_active=5, zscore_window=252, q=0.98,
    prefix_name='G'
):
    n = len(df)
    pc1 = np.full(n, np.nan)
    Xraw = df[cols].values

    for t in range(n):
        if method == 'rolling':
            if t < max(warmup, window):
                continue
            a, b = t - window, t
        else:
            if t < warmup:
                continue
            a, b = 0, t

        active = (~df[cols].iloc[a:b].isna().all(axis=0)).values
        if active.sum() < min_active:
            continue

        X_tr = Xraw[a:b, :][:, active]
        lo = np.nanquantile(X_tr, 1 - q, axis=0)
        hi = np.nanquantile(X_tr, q, axis=0)
        X_tr = np.clip(X_tr, lo, hi)
        X_tr = np.nan_to_num(X_tr)

        x_t = np.clip(Xraw[t:t+1, :][:, active], lo, hi)
        x_t = np.nan_to_num(x_t)

        pca = PCA(n_components=1)
        pca.fit(X_tr)
        pc1[t] = pca.transform(x_t)[0, 0]

    s_pc1 = pd.Series(pc1, index=df.index, name=f'{prefix_name}_pc1')

    mu = s_pc1.rolling(zscore_window, min_periods=1).mean()
    sd = s_pc1.rolling(zscore_window, min_periods=1).std().replace(0, np.nan)
    s_pc1_z = (s_pc1 - mu) / sd
    s_pc1_z.name = f'{prefix_name}_pc1_z{zscore_window}'

    return s_pc1, s_pc1_z


def add_group_pcas(df: pd.DataFrame):
    out = df.copy()

    plan = {
        'V':   dict(method='rolling',  window=126, warmup=126, zwin=126),
        'M':   dict(method='rolling',  window=126, warmup=126, zwin=126),
        'MOM': dict(method='rolling',  window=63,  warmup=63,  zwin=126),
        'E':   dict(method='rolling',  window=252, warmup=252, zwin=252),
        'S':   dict(method='rolling',  window=126, warmup=126, zwin=252),
        'I':   dict(method='expanding',              warmup=504, zwin=252),
        'P':   dict(method='expanding',              warmup=504, zwin=252),
    }

    for pref, cfg in plan.items():
        cols = [c for c in out.columns if c.startswith(pref) and c.endswith('_z21')]
        if not cols:
            continue

        pc1, pc1_z = causal_pc1_onfly_winsor(
            out, cols=cols,
            method=cfg['method'],
            window=cfg.get('window', 252),
            warmup=cfg['warmup'],
            zscore_window=cfg['zwin'],
            prefix_name=pref
        )

        out = pd.concat([out, pc1, pc1_z], axis=1)
        out[f'{pref}_pc1_pos'] = np.clip(out[pc1_z.name], 0, None)

    return out


# ============================================================
# 6. Gate interactions
# ============================================================

def make_gate(s: pd.Series, alpha=1.0, clip_hi=3.0):
    pos = np.clip(s.fillna(0), 0, clip_hi)
    return 1.0 + alpha * pos

def pick_z_for_gate(
    df: pd.DataFrame,
    plan: Dict[str, Tuple[int, ...]],
) -> List[str]:
    """
    Pick representative z-score features for each prefix and window.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table after rolling / PCA steps.
    plan : dict
        Mapping from prefix (e.g. 'V', 'S', 'MOM') to a tuple of windows,
        e.g. {'V': (21, 63)}. For each (prefix, window), this picks columns
        of the form '{prefix}*..._z{window}'.

    Returns
    -------
    List[str]
        List of column names in df that match (prefix, window) pairs.
    """
    picked: List[str] = []
    for pref, wins in plan.items():
        for w in wins:
            suffix = f"_z{w}"
            picked.extend(
                [
                    c
                    for c in df.columns
                    if c.startswith(pref) and c.endswith(suffix)
                ]
            )
    return picked


def apply_gate(
    df: pd.DataFrame,
    base_cols: list[str],
    gate: pd.Series,
    name: str,
    add_missing_term: bool = False,
    use_availability_mask: bool = False,
    avail_min_obs: int = 21,
    avail_window: int = 63,
) -> pd.DataFrame:
    """
    Create gated interaction features:

      - Main term: {col}_g{name} = col(fillna(0)) * gate
      - Optional missing term: {col}_g{name}_miss = 1(col isna) * gate
      - Optional availability mask: {col}_g{name}_unavail = 1(history not enough)

    Parameters
    ----------
    df : pd.DataFrame
        Input feature table.
    base_cols : list[str]
        Columns to be multiplied by the gate.
    gate : pd.Series
        1D gate series, will be broadcast to all base_cols.
    name : str
        Gate name to be appended in the suffix, e.g. "E", "V", "Vstrong".
    add_missing_term : bool
        Whether to add *_miss term (missing indicator * gate).
    use_availability_mask : bool
        Whether to zero out rows with insufficient history.
    avail_min_obs : int
        Minimum number of non-missing observations in the rolling window.
    avail_window : int
        Window length to compute availability.

    Returns
    -------
    pd.DataFrame
        Table with gated interaction columns appended.
    """
    out = df.copy()
    g = gate.fillna(1.0)  # neutral gate when NaN

    for c in base_cols:
        if c not in out.columns:
            continue

        gcol = f"{c}_g{name}"

        # main interaction: treat NaN as 0, do not propagate NaN
        main = out[c].fillna(0.0) * g

        if use_availability_mask:
            # use only past information to compute availability
            avail_cnt = (~out[c].isna()).rolling(avail_window, min_periods=1).sum()
            mask = (avail_cnt >= avail_min_obs).astype("int8")
            main = main * mask
            out[f"{gcol}_unavail"] = (1 - mask).astype("int8")

        out[gcol] = main

        if add_missing_term:
            out[f"{gcol}_miss"] = out[c].isna().astype("int8") * g

    return out


# ============================================================
# 7. Master pipeline
# ============================================================
GATE_PLAN = {
    'V':   (21, 63),
    'S':   (21, 63),
    'M':   (21, 63),
    'MOM': (21, 63),
    'E':   (126, 252),
    'P':   (126, 252),
    'I':   (63, 126),
}

def build_features(
    df_raw: pd.DataFrame,
    n_raw: int = 5000,
    drop_all_nan: bool = False,
) -> pd.DataFrame:
    """
    Master feature engineering pipeline for Hull Tactical train.csv.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Full raw train.csv loaded in memory.
    n_raw : int
        Use only the last n_raw rows before feature engineering.
    drop_all_nan : bool
        If True, drop rows that still contain any NaN at the very end
        (default False – safer to keep rows and let models handle NaNs).

    Returns
    -------
    pd.DataFrame
        Feature matrix (no file I/O here).
    """
    # ---- 0) Keep last N_RAW rows ----
    data = df_raw.iloc[-n_raw:].reset_index(drop=True)

    # ---- 1) Shift E/S block to align with test ----
    es_cols = [c for c in data.columns if c.startswith(("E", "S"))]
    if es_cols:
        data[es_cols] = data[es_cols].shift(10)
        data = data.iloc[10:].reset_index(drop=True)

    # ---- 2) Build lagged_* labels (to match test schema) ----
    LABELS = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
    for c in LABELS:
        if c in data.columns:
            data[f"lagged_{c}"] = data[c].shift(1)

    # ---- 3) Lagged block for labels / excess ----
    data = build_lagged_block(
        data,
        wins=(1, 2, 3, 5, 10, 21, 63),
        excess_wins=(1, 2, 3, 5, 10),
        make_diff=True,
    )

    # ---- 4) Excess return target (used in training) ----
    if {"forward_returns", "risk_free_rate"}.issubset(data.columns):
        data["excess_returns"] = data["forward_returns"] - data["risk_free_rate"]

    # ---- 5) Unified burn by max window (63) ----
    BURN = 63 - 1
    data = data.iloc[BURN:].reset_index(drop=True)

    # ---- 6) First-valid filtering by prefix (drop “late starting / all-NaN” factors) ----
    fv_table, fv_summary, data_clean = first_valid_by_prefix(data)
    # 这里只是看 shape，你原始版本也有打印；如果想保留 print 可以在外面做

    # ---- 7) Simple deterministic interactions (I/P/V/S wiring) ----
    data_clean = add_simple_interactions(
    data_clean,
    INTERACTION_FEATURES,
    add_missing_flags=False,
)

    # ---- 8) Rolling stats / z / pos / d1 on base numeric cols ----
    data_feat = add_roll_features(data_clean)

    # ---- 9) Optionally drop some *_mean* groups (P / I / S) if desired ----
    drop_mean_groups = ("P", "I", "S")
    cols_to_drop = []
    for c in data_feat.columns:
        prefix = c.split("_", 1)[0]
        if prefix in drop_mean_groups and "_mean" in c:
            cols_to_drop.append(c)
    if cols_to_drop:
        data_feat = data_feat.drop(columns=cols_to_drop)

    # ---- 10) Group-level PCAs (V/M/MOM/E/S/I/P), causal + winsor ----
    data_feat = add_group_pcas(data_feat)

    # ---- 11) Gate interactions (this is你指出我漏掉的部分) ----
    data_after_gate = data_feat.copy()

    # 11.1) E large → amplify V z features
    if "E_pc1_z252" in data_after_gate.columns:
        gate_E = make_gate(
            data_after_gate["E_pc1_z252"],
            alpha=0.5,
            clip_hi=3.0,
        ).fillna(1.0)
        Vz_rep = [
            c
            for c in data_after_gate.columns
            if c.startswith("V")
            and c in pick_z_for_gate(data_after_gate, {"V": GATE_PLAN["V"]})
        ]
        data_after_gate = apply_gate(
            data_after_gate,
            base_cols=Vz_rep,
            gate=gate_E,
            name="E",
            add_missing_term=False,
            use_availability_mask=False,
            avail_min_obs=21,
            avail_window=63,
        )

    # 11.2) V large → amplify non-V (S/MOM/M/E/P/I) z features
    if "V_pc1_z126" in data_after_gate.columns:
        gate_V = make_gate(
            data_after_gate["V_pc1_z126"],
            alpha=0.5,
            clip_hi=3.0,
        ).fillna(1.0)
        nonV_plan = {k: v for k, v in GATE_PLAN.items() if k != "V"}
        core_z = pick_z_for_gate(data_after_gate, nonV_plan)
        data_after_gate = apply_gate(
            data_after_gate,
            base_cols=core_z,
            gate=gate_V,
            name="V",
            add_missing_term=False,
            use_availability_mask=False,
            avail_min_obs=21,
            avail_window=63,
        )

    # 11.3) Strong volatility → extra amplification on S z features
    if "V_pc1_z126" in data_after_gate.columns:
        gate_Vstrong = make_gate(
            data_after_gate["V_pc1_z126"],
            alpha=0.8,
            clip_hi=4.0,
        ).fillna(1.0)
        Sz_rep = [
            c
            for c in data_after_gate.columns
            if c.startswith("S")
            and c in pick_z_for_gate(data_after_gate, {"S": GATE_PLAN["S"]})
        ]
        data_after_gate = apply_gate(
            data_after_gate,
            base_cols=Sz_rep,
            gate=gate_Vstrong,
            name="Vstrong",
            add_missing_term=False,
            use_availability_mask=False,
            avail_min_obs=21,
            avail_window=63,
        )

    # ---- 12) Optional: drop rows with any remaining NaNs ----
    if drop_all_nan:
        data_after_gate = data_after_gate.dropna(axis=0)

    return data_after_gate
