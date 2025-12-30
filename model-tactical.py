import numpy as np
import pandas as pd

IN_DIR = r"E:\Hull Tactical"
all_feature = pd.read_csv(r"E:\Hull Tactical\all_feature_last_5000_no_nan.csv")

y = all_feature["forward_returns"]

LABEL_COLS = [
    "forward_returns",
    "risk_free_rate",
    "market_forward_excess_returns",
    "excess_returns",
]

def is_label(col: str) -> bool:
    return col in LABEL_COLS

def is_meta(col: str) -> bool:
    # Add other ID/meta columns here if needed
    return col in ("date_id", "is_scored")


def get_main_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the list of "main feature" columns:
    - Keep numeric columns only
    - Exclude label / meta / all kinds of nan-indicator columns
    - Keep lagged_* columns (as requested)
    """
    num_cols = df.select_dtypes(include=[np.number]).columns

    main_feats = []
    for c in num_cols:
        if is_label(c):
            continue
        if is_meta(c):
            continue
        # Do NOT filter out lagged_*; keep them as main feature candidates
        main_feats.append(c)

    return main_feats


main_feature_cols = get_main_feature_cols(all_feature)
print("Number of main features:", len(main_feature_cols))
# Quick check: print first few columns
print(main_feature_cols[:30])



import pandas as pd
from lightgbm import LGBMRegressor

# --- 1) Split: last 180 rows as holdout (public-leaderboard equivalent) ---
HOLDOUT = 180
core = all_feature.iloc[:-HOLDOUT].reset_index(drop=True)
hold = all_feature.iloc[-HOLDOUT:].reset_index(drop=True)

X_core = core[main_feature_cols]
y_core = y.iloc[:-HOLDOUT].reset_index(drop=True)

X_hold = hold[main_feature_cols]
y_hold = y.iloc[-HOLDOUT:].reset_index(drop=True)



def coarse_screen(
    X_core: pd.DataFrame,
    nan_thr=0.98,
    min_std=1e-12,
    corr_thr=0.98,
    corr_sample=None,          # None = use all rows
    corr_method="spearman",
):
    cols = X_core.columns.tolist()

    # 2.1 Missing-rate filter
    na_ratio = X_core.isna().mean()
    keep = [c for c in cols if na_ratio[c] <= nan_thr]

    # Choose the subset of rows used for correlation/std checks
    if (corr_sample is None) or (len(X_core) <= corr_sample):
        X_tail = X_core
    else:
        X_tail = X_core.iloc[-corr_sample:]

    # 2.2 Near-constant columns (decide using X_tail)
    stds = X_tail[keep].std(numeric_only=True)
    keep = [c for c in keep if stds.get(c, 0.0) > min_std]

    # 2.3 High-correlation de-duplication: for |rho| > thr, keep only one
    Xt = X_tail[keep]
    corr = Xt.corr(method=corr_method).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > corr_thr)]
    kept = [c for c in keep if c not in set(drop)]

    info = {
        "n_total": len(cols),
        "drop_na": int((na_ratio > nan_thr).sum()),
        "drop_const": int((stds <= min_std).sum()),
        "drop_corr": len(drop),
        "n_kept": len(kept),
        "rows_used_for_corr": len(X_tail),
    }
    return kept, info


kept_cols, info = coarse_screen(
    X_core,
    nan_thr=0.98,
    min_std=1e-12,
    corr_thr=0.98,
    corr_sample=None,          # use full sample
    corr_method="spearman",
)

print(info)

# --- 3) Build matrices for later OOF using the coarse-screened columns ---
X_core_cs = X_core[kept_cols]
X_hold_cs = X_hold[kept_cols]
print("coarse-screen shapes:", X_core_cs.shape, X_hold_cs.shape)



import re
from typing import List, Tuple, Dict, Set

X_core_oof = core[kept_cols].copy()



from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

def oof_with_expanding_window(X_oof, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)  # expanding-style splits by default

    fi_sum = pd.Series(0.0, index=X_oof.columns)

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_oof)):
        X_tr, X_va = X_oof.iloc[tr_idx], X_oof.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = LGBMRegressor(
            learning_rate=0.03,
            n_estimators=3000,
            num_leaves=126,
            max_depth=20,
            min_data_in_leaf=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=5.0,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
        )

        fi_fold = pd.Series(model.feature_importances_, index=X_oof.columns)
        fi_sum += fi_fold

    fi_mean = fi_sum / n_splits
    return fi_mean


fi_all = oof_with_expanding_window(X_core_oof, y_core)



main_in_fi = [c for c in kept_cols if c in fi_all.index]
fi_main = fi_all.loc[main_in_fi].copy()

# Sort by importance descending
fi_main = fi_main.sort_values(ascending=False)

for thr in [300, 250, 200, 100, 80, 50]:
    cnt = (fi_main >= thr).sum()
    print(f"importance ≥ {thr}: {cnt}")



total_imp = fi_main.sum()
cum_ratio = fi_main.cumsum() / total_imp

# Find the first position where cumulative importance >= 0.95
cutoff_label = (cum_ratio >= 0.95).idxmax()

# Select feature list up to that cutoff
lgbm_features = fi_main.loc[:cutoff_label].index.tolist()

print(
    f"Total features: {len(fi_main)}, "
    f"selected for 85% cum importance: {len(lgbm_features)}"
)

selected_main = fi_main[lgbm_features].index.tolist()

print("Number of selected main features:", len(selected_main))



X_core_last = core[selected_main]
X_hold_last = hold[selected_main]
y_core_last = y_core
y_hold_last = y_hold
print("All selected features (main + nan_indicator):", len(selected_main))


# ===== For linear regression =====
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


linear_main = fi_main[fi_main >= 260.0].index.tolist()
print("Linear model main features:", len(linear_main))

# Linear: use main columns only (no companion / nan-indicator)
X_core_lin = core[linear_main].copy()
X_hold_lin = hold[linear_main].copy()

from sklearn.preprocessing import FunctionTransformer

def forward_fill_array(X):
    # X may be an ndarray; convert to DataFrame and do ffill/bfill along time-ordered rows
    X_df = pd.DataFrame(X)
    X_ff = X_df.ffill().bfill()
    return X_ff.values

lin_imputer = FunctionTransformer(forward_fill_array)

X_core_lin_imp = lin_imputer.fit_transform(X_core_lin)
X_hold_lin_imp = lin_imputer.transform(X_hold_lin)

# Standardize after imputation
scaler_lin = StandardScaler()
X_core_lin_scaled = scaler_lin.fit_transform(X_core_lin_imp)
X_hold_lin_scaled = scaler_lin.transform(X_hold_lin_imp)

# Initialize and fit linear model offline
lin_model = SGDRegressor(
    loss="squared_error",
    penalty="l2",
    alpha=1e-4,
    max_iter=1000,
    learning_rate="invscaling",
    eta0=0.01,
    random_state=42,
)
linear_params = {
    "loss": "squared_error",
    "penalty": "l2",
    "alpha": 1e-4,
    "max_iter": 1000,
    "learning_rate": "invscaling",
    "eta0": 0.01,
    "random_state": 42,
}
lin_model.fit(X_core_lin_scaled, y_core_last.values)


# ---- Linear model predictions (core / hold) ----
yhat_core_lin = lin_model.predict(X_core_lin_scaled)   # shape = (len(core),)
yhat_hold_lin = lin_model.predict(X_hold_lin_scaled)   # shape = (len(hold),)

# Convert to Series to align indexes
yhat_core_lin_s = pd.Series(yhat_core_lin, index=y_core_last.index)
yhat_hold_lin_s = pd.Series(yhat_hold_lin, index=y_hold_last.index)

# ---- residual = truth - linear prediction ----
res_core_last = y_core_last - yhat_core_lin_s
res_hold_last = y_hold_last - yhat_hold_lin_s



import numpy as np
import pandas as pd

MIN_INVESTMENT, MAX_INVESTMENT = 0.0, 2.0

def position_rule(raw_pred, X_all_lgbm, t, pos_params):
    K       = pos_params.get("K", 6.0)
    max_lev = pos_params.get("max_leverage", 2.0)
    min_lev = pos_params.get("min_leverage", 0.0)
    vol_col = pos_params.get("vol_col", "lagged_forward_returns_std21")
    mom_col = pos_params.get("mom_col", "lagged_forward_returns_mean21")
    vol_floor = pos_params.get("vol_floor", 1e-4)
    crash_mom_th = pos_params.get("crash_mom_threshold", -0.0005)

    sig = raw_pred

    # === Use 21-day std as volatility ===
    vol_t = X_all_lgbm.iloc[t][vol_col]
    if not np.isfinite(vol_t) or vol_t < vol_floor:
        vol_t = vol_floor

    sig = sig / vol_t   # turn into "predicted Sharpe"

    # === tanh to position ===
    pos = 1.0 + np.tanh(K * sig)

    # === Use 21-day mean for crash check ===
    mom_t = X_all_lgbm.iloc[t][mom_col]
    if np.isfinite(mom_t) and (mom_t < crash_mom_th) and (pos > 1.0):
        pos = 1.0

    pos = float(np.clip(pos, min_lev, max_lev))
    return pos



def _to_position(yhat, mode: str = "excess", K: float = 6.0):
    """
    yhat:
      - mode='excess'   -> predicted excess return (pred_excess)
      - mode='prob'     -> up-move probability p in [0,1]
      - mode='position' -> yhat is already a position in [0,2]
    """
    yhat = np.asarray(yhat, dtype=float)

    if mode == "position":
        # Already a position; just clip to [0,2]
        return np.clip(yhat, MIN_INVESTMENT, MAX_INVESTMENT)

    if mode == "excess":
        # Your previous rule: 1 + tanh(K * yhat_excess)
        pos = 1.0 + np.tanh(K * yhat)
        return np.clip(pos, MIN_INVESTMENT, MAX_INVESTMENT)

    if mode == "prob":
        # Probability -> position: simple linear map [0,1] -> [0,2]
        pos = 2.0 * yhat
        return np.clip(pos, MIN_INVESTMENT, MAX_INVESTMENT)

    raise ValueError(f"Unknown mode={mode}")



def adjusted_sharpe_kaggle_like(
    df: pd.DataFrame,
    yhat,
    mode: str = "forward return",
    K: float = 2.0,
    trading_days_per_yr: int = 252,
    return_components: bool = False,
):
    """
    df must contain: 'forward_returns', 'risk_free_rate'
    yhat meaning depends on mode:
      - mode='forward return'
      - mode='prob'     : yhat is up-move probability p
      - mode='position' : yhat is already a position in [0,2]
    """
    # 1) Map to position (or directly clip if already a position)
    position = _to_position(yhat, mode=mode, K=K)

    # 2) Strategy return (consistent with Kaggle evaluation)
    fr = df["forward_returns"].astype(float).values
    rf = df["risk_free_rate"].astype(float).values

    strat_ret = rf * (1.0 - position) + position * fr

    # 3) Sharpe (based on excess return)
    strat_excess = strat_ret - rf

    # Annualized mean (geometric compounding converted to average)
    strat_excess_cum = np.prod(1.0 + strat_excess)
    strat_mean_excess = strat_excess_cum ** (1.0 / len(strat_excess)) - 1.0

    strat_std = strat_ret.std()
    if strat_std == 0:
        return (0.0, {}) if return_components else 0.0

    sharpe = strat_mean_excess / strat_std * np.sqrt(trading_days_per_yr)
    strat_vol_annual = float(strat_std * np.sqrt(trading_days_per_yr) * 100.0)

    # 4) Market benchmark annualized vol and mean excess
    mkt_excess = fr - rf
    mkt_excess_cum = np.prod(1.0 + mkt_excess)
    mkt_mean_excess = mkt_excess_cum ** (1.0 / len(mkt_excess)) - 1.0
    mkt_std = fr.std()
    if mkt_std == 0:
        return (0.0, {}) if return_components else 0.0
    mkt_vol_annual = float(mkt_std * np.sqrt(trading_days_per_yr) * 100.0)

    # 5) Penalties (consistent with Kaggle evaluation)
    # Volatility penalty: penalize only when strategy vol > 1.2 * market vol
    excess_vol = max(0.0, strat_vol_annual / mkt_vol_annual - 1.2) if mkt_vol_annual > 0 else 0.0
    vol_penalty = 1.0 + excess_vol

    # Return shortfall penalty: (market mean excess - strategy mean excess), only penalize if positive
    return_gap = max(0.0, (mkt_mean_excess - strat_mean_excess) * 100.0 * trading_days_per_yr)
    return_penalty = 1.0 + (return_gap ** 2) / 100.0

    adj_sharpe = sharpe / (vol_penalty * return_penalty)
    adj_sharpe = float(min(adj_sharpe, 1_000_000))

    if return_components:
        comps = dict(
            sharpe=float(sharpe),
            strategy_vol_annual=strat_vol_annual,
            market_vol_annual=mkt_vol_annual,
            vol_penalty=float(vol_penalty),
            return_penalty=float(return_penalty),
            strat_mean_excess=float(strat_mean_excess),
            mkt_mean_excess=float(mkt_mean_excess),
        )
        return adj_sharpe, comps

    return adj_sharpe



# ==== 0) imports ====
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMRegressor
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Reuse previously defined:
# - _to_position(...)
# - adjusted_sharpe_kaggle_like(...)

# ==== 1) Prepare label columns needed for validation ====
# When building X_core_last / X_hold_last, label columns were removed from X.
# For adjusted_sharpe we need forward_returns / risk_free_rate, so fetch them
# from core/hold and align by index.
labels_core = core.loc[X_core_last.index, ["forward_returns", "risk_free_rate"]].copy()
labels_hold = hold.loc[X_hold_last.index, ["forward_returns", "risk_free_rate"]].copy()


# ==== 3) Utility: rolling splits ====
def rolling_splits(n, train_len=752, val_size=21, step=21, embargo=0, max_splits=None):
    """
    Rolling-window CV:
      For each fold:
        train: [tr_start, tr_end)
        valid: [start_va, end_va)

      where:
        tr_end   = start_va - embargo
        tr_start = tr_end - train_len

      train_len: fixed training window length (e.g., 1044 days)
      val_size : validation window length per fold
      step     : shift forward each fold (default 21 days)
    """
    start_va = train_len  # first validation window start (ensure enough training history)
    k = 0

    while True:
        end_va = start_va + val_size
        if end_va > n:
            break

        tr_end = max(start_va - embargo, 0)
        tr_start = max(tr_end - train_len, 0)

        train_idx = np.arange(tr_start, tr_end, dtype=int)
        val_idx   = np.arange(start_va, end_va, dtype=int)

        if len(train_idx) == 0 or len(val_idx) == 0:
            break

        yield train_idx, val_idx

        k += 1
        if (max_splits is not None) and (k >= max_splits):
            break

        start_va += step



# =====================
# 2) Stage 1: Optuna tunes only LightGBM hyperparameters (exclude K)
# =====================

def objective_lgbm_only(trial: optuna.Trial):
    """
    Tune LightGBM parameters only. Objective:
      mean_fold( mse_model_fold / mse_zero_fold )
    """
    params = dict(
        objective="regression",
        n_estimators=trial.suggest_int("n_estimators", 100, 400),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 30, 400),
        subsample=trial.suggest_float("subsample", 0.6, 0.8),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.8),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 50.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-2, 20.0, log=True),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    LAST_K = 10

    n = len(X_core_last)
    ratios = []   # per-fold mse_model / mse_zero

    for tr_idx, va_idx in rolling_splits(
        n,
        train_len=752,
        val_size=21,
        step=21,
        embargo=0,
        max_splits=None,
    ):
        Xtr, Xva = X_core_last.iloc[tr_idx], X_core_last.iloc[va_idx]
        ytr, yva = res_core_last.iloc[tr_idx], res_core_last.iloc[va_idx]

        model = LGBMRegressor(**params)
        model.fit(Xtr, ytr)

        yhat_va = model.predict(Xva)
        err = yhat_va - yva.values

        mse_model = float(np.mean(err ** 2))
        mse_zero  = float(np.mean(yva.values ** 2))  # baseline for this fold

        if mse_zero <= 0:
            ratio = 1.0
        else:
            ratio = mse_model / mse_zero

        ratios.append(ratio)
        fold_id = len(ratios)

        if len(ratios) >= LAST_K:
            obj = float(np.mean(ratios[-LAST_K:]))
        else:
            obj = float(np.mean(ratios))

        print(
            f"[Trial {trial.number}] Fold {fold_id} done: "
            f"mse={mse_model:.6f}, mse_zero={mse_zero:.6f}, "
            f"ratio={ratio:.3f}, mean_ratio={np.mean(ratios):.3f}, "
            f"recent_{LAST_K}_ratio={obj:.3f}"
        )

        trial.report(obj, step=fold_id)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return obj  # minimize: average relative MSE


# ---- Run Optuna: tune LGBM only ----
study_lgbm = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_warmup_steps=1),
)
study_lgbm.optimize(objective_lgbm_only, n_trials=60, show_progress_bar=True)

print("[Stage1] Best recent_ratio:", study_lgbm.best_value)
print("[Stage1] Best LGBM params:", study_lgbm.best_params)

best_lgbm_params = study_lgbm.best_params.copy()



# =====================
# 3) Stage 2: On hold (validation), retrain every 21 days; search K on that process
# =====================

import copy
import numpy as np

def walk_forward_sharpe_on_hold(
    X_full_lgbm: pd.DataFrame,   # tree features (X_core_last + X_hold_last)
    X_full_lin: pd.DataFrame,    # linear features (X_core_lin + X_hold_lin)
    y_full: pd.Series,           # forward_returns (core + hold)
    labels_full: pd.DataFrame,   # [['forward_returns','risk_free_rate']] (core + hold)
    start_hold_idx: int,         # hold start index
    end_hold_idx: int,           # hold end index (usually len(X_full_lgbm))
    lgbm_params: dict,
    pos_params: dict,
    linear_model_cls=SGDRegressor,
    linear_params: dict | None = None,
    train_len: int = 504,        # LGBM rolling window length
    step_tree: int = 21,         # LGBM retrain frequency (in hold days)
    K_for_metric: float = 6.0,   # in adjusted_sharpe_kaggle_like (not used under mode='position')
):
    """
    Walk-forward with:
      - Linear model: EXPANDING window, retrain daily.
          * For day t, linear train window = [0, t)
      - LGBM (tree on residuals): ROLLING window, retrain every `step_tree` days.
          * For day t, tree train window = [max(0, t-train_len), t)

    core period: [0, start_hold_idx)
    hold period: [start_hold_idx, end_hold_idx)
    """
    if linear_params is None:
        linear_params = {}

    n_total = len(X_full_lgbm)
    assert n_total == len(X_full_lin) == len(y_full) == len(labels_full)
    n_hold = end_hold_idx - start_hold_idx

    # -------- 1) Linear features: one-time ffill + standardization --------
    X_lin_imp = forward_fill_array(X_full_lin.values)      # (n_total, d_lin)
    X_lin_imp = pd.DataFrame(X_lin_imp, columns=X_full_lin.columns)

    # Fit scaler only on core segment to avoid leakage
    scaler_lin = StandardScaler()
    X_core_lin_imp = X_lin_imp.iloc[:start_hold_idx, :]
    scaler_lin.fit(X_core_lin_imp)

    X_full_lin_scaled = scaler_lin.transform(X_lin_imp)    # (n_total, d_lin)

    # -------- 2) Prepare y/labels/tree features --------
    y_full = y_full.reset_index(drop=True)
    labels_full = labels_full.reset_index(drop=True)

    X_full_lgbm = X_full_lgbm.reset_index(drop=True)
    if X_full_lgbm.isna().any().any():
        X_full_lgbm = X_full_lgbm.ffill().bfill()
        X_full_lgbm = X_full_lgbm.dropna(axis=1, how="all")

    # -------- 3) Buffers for hold predictions --------
    yhat_lin_hold = np.zeros(n_hold, dtype=float)
    res_hat_hold  = np.zeros(n_hold, dtype=float)

    tree = None  # current residual tree

    # -------- 4) Daily walk-forward over hold segment --------
    for idx in range(start_hold_idx, end_hold_idx):
        j = idx - start_hold_idx   # 0 .. n_hold-1

        # ===== 4.1 Linear: expanding window [0, idx), retrain daily =====
        X_lin_tr = X_full_lin_scaled[0:idx, :]
        y_tr_exp = y_full.iloc[0:idx]
        X_lin_va = X_full_lin_scaled[idx:idx+1, :]

        lin_model = linear_model_cls(**linear_params)
        lin_model.fit(X_lin_tr, y_tr_exp)

        # Today's linear prediction
        yhat_va_lin = lin_model.predict(X_lin_va)[0]

        # ===== 4.2 Tree: rolling residual + retrain every step_tree days =====
        if (tree is None) or (j % step_tree == 0):
            # Compute residuals on expanding window (0..idx-1)
            yhat_tr_full = lin_model.predict(X_full_lin_scaled[0:idx, :])
            res_tr_full = y_full.values[0:idx] - yhat_tr_full  # length = idx

            # Tree training window: last `train_len` days
            tree_start = max(0, idx - train_len)
            tree_end   = idx
            tree_idx = np.arange(tree_start, tree_end, dtype=int)

            X_tree_tr = X_full_lgbm.iloc[tree_idx, :]
            res_tr    = res_tr_full[tree_idx]

            tree = LGBMRegressor(**lgbm_params)
            tree.fit(X_tree_tr, res_tr)

        # Predict residual for today using current tree
        X_tree_va = X_full_lgbm.iloc[[idx], :]
        res_va = tree.predict(X_tree_va)[0]

        # Store
        yhat_lin_hold[j] = yhat_va_lin
        res_hat_hold[j]  = res_va

    # -------- 5) Total prediction on hold = linear + residual --------
    yhat_total_hold = yhat_lin_hold + res_hat_hold

    # -------- 6) Map to positions --------
    pos_hold = np.zeros(n_hold, dtype=float)
    for j in range(n_hold):
        t_global = start_hold_idx + j
        pos_hold[j] = position_rule(
            raw_pred=yhat_total_hold[j],
            X_all_lgbm=X_full_lgbm,
            t=t_global,
            pos_params=pos_params,
        )

    # -------- 7) Compute Sharpe only on hold segment --------
    labels_hold = labels_full.iloc[start_hold_idx:end_hold_idx].reset_index(drop=True)

    adj_sharpe, comps = adjusted_sharpe_kaggle_like(
        labels_hold,
        pos_hold,
        mode="position",
        K=K_for_metric,
        return_components=True,
    )

    return adj_sharpe, comps, pos_hold, yhat_total_hold



X_full_lgbm = pd.concat([X_core_last, X_hold_last], axis=0).reset_index(drop=True)
X_full_lin  = pd.concat([X_core_lin,  X_hold_lin],  axis=0).reset_index(drop=True)
y_full      = pd.concat([y_core_last, y_hold_last], axis=0).reset_index(drop=True)
labels_full = pd.concat([labels_core, labels_hold], axis=0).reset_index(drop=True)

start_hold_idx = len(X_core_last)
end_hold_idx   = len(X_full_lgbm)



pos_params = {
    "K": 6.0,                         # tanh scaling factor (position sensitivity)
    "max_leverage": 2.0,              # position upper bound
    "min_leverage": 0.0,              # position lower bound
    "vol_col": "lagged_forward_returns_std21",    # volatility column used in denominator
    "mom_col": "lagged_forward_returns_mean21",   # mean column used for crash check
    "vol_floor": 1e-4,                # floor to avoid division by zero
    "crash_mom_threshold": -0.0005,   # if 21d mean is too bad, disallow long
}


adj_sharpe, comps, pos_hold, yhat_total_hold = walk_forward_sharpe_on_hold(
    X_full_lgbm=X_full_lgbm,
    X_full_lin=X_full_lin,
    y_full=y_full,
    labels_full=labels_full,
    start_hold_idx=start_hold_idx,
    end_hold_idx=end_hold_idx,
    lgbm_params=best_lgbm_params,
    pos_params=pos_params,
    linear_model_cls=SGDRegressor,
    linear_params=linear_params,
    train_len=504,
    step_tree=21,
)

print("Hold Adjusted Sharpe:", adj_sharpe)
print("Components:", comps)



X_all_lgbm = pd.concat([X_core_last, X_hold_last], axis=0).reset_index(drop=True)
n_core = len(X_core_last)
n_hold = len(X_hold_last)

# Linear model predictions on hold: yhat_hold_lin_s (Series)
yhat_hold_lin_arr = yhat_hold_lin_s.values  # convert to np.array, length = n_hold

# Map "linear-only" predictions to positions using the same position_rule
pos_hold_lin = np.zeros(n_hold, dtype=float)
for j in range(n_hold):
    t_global = n_core + j  # global row index in X_all_lgbm
    pos_hold_lin[j] = position_rule(
        raw_pred=yhat_hold_lin_arr[j],
        X_all_lgbm=X_all_lgbm,
        t=t_global,
        pos_params=pos_params,   # same params as hybrid strategy
    )

# Kaggle-like Sharpe evaluation in position mode
adj_S_lin, comps_lin = adjusted_sharpe_kaggle_like(
    labels_hold,
    pos_hold_lin,
    mode="position",
    K=6.0,
    return_components=True,
)

print("Hold Adjusted Sharpe (Linear only):", adj_S_lin)
print("Components (Linear only):", comps_lin)



import os
import json
import joblib
import pandas as pd

# ===============================
# Unified save directory: E:\kaggle_hull\5000
# ===============================
BASE_DIR = r"E:\kaggle_hull"
SAVE_DIR = os.path.join(BASE_DIR, "5000")

os.makedirs(SAVE_DIR, exist_ok=True)
print("Save directory:", SAVE_DIR)

# ===============================
# 1) Save feature lists
# ===============================

# Main features used by LGBM residual model
pd.Series(selected_main, name="feature").to_csv(
    os.path.join(SAVE_DIR, "lgbm_selected_main_features.csv"),
    index=False,
)

# Main features used by linear model
pd.Series(linear_main, name="feature").to_csv(
    os.path.join(SAVE_DIR, "linear_main_features.csv"),
    index=False,
)

print("✓ Saved feature lists.")

# ===============================
# 2) Save best params + Sharpe results
# ===============================
results = {
    "best_lgbm_params": best_lgbm_params,
    "pos_params": pos_params,
    "hold_adj_sharpe_hybrid": float(adj_sharpe),
    "hold_components_hybrid": {k: float(v) for k, v in comps.items()},
    "hold_adj_sharpe_linear": float(adj_S_lin),
    "hold_components_linear": {k: float(v) for k, v in comps_lin.items()},
}

with open(os.path.join(SAVE_DIR, "model_results_and_config.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("✓ Saved model config & metrics.")

# ===============================
# 3) Save linear model + preprocessors
# ===============================
joblib.dump(
    lin_model,
    os.path.join(SAVE_DIR, "linear_sgd_model.pkl"),
)

joblib.dump(
    lin_imputer,
    os.path.join(SAVE_DIR, "linear_imputer.pkl"),
)

joblib.dump(
    scaler_lin,
    os.path.join(SAVE_DIR, "linear_scaler.pkl"),
)

print("✓ Saved linear model, imputer, scaler.")
