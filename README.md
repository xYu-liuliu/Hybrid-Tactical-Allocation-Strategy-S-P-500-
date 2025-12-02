# Hybrid Tactical Allocation Strategy on S&P 500

This project implements an end-to-end **signal → position → strategy return** pipeline for tactical asset allocation on the S&P 500, built upon the Kaggle **Hull Tactical Market Prediction** dataset. The central technical contribution of the system is a **causal, regime-aware feature engineering framework** designed for stable time-series modeling.

A **linear base model** provides the main return signal, while a **LightGBM model is trained only on the residual errors** to capture nonlinear deviations and regime-dependent corrections. The combined signal is transformed into **dynamic trading positions** with explicit risk control.

Strategy performance is evaluated using an **out-of-sample Sharpe ratio** under a **strict walk-forward evaluation scheme** with no look-ahead bias. The project aims to showcase an **industrial-grade quantitative modeling pipeline**, emphasizing feature stability, model consistency, and realistic trading evaluation.

## 1. Dataset
This project is based on the **Hull Tactical Market Prediction** dataset from Kaggle:  
https://www.kaggle.com/competitions/hull-tactical-market-prediction/data  

The dataset includes daily S&P 500 **forward returns**, **risk-free rates**, and a rich set of predictors grouped by prefixes:

- **E**: economic / macro indicators  
- **S**: sentiment signals  
- **V**: volatility measures  
- **M**: market / price-based indicators  
- **MOM**: momentum factors  
- **P**: positioning-related variables  
- **I**: institutional / information-type features  
- **D**: date / calendar features  

Following the competition setup, the **last 180 trading days** are used as a fixed **out-of-sample hold-out set**, while all earlier data are used for training and walk-forward validation.

To ensure strict information-set consistency, all return-based targets are converted into **lagged variables**, so that each trading signal is generated using only information available up to \(t-1\), fully eliminating look-ahead bias.

## 2. Feature Engineering (Core Contribution)

The main technical contribution of this project lies in the **feature engineering pipeline**, rather than in the choice of predictive models.

Key principles:
- All features are constructed **causally**, using only past information.
- All return-related variables are converted into **lagged form to align with the public leaderboard information set**.
- Different economic groups (E, S, V, M, MOM, P, I) are assigned **group-specific rolling windows**.
- Both **short-term dynamics and long-term regimes** are explicitly modeled.

### (1) Lagged Returns and Excess Returns (Leaderboard Alignment)
All return-based variables are shifted by one period to ensure strict information-set consistency:
- Lag-1 forward returns  
- Lag-1 risk-free rate  
- Lag-1 market excess return  
- Lagged excess return  

These lagged variables also serve as inputs for subsequent rolling-window construction.

### (2) Multi-Scale Rolling Statistics
For each base feature and each lagged return variable, the following transformations are applied:
- Rolling mean
- Rolling z-score
- Rolling min–max position
- First difference

Rolling windows are **prefix-dependent and model-feedback refined**:
- **Lagged returns**: ultra–short horizons **(2, 3, 5, 10, 21)**
- **V (Volatility)**: short–medium windows **(5, 21, 63)**
- **S (Sector / Sentiment)**: short-term windows **(5, 10, 21)**
- **M (Market)**: short–medium windows **(5, 10, 21)**
- **MOM (Momentum)**: medium-term windows **(5, 21)**
- **E (Economics)**: long-horizon windows **(63, 126, 252)**
- **P (Price level / Valuation)**: long-horizon windows **(63, 252)**
- **I (Interest / Macro rates)**: medium-horizon windows **(21, 63)**

Window configurations were finalized after **LightGBM importance diagnostics**, with unstable short-horizon or redundant long-horizon windows pruned accordingly.

### (3) Causal Group-wise PCA with Winsorization
- Group-wise PCA is computed on rolling or expanding windows.
- Inputs are winsorized using **in-window quantiles** to control outliers.
- PCA directions are aligned via correlation with the in-sample reference.

### (4) Regime-Gated Interactions
- Macro and volatility PC factors act as **regime gates**.
- Selected rolling z-score features are multiplicatively amplified under strong regimes.
- All gate values are bounded for numerical and risk stability.

### (5) LightGBM-Driven Feature Selection and Window Refinement
Feature pruning and window redesign are guided by **LightGBM importance analysis** rather than heuristic filters:
- Low-importance and noisy features are removed,
- Short-horizon windows with weak contribution are pruned,
- Redundant temporal scales are compressed.

This process turns feature engineering into a **model-feedback-driven optimization loop**, improving both signal quality and stability.


## 3. Modeling Strategy 
### 3.1 Final Hybrid Model

The final production model follows a **hybrid linear–nonlinear return forecasting framework** under rolling retraining:

- A **linear model** is used to **directly predict forward returns** and is **retrained daily**.
- A **LightGBM model** is trained to predict the **residual errors of the linear forward-return forecasts** and is **retrained every 21 trading days** using a rolling window.
- The final **forward return forecast** is formed as:
  
  > forward_return_hat = linear_prediction + residual_correction

- This predicted forward return is then transformed into a **tradable position** through a deterministic **position sizing rule**.

This design reflects a clear **division of labor**:
- The linear model captures the dominant, fast-moving return dynamics.
- LightGBM provides a slower-moving nonlinear correction without destabilizing the core return signal.

Both models are trained and evaluated under a **strict walk-forward rolling-window protocol**, with feature windows and the nonlinear component updated every 21 trading days.

### 3.2 Model Refinement Path 
Before finalizing the feature configuration, multiple modeling paths were tested to **stress-test the stability and usefulness of the feature set**:

1. **Direct Return Prediction (Linear vs. LightGBM)**  
   Both Linear Regression and LightGBM were first applied to directly predict forward returns.  
   Result: the linear model consistently outperformed LightGBM in both stability and out-of-sample trading performance.

2. **Regime Filtering via LightGBM**  
   LightGBM was then used to classify market regimes, conditioning whether the linear signal should be traded.  
   Result: regime predictions were unstable and significantly degraded downstream Sharpe performance.

3. **Residual Modeling with Fixed Features**  
   LightGBM was next applied to predict residuals from the linear model under the original feature configuration.  
   Result: residual predictability was weak, with signal-to-noise ratio close to zero.

4. **Feature Window Redesign Driven by Model Feedback**  
   Based on LightGBM feature importance diagnostics:
   - Volatility state features with persistently low explanatory power were removed,
   - Sentiment features (S) were shifted to **short-term windows**,
   - Lagged return windows were compressed to **ultra-short horizons (2, 3, 5, 10, 21)**.

   After this structural redesign, residual modeling began to provide measurable incremental contribution.
