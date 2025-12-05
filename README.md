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
   Based on LightGBM feature-importance diagnostics:
   - Volatility state features with persistently low explanatory power were removed,
   - Sentiment features (S) were shifted to **short-term windows**,
   - Lagged return windows were compressed to **ultra-short horizons (2, 3, 5, 10, 21)**.

   After this structural redesign of the feature space, residual modeling began to provide **measurable incremental contribution**, leading to the final hybrid production model.
## 4. Position Construction & Risk Control

The corrected **forward return forecast** from the hybrid model is transformed into a tradable position through a **deterministic, volatility-aware, and crash-protected position sizing rule**.

### (1) Signal-to-Position Mapping

Let \(\hat{r}_{t+1}\) denote the hybrid model’s forward return forecast at time \(t\).  
Position construction proceeds in three steps: **volatility normalization, nonlinear squashing, and crash protection**.

**Step 1: Volatility-Normalized Signal (Predicted Sharpe)**  

Let \(\sigma_t\) be the 21-day volatility proxy (`lagged_forward_returns_std21`) and \(\sigma_{\min}\) a small positive volatility floor. The raw forecast is first normalized:

$$
s_t = \frac{\hat{r}_{t+1}}{\max(\sigma_t, \sigma_{\min})}
$$

so that \(s_t\) behaves like a **predicted Sharpe ratio**.

**Step 2: Smooth Nonlinear Position Mapping**

With sensitivity parameter \(K > 0\), the base position is defined as:

$$
\tilde{p}_t = 1 + \tanh\!\big( K \, s_t \big)
$$

which maps small signals to near-neutral exposure and large signals to higher long exposure in a **smooth and monotonic** manner.

**Step 3: Crash Brake and Final Clipping**

Let \(m_t\) denote the 21-day momentum proxy (`lagged_forward_returns_mean21`) and \(\theta_{\text{crash}}\) a negative crash threshold. If

If \(m_t < \theta_{\text{crash}}\) and \(\tilde{p}_t > 1\), the position is capped at \(\tilde{p}_t = 1\).

$$
p_t = \min\\big(2,\; \max(0,\; \tilde{p}_t)\big)
$$

ensuring all exposures remain within \([0, 2]\).

This construction guarantees that:
- Signals are **scaled by recent volatility** (Sharpe-like normalization),
- Positions are **smooth, bounded, and stable**,
- Overweight exposure is **automatically suppressed under crash conditions**.

---

### (2) Leverage and Exposure Control
- A **hard upper bound of 2** on position size is enforced to comply with the Kaggle exposure constraint.
- The baseline neutral exposure is centered at **1.0**, corresponding to full benchmark allocation.
- The strategy therefore operates in a **risk-budgeted long-only regime**, avoiding unconstrained leverage.

---

### (3) Causality and No Look-Ahead

All inputs to the position rule (forecasts, volatility, and momentum) are based exclusively on **lagged and historical information**.  
Together with the walk-forward training and evaluation protocol, this guarantees that each position \(p_t\) is formed using only information that would have been available at the actual decision time, with **no look-ahead bias**.


### (1) Walk-Forward Evaluation Protocol
- The full dataset is split chronologically.
- The **last 180 trading days** are reserved as a fixed **out-of-sample validation / test set**.
- All earlier data are used for **rolling training and validation**.
- At each time step:
  - The linear model is retrained **daily**,
  - The LightGBM residual model is retrained every **21 trading days**,
  - Predictions and positions are generated strictly forward in time.

No information from the validation or test period is used in model training or feature construction.

### (2) Trading Simulation
- At each date, the hybrid model produces a **forward return forecast**.
- The forecast is transformed into a trading position using the deterministic position sizing rule.
- Strategy returns are computed from realized forward returns and the corresponding positions.

All signals, positions, and returns are generated in a **fully causal manner** with no look-ahead bias.

### (3) Performance Metrics
- The primary evaluation metric is the **out-of-sample Sharpe ratio**.
- Additional diagnostics include:
  - Cumulative return curve,
  - Daily return distribution,
  - Maximum drawdown,
  - Turnover statistics.

Performance is reported **only on the forward hold-out window**, reflecting deployable rather than in-sample behavior.
