# Hull Tactical — predict, then size

This project is based on the **Hull Tactical Market Prediction** dataset from
Kaggle: <https://www.kaggle.com/competitions/hull-tactical-market-prediction/data>
It gives daily S&P 500 forward returns, the risk-free rate, and 93 raw predictors
grouped by prefix. The competition is over and nothing here was submitted to it.
The metric is kept because it is a reasonable scoring rule with a volatility
constraint built in.

**The model is the smallest part of this.** What the project is, besides its
result, in the order the three code directories run:

**1. Feature engineering** (`stage1/Hull_Tactical_feature_engineering.py`).
93 raw columns to 1132 features. Group-specific rolling windows, seven groups
each on its own clock. Causal group PCA refitted at every row on a trailing
window, inputs winsorised at the in-window 98th percentile. Regime gates built
from those PCA factors. A lagged return block at ultra-short horizons. Every
transform reads rows `<= t` only, and the three-way walk-forward split is cut
once and enforced structurally, so the holdout span is unreachable unless a
script names it.

**2. Feature diagnosis** (the rest of `stage1/`). The group factors are checked
for the sign flips their own construction can produce, at 4–5% of rows. Model
capacity is frozen before any feature is tested, because a tree at four leaves
touches 267 of 1132 columns and cannot report on the rest. Each feature block is
then removed in turn at a model large enough to reach it: every removal costs or
ties, so all 1132 columns stay. The last step separates the usable half of the
model's output from the unusable, with the magnitude overstated 11.5× at
`R2 = -0.261` against zero, and the order strong at a mean per-block Spearman of
`0.1060`, `t = 10.93`, positive in 91.3% of blocks.

**3. Model selection and mapping** (`stage2/`). Ridge and LightGBM order returns
about equally, and the Ridge ships because a configuration chosen on one span
carries to the next for it (+0.43) and inverts for the tree (−0.52). The rest
turns that order into a daily position under a causally solved volatility budget:
seven curve forms at matched risk, the amplitude read off the unpenalised Sharpe,
the reference set the percentile is taken against, the length of the removed mean,
and the turnover control.

Throughout, every contrast is reported beside its detectable width, twice the
standard error, rather than a p-value. Thresholds and selection rules are fixed
before the run, and the lines of work that closed are recorded with the
measurement that closed them.

**Every step, with its numbers and the script that produced it, is in
[`docs/performance_report.pdf`](docs/performance_report.pdf).** That report is a
record of the run: each step is a script, the span it was scored on, the numbers
it returned, and the question those numbers hand to the next step. It reports
measurements rather than interpreting them, so it is not a market analysis or a
strategy write-up. This file is the map.

---

## Result

Ridge on a rolling 756-row window, refit every 84 rows; the recent mean of its
own predictions removed; a trailing percentile of what is left; the scale solved
daily to the metric's volatility kink; the position smoothed with an EWMA.

Adjusted Sharpe, against buy-and-hold at 0.415 / 0.968 / 0.302:

| selection rule | configuration | dev | valid | **test** |
|---|---|---|---|---|
| **A** maximin(dev, valid) | `ewm3 × hl3 × rank252` | 0.451 | 1.030 | **0.435** |
| **B/C** max valid, max mean | `mean5 × hl2 × rank504` | 0.448 | 1.183 | **0.448** |
| **D** simplest flat-window mean | `mean8 × hl5 × rank252` | 0.449 | 1.011 | **0.399** |

`dev` fits, `valid` selects, `test` is held out and scored once. Three rules are
reported rather than one because the top of the dev+valid ranking is a tie.
Turnover runs 4–15% of notional a day, and neither of the metric's penalties ever
binds, so the adjusted Sharpe above is the plain Sharpe.

The strongest statement the evidence supports is not any one of those rows: of
the 90 smoothed configurations, 29 clear both dev and valid, and **all 29 are
positive on test**: mean +0.129, minimum +0.079.

---

## Where results are stored

The data and every cached result live outside the repo, beside it. This is what a
full run writes:

```
<working dir>/
│
├─ train.csv                                  # Kaggle raw series, 8990 rows
│
└─ kaggle_hull/
   ├─ all_feature_last_8990.csv               # Stage 1: 8918 rows × 1132 feature columns
   │
   ├─ splits/                                 # Stage 2: the only way in after this point
   │  ├─ train_processed.csv                  #   6609 rows, 5796 scored, 69 folds
   │  ├─ valid_processed.csv                  #   1597 rows,  840 scored, 10 folds
   │  ├─ test_processed.csv                   #   1694 rows,  924 scored, 11 folds
   │  └─ manifest.json                        #   where each slice begins, and under what constants
   │
   └─ probe/                                  # one directory per script, named after it
      │
      ├─ capacity_data/                       # Stage 3: per-cell fold scores
      ├─ capacity_window_grid/                # Stage 3: window × capacity  ← FROZEN HERE
      │  └─ g_W{w}_T{t}_bag7_seed{s}.csv      #   cached tree signals, reused by Stage 6
      ├─ embargo_capacity/                    # Stage 3: capacity re-swept at embargo 0
      │  └─ signals/e0_c117_{span}_seed{s}.csv  #   the tree cache Stage 8's contrast arm reads
      │
      ├─ ablation/                            # Stage 4: ablation at the default tree
      ├─ ablation_T117/                       # Stage 4: the same at 117 leaves
      ├─ tree_sensitivity/                    # Stage 4: five hyperparameter axes
      │
      ├─ ridge_signal/                        # Stage 5: level vs order, and the Ridge cache
      │  └─ ridge_{dev,valid,test}.csv           #   the only Ridge cache; read by Stage 8
      ├─ signal_diagnostics/                  # Stage 5: the same three tests, LightGBM
      │
      ├─ mapping_form/                        # Stage 6: curve, amplitude, and the two windows
      │
      ├─ level_refit/{tree,ridge}/            # Stage 7: can a fit's output level be estimated
      ├─ one_model/{tree,ridge}/              # Stage 7: refit count vs training size
      │
      ├─ detrend_sweep/                       # Stage 8: what is removed, and what it costs
      │
      ├─ pipeline/                            # Stage 9: the final rule
      │  ├─ signal.csv                        #   the Ridge OOS predictions, all three spans
      │  └─ pipeline.json                     #   the result table
      └─ scale_order/                         # Stage 9: solve the scale before or after smoothing
```

Every script writes `<name>.json` into its own directory and reads nothing from
another script's, except the two signal caches marked above.

---

## Scripts

Sixteen entry points and ten utilities. Each one settles a numbered step of the
report and caches its output, so re-running a later stage does not require
re-running an earlier one.

```
├─ tools/                                     # shared machinery, imported never run
│  ├─ paths.py                                #   every path; ROOT walks up to the data
│  ├─ wfo.py                                  #   walk-forward fold geometry, purge/embargo
│  ├─ hull_probe.py                           #   data, metric, frozen constants
│  ├─ split_data.py                           # 1.1: cut the split once; load_span / load_through
│  ├─ mapping.py                              #   detrended percentile, scale solver
│  ├─ ranking.py                              #   detrend, trailing rank, block statistics, tau
│  ├─ scoring.py                              #   cached signals, adjusted Sharpe, paired tests
│  ├─ capacity_data.py                        #   bagged tree signal, fold cache
│  ├─ step2_tails.py                          #   decile-tail scoring
│  ├─ diagnostics.py                          #   HAC OLS and spanning regression
│  ├─ select_probe.py                         #   feature-selection probe
│  └─ audit_config.py                         #   guard: assert the frozen configuration
│
├─ stage1/                                    # the signal layer
│  ├─ Hull_Tactical_feature_engineering.py    # 1.1: build the 1132 columns
│  ├─ pc1_check.py                            # 1.2: group-PCA sign-flip check
│  ├─ capacity_window_grid.py                 # 1.3: window × capacity, frozen at 756/117
│  ├─ embargo_capacity.py                     # 1.3: capacity re-swept at embargo 0
│  ├─ ablate_blocks.py                        # 1.4: remove one feature block at a time
│  ├─ tree_sensitivity.py                     # 1.5: five hyperparameter axes
│  ├─ ridge_signal.py                         # 1.6: fit and cache the Ridge, then diagnose it
│  └─ signal_diagnostics.py                   # 1.6: the same three tests, on the LightGBM
│
├─ stage2/                                    # the mapping layer
│  ├─ mapping_form.py                         # 2.1, 2.2, 2.6: --curves / --params / --axes
│  ├─ level_refit.py                          # 2.3, 2.4: --model {tree,ridge}, level estimators
│  ├─ one_model.py                            # 2.4, 2.5: --model {tree,ridge}, refits vs data
│  ├─ detrend_sweep.py                        # 2.3, 2.5-2.8: --grid / --axis / --control
│  ├─ pipeline.py                             # 3.2, 3.3: the rule, end to end, and the result
│  └─ scale_order.py                          #   A: solve the scale before or after smoothing
│
├─ docs/                                      # where the chain above is written up
│  ├─ performance_report.pdf                  #   THE REPORT: what each step ran and returned
│  ├─ method.tex                              #   its source
│  └─ METHOD.md                               #   the same chain in markdown
└─ README.md
```

**[`docs/performance_report.pdf`](docs/performance_report.pdf) is the end of that
chain.** It walks the files above in the order they were run, and for each one
gives the question, the numbers, what the numbers settled, and what the next file
therefore had to measure. Anything it does not measure, it does not claim.

The directories are for reading, not layering: imports cross between all three, so
every entry script puts the repo root and the three group directories on
`sys.path`, and `paths.py` walks up to find `kaggle_hull/`. No import line depends
on the layout.

`audit_config.py` sits in `tools/` because everything imports the constants it
checks, but it is run directly and exits with the number of mismatches.

---

## Steps

The numbers above and below are the report's section numbers, so a file in the
tree leads straight to the pages that use it.

| step | what it settles | span |
|---|---|---|
| **1.1** | the 1132 feature columns, the split, and that every transform is causal | — |
| **1.2** | the group factors survive the sign flips their own construction can cause | train |
| **1.3** | training window and model capacity, **frozen before any feature test** | dev |
| **1.4** | which feature blocks carry signal | dev |
| **1.5** | the remaining tree hyperparameters | dev |
| **1.6** | the model's order is informative and its magnitude is not | dev |
| **2.1** | the curve's shape does not matter | dev |
| **2.2** | the volatility budget does | dev |
| **2.3** | how much there is to gain from the reference set | dev |
| **2.4** | why the Ridge ships and the LightGBM does not | dev, valid |
| **2.5** | configurations are chosen on the metric, not on the ordering | dev, valid |
| **2.6** | how much of the recent mean to remove | dev, valid |
| **2.7** | turnover is the cost, and the metric does not charge it | valid |
| **2.8** | how to hold turnover down without giving up the edge | dev, valid |
| **3.1–3.3** | the selection rule, the rule itself, and the one reading of `test` | all three |
| **A** | the volatility budget is left unspent on purpose | dev, valid |

Step 1.3 comes before 1.4 on purpose. The first ablation ran at the default tree
settings and returned null for every block, but those settings grow four leaves on
a 756-row window and touch 267 of 1132 columns, so the null was a statement about
the model rather than the features. Capacity had to be fixed first: at 4 leaves
every block's effect sits inside ±0.067, at 117 the D group separates at −0.187
(t = −2.81).

Inside 1.1 the split comes after the feature builder, also on purpose. The builder
drops the leading warm-up rows, so the raw series has 8990 rows and the feature
table 8918; a split computed on the raw index lands 58 rows away from where the
models cut.

---

## Reproducing

```bash
python stage1/Hull_Tactical_feature_engineering.py   # writes all_feature_last_8990.csv
python tools/split_data.py --commit                 # writes the three slices and manifest.json
python tools/audit_config.py                        # asserts the frozen configuration
python stage2/pipeline.py --commit                  # 90 closed-form Ridge fits, then score
```

`pipeline.py` is self-contained: it imports nothing from the experiment scripts,
because a strategy should be readable without its search history.

Two conventions live side by side: the scripts that write a signal cache print a
plan and exit unless given `--commit`, while `capacity_window_grid.py`,
`ablate_blocks.py` and `tree_sensitivity.py` do the work by default and take
`--plan` for the dry run. `detrend_sweep.py` and `scale_order.py` score dev and
valid only; `--reveal-test` opens the third span.

## What this is not

The edge is small, `+0.036` on dev and that is the widest span, and it is scored by a
metric that charges nothing for trading on the most liquid index exposure there
is.

Two mechanisms are open. Why removing a *short* recent mean pays when removing
nothing orders better (§2.5), and why the useful lengths stop at 21 (§2.6). The
account that fitted the LightGBM does not apply to the model that ships.
