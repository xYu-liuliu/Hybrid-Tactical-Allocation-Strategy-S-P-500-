# Method — how the position rule was derived

Each section: the question, how it was measured, the result, the conclusion, and
what it rules out. Every number carries the script that produced it and the span
it came from.

**Split discipline.** `dev` (69 blocks, ~23 years, rows 1261–7056) fits models
and measures mechanisms. `valid` (10 blocks) chooses between configurations.
`test` (11 blocks) is read once, at the end. `dev_end = 7113`; the first holdout
block starts at row 7141. Part 1 touches neither holdout span.

**The metric** is the competition's adjusted Sharpe: `w` earns
`rf*(1-w) + w*fr`, divided by `1 + max(0, sigma_s/sigma_m - 1.2)` and by
`1 + gap^2/100`, with `w` in `[0, 2]`. Buy-and-hold is `w = 1` and is also the
market, so its adjusted Sharpe is its plain Sharpe.

**Terms.** `block_ic` — mean per-block Spearman between signal and next day's
excess return, one block being the 84 rows a single fit predicts.
`spread_ann` — annualised top-minus-bottom-decile return, averaged over folds.
`detectable` — twice the standard error of a paired contrast; a smaller
difference is not separable from zero.

**Naming.** The same parameter carries four names across the sections that sweep
it. This document uses `pipeline.py`'s.

| meaning | `pipeline` | `mapping_form` | `detrend_sweep` A/B | `detrend_sweep` C |
|---|---|---|---|---|
| rolling mean subtracted before ranking | `detrend` | `mu_window` | `roll21`, `none` | `mean8`, `ewm5`, `med8` |
| the percentile's reference window | `rank_window` | `rank_window` | `rank_win` | `rank_win` |
| EWMA half-life on the **position** | `smooth_halflife` | — | — | `smooth5` |

So `mean8 × smooth5 × rank252` reads: subtract the mean of the last 8
predictions, take the percentile within the last 252 days, solve the scale, then
apply an EWMA of half-life 5 days to the position.

---

# Part 1 — the signal layer

## 1.1 Build the features

`Hull_Tactical_feature_engineering.py`

**Question.** What are the inputs, and is any of them contaminated?

**Method.** 1132 numeric columns: rolling means, standard deviations, z-scores,
within-window position, long-window levels, regime gates, per-group PCA factors.
Audited for full-sample statistics.

**Result.** Every dispersion statistic is computed on
`s.rolling(w, min_periods=w)`; group factors come from
`causal_pc1_onfly_winsor`, refitted per row on a trailing window. No
`fit_transform`, no global mean or standard deviation in either file. A feature
at row `t` uses only rows `<= t`.

One structural dependency: the coverage filter normalises by the whole series,

```python
fv["coverage"] = 1.0 - fv["first_valid_row"] / float(len(df))   # len(df) = 8918
fv["is_thin"]  = fv["coverage"] < MIN_COVERAGE                  # 0.30
```

so its threshold is `6243` rather than the `4979` a development-only view gives;
normalising by development length would drop 71 more columns.

**Conclusion.** Causal. The one non-causal input is the scalar "how long is the
dataset" — no holdout value participates, and **no surviving column's data
begins inside the holdout** (latest first-valid row: 6039).

## The split, cut once — and the only way in

`split_data.py` · runs after 1.1, before everything else

**Question.** The split discipline above is a rule. What enforces it?

**Method.** The feature table is cut once into three files — `train_processed.csv`,
`valid_processed.csv`, `test_processed.csv` — and every script after this one
reads a span through `paths.load_split(span)` rather than the whole table. The
geometry is not written down; it is derived from `wfo.make_folds` under the frozen
constants, so it cannot drift from what the models actually do:

| | rows kept | scored | folds | history |
|---|---|---|---|---|
| `train` | 504–7112 | 1261–7056 | 69 | 757 |
| `valid` | 6384–7980 | 7141–7980 | 10 | 757 |
| `test` | 7224–8917 | 7981–8904 | 11 | 757 |

A span's models train on rows that precede it, so each file carries its own
scored rows **plus** the 757 rows of history those fits need — the training
window, the purge and the embargo. The three files therefore overlap, and any one
of them runs the full walk-forward for its span without the others. No columns
are added: each file is an exact slice, so `feature_cols` returns the same 1132
columns it always did, and where each slice begins is recorded in
`manifest.json`.

**Why this runs after the feature builder, not before.** The builder drops the
leading warm-up rows — `date_id` 0 to 71, where the longest rolling windows have
no history — so the raw series has 8990 rows and the feature table has 8918. The
fold grid is positional on the feature table, so a split computed on the raw
index lands 58 rows away from where the models cut. The split has to be taken on
the table the models read.

**Result.** Three entry points cover every later use. `load_span(span)` returns a
slice and its folds, indexed into it. `load_through(span)` returns every row from
the first training row through the end of that span, for protocols that train on
an earlier span and predict this one — one fit on all of development needs 6636
training rows while valid's slice carries 757. `remap_rows` and `slice_covering`
put a script's own fold grid onto a slice and refuse, loudly, when the slice does
not carry the history that grid needs.

**Conclusion.** `test` is unreachable unless a script names it. Nothing before
the final scoring does.

**Rules out.** The class of mistake where a holdout row reaches a fit through a
trailing statistic, and the quieter one where a fold grid silently loses folds
because a slice was too short to hold them.

## 1.2 Check that the group factors are still factors

`pc1_check.py` · **train split**

**Question.** `MIN_COVERAGE = 0.30` let late-starting columns back in, so the
PCA's active set now grows several times per group. Does PC1 survive that?

**Method.** Count sign flips and jumps in PC1 as columns come online. A rolling
z-score absorbs a scale change but not a sign change.

**Result.** Flips and jumps at about 4–5% of rows.

**Conclusion.** Small enough to keep the factors, large enough to record:
`V_pc1_z126` and `E_pc1_z252` are downstream of it.

## 1.3 Fix the model's capacity before testing the features

`capacity_window_grid.py` · **dev**, 69 folds, read off the causal trailing percentile

**Question.** Training window and leaf count — which matters?

**Method.** 3 × 5 grid, scored by annualised decile spread, with paired
contrasts on shared folds against `detectable`.

**Result.**

| window \ target | 5 | 20 | 62 | 117 | 176 |
|---|---|---|---|---|---|
| 252 | 0.183 | 0.519 | 0.670 | — | — |
| 504 | 0.446 | 0.672 | 0.672 | 0.626 | — |
| **756** | 0.364 | 0.699 | 0.704 | **0.854** | 0.928 |

| contrast | diff | t | detectable | separable |
|---|---|---|---|---|
| @756: T62 − T5 | +0.511 | 2.11 | 0.507 | yes |
| @756: T117 − T5 | +0.669 | 2.52 | 0.555 | yes |
| @756: T176 − T5 | +0.737 | 2.84 | 0.545 | yes |
| @T5: W252 − W756 | −0.171 | −0.76 | 0.464 | no |
| @T20: W252 − W756 | −0.247 | −1.11 | 0.458 | no |
| @T62: W252 − W756 | −0.180 | −0.79 | 0.470 | no |
| @T117: W504 − W756 | −0.039 | −0.17 | 0.491 | no |
| W756T176 − W756T62 | +0.195 | 1.85 | 0.215 | no |

**Conclusion.** Capacity separates; the window does not. Above 62 leaves the
gains are not separable from each other, so 62/117/176 are one band, not a
ranking. **Frozen at window 756, target 117** — the middle of that band, at the
longest window. `embargo_capacity.py` later re-swept {5,20,62,117} at embargo 0
and put 117 top on dev again (block_ic 0.1011 against 0.0943 at 62).

## 1.4 Ablate the feature blocks — at a model large enough to use them

`ablate_blocks.py` · **dev**, 69 folds, window 756, target 117, read off the
causal trailing percentile

**Question.** Which feature blocks carry signal?

**Method.** Remove one structural block at a time — but only at a capacity large
enough for the model to use the columns. On a 756-row window:

| | splits/tree | distinct columns/tree | columns ever used |
|---|---|---|---|
| 4 leaves | 3 | 2.9 | **267 of 1132 (24%)** |
| 92 leaves | 91 | 65.7 | 666 of 1132 (59%) |

A model that touches a quarter of the columns cannot report whether the other
three quarters matter, which is why 1.3 comes first. This runs at 117.

**Result.**

| removed | columns | spread_ann | vs baseline | t | detectable | at 4 leaves |
|---|---|---|---|---|---|---|
| baseline | 0 | 0.854 | — | — | — | — |
| **D group** | 18 | 0.707 | **−0.187** | **−2.81** | 0.136 | −0.067 |
| long `_mean` | 97 | 0.754 | −0.095 | −1.32 | 0.147 | +0.013 |
| all `_pos` | 261 | 0.691 | −0.172 | −1.78 | 0.197 | +0.024 |
| long `_z` | 64 | 0.864 | +0.073 | +0.98 | 0.152 | +0.037 |
| gates | 159 | 0.699 | −0.109 | −1.32 | 0.169 | −0.051 |

**Conclusion.** Removing any block costs or ties; nothing gains. Only the D
group is separable, and it is a loss. **Keep all 1132 columns.**

**Rules out.** Testing features on a model too small to use them — the last
column is the same measurement at four leaves, where every effect sits inside
±0.067.

## 1.5 The remaining tree hyperparameters

`tree_sensitivity.py` · **dev**, 69 folds, one axis at a time, read off the
within-block ranking

**Question.** Does anything below the capacity choice move the result?

**Result.**

| axis | default | best alternative | difference | detectable |
|---|---|---|---|---|
| learning_rate | **0.05** | 0.01 | −0.060 | 0.198 |
| colsample_bytree | **0.7** | 0.9 | −0.065 | 0.242 |
| reg_lambda | **5.0** | 1.0 | −0.051 | 0.210 |
| subsample | **0.7** | 0.85 | −0.176 | 0.244 |
| n_estimators | **250** | 500 | +0.010 | 0.065 |

Extremes are genuine losses: learning_rate 0.20 at −0.528 (t = −4.22),
reg_lambda 100 at −0.651, colsample 1.0 at −0.268 (t = −2.70).

**Conclusion.** Every deviation is negative or inside its own resolution. **The
defaults stand.**

## 1.6 What kind of signal this is

`ridge_signal.py` (both models) · `signal_diagnostics.py` (LightGBM only)
· **dev**, 69 folds

**Question.** Is the model's magnitude usable, is its order usable, and does the
magnitude add anything the order does not? The whole position rule rests on the
answer, and the model that ships is a Ridge, so both are measured.

**Method.** (A) `R2_oos` against zero and against the trailing mean;
Mincer-Zarnowitz slope with Newey-West errors; then the same slope re-estimated
from earlier rows only and applied forward. (B) per-block Spearman. (C) inside
each 84-row block, `r` = normal score of the within-block rank (order only) and
`m` = `(s − block mean) / block sd` (order and magnitude); regress `y` on each
and on both.

**Result A — the level.**

| | sd(pred) | sd(truth) | R² vs zero | vs trailing mean |
|---|---|---|---|---|
| **ridge** | 0.0062 | 0.0110 | **−0.261** | −0.260 |
| tree | 0.0044 | 0.0110 | −0.119 | −0.117 |

| | MZ slope `b` | NW `t` | `1/b` | causal `b` | R² after causal rescale |
|---|---|---|---|---|---|
| **ridge** | **0.0872** | **3.03** | **11.5×** | 0.0899 | **+0.0015** |
| tree | 0.1393 | 2.54 | 7.2× | 0.1431 | +0.0015 |

**Result B — the order.**

| | block_ic | t | blocks positive | pooled IC |
|---|---|---|---|---|
| **ridge** | **0.1060** | **10.93** | **91.3%** | 0.0448 |
| tree | 0.1011 | 8.11 | 87.0% | 0.0531 |

**Result C — the increment.**

| signal | model | R² | t on `r` | t on `m` |
|---|---|---|---|---|
| **ridge** | r only | 0.0108 | 8.36 | |
| | m only | 0.0109 | | 8.28 |
| | **both** | **0.0110** | **0.46** | **0.74** |
| tree | r only | 0.0094 | 7.57 | |
| | m only | 0.0117 | | 8.10 |
| | **both** | **0.0136** | −2.41 | **+3.52** |

**Conclusion.** There is a linear signal (`t = 3.03`) that the prediction
overstates by 11.5×, and the shrinkage was knowable at the time — which is why
R² moves from −0.261 to +0.0015. The order is strong. For the Ridge the
magnitude adds `+0.0002` of R² and neither coefficient survives, so **discarding
the magnitude costs nothing**. **Keep the rank, map through a percentile.**

**Rules out.** Every rule that converts a prediction into return units before
sizing.

**Carried forward as a constraint.** For the tree, `m` stays significant and
joint R² rises 0.0094 → 0.0136: a rank-based mapping handicaps it specifically.
Part 2's search must offer the tree a magnitude encoding, or dropping it would
be circular.

**Scope.** `ridge_signal.py` fits each span's signal from that span's own slice
and diagnoses the development span only. `signal_diagnostics.py` reports the same
three tests for the LightGBM; its section A is parameter-free, its section B is
computed through a particular mapping and is not cited here.

---

# Part 2 — the mapping layer

## 2.1 Every shape of the rank-to-position curve works, and none works better

`mapping_form.py --curves` · **dev**, 69 folds, `wbar = 1.0`, scale solved causally to a
volatility ratio of 1.2

**Question.** Part 1 leaves one number per day — where today sits among the last
252. What curve turns it into a position?

**Method.** Seven forms from the literature (AFML sigmoid sizing, GKX
extreme-decile spread, dead bands), each given its own causally-solved scale so
none can win by taking more risk. They are not cosmetic variants:

| `p` | linear | normal score | decile | extreme decile | dead band |
|---|---|---|---|---|---|
| 0.02 | 0.31 | 0.21 | 0.34 | 0.31 | **0.02** |
| 0.25 | 0.64 | 0.74 | 0.63 | **1.00** | **1.00** |
| 0.50 | 1.00 | 1.00 | 1.07 | 1.00 | 1.00 |
| 0.75 | 1.36 | 1.26 | 1.37 | **1.00** | **1.00** |
| 0.98 | 1.69 | 1.79 | 1.66 | 1.69 | **1.98** |

**Result.**

| curve | pooled adj | vs market | t | vs linear | detectable |
|---|---|---|---|---|---|
| normal score | 0.6103 | +0.2817 | 3.22 | +0.0019 | 0.045 |
| **linear** | 0.6015 | **+0.2799** | **3.25** | — | — |
| decile 10 | 0.5866 | +0.2813 | 3.36 | **+0.0014** | **0.021** |
| multi-horizon | 0.5862 | +0.2942 | 3.57 | +0.0143 | 0.050 |
| extreme decile | 0.5807 | +0.2820 | 2.79 | +0.0021 | 0.143 |
| dead band | 0.5777 | +0.2798 | 3.24 | −0.0000 | 0.132 |
| isotonic | 0.4788 | +0.2032 | 4.60 | −0.0766 | 0.128 |

| curve | turnover | days holding a view |
|---|---|---|
| dead band | **0.159** | **21.5%** |
| extreme decile | 0.190 | 21.5% |
| linear | 0.274 | 98.6% |
| multi-horizon | 0.288 | 99.0% |

**Conclusion.** Every curve beats buy-and-hold (+0.28 per fold, `t` 2.8–4.6;
pooled 0.60 against the market's 0.406) and none is separable from any other —
the tightest null is decile 10 at +0.0014 against a detectable of 0.021.
Turnover varies 1.8× and time in the market 4.6× with no change in score: once
the budget is fixed, any monotone transform of the rank extracts the same
information and the shape only redistributes where risk is taken. **Take the
linear curve.**

**Rules out.** Shape tuning. It also sets the baseline the rest has to explain:
this same mapping carried forward unchanged scores −0.390 on valid (2.6's
`roll63` column), so the shape is not what fails out of sample.

**Leakage note.** `_std` normalises by one full-sample standard deviation, but
the causal solver cancels it exactly — scaling `z` by `c` makes `solve_tau`
return `tau/c`, measured at 4.4e-16 beyond row 120. The residual covers 120 of
5796 rows and is identical across all seven candidates. Everything else is
causal; the isotonic fit is refitted every 84 rows on rows strictly before the
block. No fold reaches past row 7066.

## 2.2 How much of the signal can be deployed

`mapping_form.py --params` · **dev**, 69 folds · *LightGBM evidence; corroborated on the
Ridge at the end of this document*

**Question.** 2.1 fixed the curve and left its amplitude open. The position is
`w = 1 + tau*z`; `tau` is solved so `w` runs at a chosen ratio of market
volatility. The metric taxes above 1.2 linearly — stop at the kink, or cross it?

**Method.** Sweep the target ratio. Read the **unpenalised** Sharpe, which has
the tax divided out, so it measures the signal rather than the metric.

**Result.**

| budget | adjusted | **unpenalised Sharpe** | vol penalty | mean tau |
|---|---|---|---|---|
| 1.0 | 0.410 | 0.410 | 1.000 | 0.009 |
| 1.1 | 0.550 | 0.550 | 1.000 | 0.291 |
| **1.2** | **0.602** | 0.602 | 1.000 | 0.464 |
| 1.3 | 0.572 | **0.617** | 1.080 | 0.609 |
| 1.4 | 0.541 | **0.617** | 1.140 | 0.740 |
| 1.5 | 0.524 | **0.616** | 1.176 | 0.862 |
| 1.7 | 0.509 | **0.619** | 1.216 | 1.090 |
| 2.0 | 0.495 | **0.617** | 1.246 | 1.409 |

| how `tau` is obtained | pooled adjusted | realised ratio | vol penalty |
|---|---|---|---|
| one solve per fold, held 84 days | 0.6281 | **1.234** | **1.034** |
| **re-solved every day** | **0.6773** | 1.199 | 1.000 |

`tau_min_hist` 60/120/252 → 0.6054/0.6015/0.6011. `tau_fallback` 0.0/0.3/0.6/1.0
→ 0.6008/0.6015/0.6020/0.6037. Solver window 252/504/756 → 0.580/0.600/0.602.

**Conclusion.** The unpenalised Sharpe climbs +0.19 from 1.0 to 1.2 then reads
0.617 flat across a doubling of the swing, while the tax grows linearly: past a
ratio of ~1.3 a wider swing adds variance and no return. **`wbar = 1.0`, budget
1.2, `tau` re-solved daily on a trailing 756 rows.** The daily re-solve is worth
+0.049 pooled — the same order as the budget itself — and its per-fold paired
difference is −0.0096 against a detectable of 0.115, so only the pooled number
sees it. The solver's own constants are free (±0.005). Sanity: budget 1.0 scores
0.410 with mean tau 0.009, reproducing the market's 0.406.

**Rules out.** Crossing the kink, and picking `tau` by hand.

**Transfers / does not.** The saturation is a signal property and carries; 1.2 is
this metric's kink and does not.

---

## 2.3 How much is there to gain from the reference set

`detrend_sweep.py --grid` · **Ridge, dev**, 69 folds

**Question.** 2.1 fixed the curve and 2.2 its amplitude. One degree of freedom
is left: what the percentile is measured against. The shipped rule ranks today
against the trailing 252 days, after removing a 63-day rolling mean. The
alternative is the 84 rows *this fit produced* — block and fit are the same
thing, since each model predicts exactly one block. How much is in it?

**Method.** Re-rank the same Ridge predictions against each reference, and split
the within-block rank into its parts: the block's actual mean alone, then the
full rank.

**Result.**

| reference | block_ic |
|---|---|
| trailing 252, after a 63-day mean (**shipped**) | 0.0756 |
| trailing 252, **no mean removed** | **0.0950** |
| the block's actual mean removed (look-ahead) | 0.1002 |
| the full within-block rank (look-ahead) | 0.1060 |

| gap to perfect centring | |
|---|---|
| from the shipped rule | 0.0247 |
| **from no mean removed** | **0.0053** |

**Conclusion.** **The Ridge has almost no level problem.** Ranking against a
trailing window with nothing removed sits within 0.0053 of knowing the block's
true centre and within 0.0110 of the full within-block rank. Of the 0.0247 the
shipped rule gives away, **0.0194 — four fifths — is the 63-day mean damaging
the signal**, not a level the reference set fails to remove. 63 is the wrong length for this model, and 2.6 finds the
right one.

**Rules out.** For this model, the whole line of work that estimates a fit's
output level and subtracts it. A perfect level estimate is worth 0.0053 here —
`level_refit.py --model ridge` confirms it directly: the best training-side estimator
correlates at +0.342 (2.9 standard errors, so it is *not* flat) but its
dispersion is a fifth of the truth's (0.0009 against 0.0042), so it captures
about 12% of the level variance and is worth roughly +0.0015 block_ic. Nothing
in that line is large enough to matter.

## 2.4 Why the model is a Ridge

`one_model.py --model {tree,ridge}`, `level_refit.py --model {tree,ridge}`,
`detrend_sweep.py` · **dev + valid**

**Question.** Both models order returns about equally on dev (1.6: block_ic
0.1060 for the Ridge, 0.1011 for the tree). Which should ship?

**Method.** Run the same diagnostics on both, from one script each with a
`--model` switch, so nothing but the fit differs: the same splits, the same fold
grid at embargo 0, both fitted fresh.

**Result — the tree carries a fit-specific level and the Ridge does not.** Hold
the mapping fixed and change only how the signal is produced, over identical
valid rows. `p_window` is the within-block rank: look-ahead, and immune to the
block's level by construction, so it is the ceiling on what any causal rule could
reach.

| signal produced by | tree, causal | tree, ceiling | **Ridge, causal** | Ridge, ceiling |
|---|---|---|---|---|
| refit every 84 rows, 756-row window | **0.0062** | 0.0468 | **0.0984** | 0.1026 |
| 1 fit, 756-row window | 0.0399 | 0.0780 | 0.0843 | 0.0861 |
| 1 fit, ~6600-row window | **0.0668** | 0.0896 | **0.0425** | 0.0835 |

The two ladders run in opposite directions. The tree's causal ordering improves
tenfold as refits are removed; the Ridge's is best with the most refits and
decays as they are taken away.

Under the shipped protocol the tree sits at 0.0062 against a ceiling of 0.0468 —
a gap of **0.041** — while the Ridge sits at 0.0984 against 0.1026, a gap of
**0.004**. The gap says *where* the tree's loss is: not in its ordering ability,
since the ceiling is still there, but in the level. And the decisive number is
simpler than the ratio between the gaps: **the Ridge's causal ordering, 0.0984,
is more than double the tree's look-ahead ceiling, 0.0468.** Even a perfect level
repair would not bring the tree to where the Ridge already stands without one.

The persistence contrast says the same thing. A block's realised level against
the previous block's, within one fit and across a refit:

| | refit every 84 rows | 1 fit, 756-row window |
|---|---|---|
| tree | **−0.39** | **+0.65** |
| Ridge | −0.02 | +0.12 |

Same rows, same period, same algorithm; the only difference is whether a refit
sat between the two blocks. The tree shows it strongly and the Ridge barely at
all.

The same split appears in the level estimators (dev, 69 folds, standard error
0.120):

| estimator | LightGBM | **Ridge** |
|---|---|---|
| the model predicting its own last 84 training rows | +0.118 | **+0.342** |
| the model predicting its whole training set | +0.211 | **+0.415** |
| mean of the training labels | +0.212 | +0.366 |

For the tree nothing clears two standard errors: its level is produced when the
model meets the test period and is unreadable beforehand. For the Ridge the same
readings correlate — the level is a milder, partly foreseeable thing. So the
tree's 0.041 is not merely unclaimed, it is **unreachable**: 2.6 closes every
route to the estimate that would claim it.

**Result — and a fit-specific level does not transfer.** Correlation of `vs_bh`
across the mapping search's 40 cells, dev → valid:

| | Pearson | Spearman |
|---|---|---|
| tree | **−0.519** | −0.506 |
| **Ridge** | **+0.426** | +0.376 |

Choosing a configuration on dev actively hurts the tree on valid. The project's
own historical benchmark for this number is +0.070 across 37 vol overlay
candidates.

**Result — the consequences.** Under the identical shipped mapping:

| | dev | valid |
|---|---|---|
| tree | +0.089 | **−0.069** |
| **Ridge** | +0.034 | **+0.043** |

and in the search of 2.8, configurations clearing dev **and** valid: tree
**0 of 210**, Ridge **90 of 210**. The tree clears nothing anywhere in that
grid, under any turnover control.

**A limit on all of this.** valid is ten blocks, and the tree arm is seven bags
at one seed against the Ridge's closed form. The directions are unambiguous and
they agree across three independent measurements; the individual values are not
precise.

**Conclusion.** **Ship the Ridge.** The mechanism is one thing measured three
ways: each of the tree's fits carries its own output level, and a level that
belongs to a fit cannot survive being refitted — so nothing chosen on one span
carries to the next. The Ridge's output level barely moves between fits, which
is why its configuration space transfers and the tree's inverts.

**Rules out.** The tree. It also closes 1.6's open constraint: the tree was
owed a magnitude encoding, since a rank-only mapping is lossless for the Ridge
(`t` on the magnitude = 0.74) and lossy for the tree (`t` = 3.52). It is not
dropped for lack of that — it is dropped because nothing selected on it
transfers.

## 2.5 Select on the scored metric, not on the ordering

`detrend_sweep.py --grid --axis` · **Ridge, dev + valid**

**Question.** 2.3 found that removing no mean gives the best ordering. Does it
give the best result?

**Result.** Ridge, rank 252:

| span | detrend | block_ic | **vs bh** | exposure | vol_ratio | turnover |
|---|---|---|---|---|---|---|
| dev | none | **0.0950** | +0.028 | 1.004 | 1.211 | 0.191 |
| dev | roll21 | 0.0558 | **+0.060** | 1.005 | 1.195 | 0.298 |
| valid | none | **0.089** | **−0.067** | 1.006 | 1.177 | 0.124 |
| valid | roll8 | 0.082 | **+0.384** | 0.999 | 1.143 | 0.344 |
| valid | roll63 | 0.064 | −0.072 | 0.989 | 1.160 | 0.131 |

**Conclusion.** The configuration that orders best pays least, on both spans. On
valid the two rules have nearly the same ordering (0.089 against 0.082), the
same average exposure (1.006 against 0.999) and the same realised volatility
(1.177 against 1.143), and differ by **0.45** in adjusted Sharpe. So the
difference is neither ordering skill nor risk taken; it is *when* the position
moves. `block_ic` is a rank correlation weighting every day equally, while the
metric is carried by a small number of large days, and a rank correlation cannot
see the difference. **`block_ic` is demoted to a diagnostic; configurations are
chosen on adjusted Sharpe against buy-and-hold.**

**Open.** The responsiveness account above is inferred from what the other
columns rule out, not measured directly. A return-weighted IC, or the same
comparison split by the size of the day's move, would test it.

## 2.6 The length of the mean that gets removed

`mapping_form.py --axes`, `detrend_sweep.py` · **dev** and **valid**

**Question.** 2.3 says the 63-day mean is the wrong value and 2.5 says the
choice is made on the metric. What length is right?

**Method.** Sweep the length, on both spans, scored by the metric.

On dev alone the sweep does not resolve it. `mapping_form.py --axes`, pooled on dev,
puts `mu_window = 21` at the top of every rank-window column, but the paired
contrast against 63 is **+0.0053 against a detectable of 0.106** — twenty times
inside the noise floor. Dev cannot separate the short lengths; valid can.

**Result.** Ridge, rank 252, valid:

| length | 5 | 8 | 13 | **21** | **34** | 42 | 63 | 126 | 252 | none |
|---|---|---|---|---|---|---|---|---|---|---|
| vs bh | +0.264 | **+0.384** | +0.198 | +0.031 | **−0.025** | −0.010 | −0.072 | −0.082 | −0.099 | −0.067 |

The break sits between 21 and 34. Two alternatives in the same grid, dev and
valid:

| detrend | dev | valid |
|---|---|---|
| none at all | +0.028 | −0.067 |
| mean of the previous 1 complete block | −0.010 | −0.074 |
| previous 3 blocks | +0.003 | −0.096 |
| previous 6 blocks | −0.008 | −0.062 |
| every previous block | +0.017 | −0.067 |
| **rolling 8** | **+0.074** | **+0.384** |

**Conclusion.** **A trailing mean of 5 to 21 rows.** Removing nothing loses money
despite ordering best (2.5); removing a long mean loses money and orders badly;
only the short end does both acceptably.

**Rules out.** Lengths ≥ 34, removing nothing, and the block-aligned means — a
level decided at refit time and held constant through the block, which never
straddles a boundary and is negative on valid in every variant. So the answer is
neither "avoid the block boundary" nor "leave the level alone".

**Open.** The mechanism is not settled. There is a clean account for a model
whose level belongs to the fit — a mean of length `L` reaches into the previous
fit on about `L/84` of days, and consecutive fits' levels are uncorrelated — but
2.4 shows the Ridge is not such a model, so it does not apply here. What is
established is the empirical break, not why it sits at 21.

## 2.7 Turnover is the cost, and the metric does not charge it

`detrend_sweep.py --axis` · **Ridge, valid**

**Question.** 2.6 says use a short mean. What does that cost?

**Result.** Turnover is monotone in the length of the mean:

| length | 5 | 8 | 13 | 21 | 34 | 63 | 252 | none |
|---|---|---|---|---|---|---|---|---|
| daily turnover | 0.399 | 0.344 | 0.275 | 0.217 | 0.171 | 0.131 | 0.129 | 0.124 |

A shorter mean means a faster-moving position, through the same operation. At the
short end the rule changes 34 to 40% of notional every day, and unconstrained
configurations across the whole grid run from 25 to 50%.

**Conclusion.** The metric contains no execution: it scores a position series, so
a rule trading 40% of notional a day and one trading 5% receive the same
treatment. Nothing in it constrains turnover, and nothing in dev or valid does
either.

**Decision — a constraint from outside the metric.** Trading 40% of notional a
day is not a deployable product whatever the metric says, so **the position is
required to be smoothed**, and the parameters are chosen inside that constraint.
This is a prior about what kind of rule is worth having, not a measurement.

**Rules out.** Nothing on the evidence. It rules the unsmoothed family out on
grounds the data does not supply.

## 2.8 Continuous smoothing gives up the least edge per unit of turnover

`detrend_sweep.py --control` · **Ridge, dev + valid**

**Question.** Given that turnover must come down, which control keeps the most?

**Method.** 15 ways of computing the removed mean × 7 controls × 2 rank windows,
controls applied to the position after the scale is solved. Read as edge retained
per unit of turnover spent, averaged over all fifteen means.

**Result.** Dev:

| control | vs bh | turnover | edge per unit turnover |
|---|---|---|---|
| unconstrained | 0.059 | 0.324 | 0.18 |
| band 0.05 | 0.044 | 0.269 | 0.16 |
| band 0.10 | 0.030 | 0.224 | 0.13 |
| band 0.20 | 0.000 | 0.154 | **0.00** |
| EWMA half-life 2 | 0.027 | 0.095 | 0.28 |
| EWMA half-life 3 | 0.031 | 0.070 | 0.44 |
| **EWMA half-life 5** | 0.035 | 0.047 | **0.74** |

Configurations clearing dev **and** valid, by family and control:

| family | unconstrained | band | EWMA |
|---|---|---|---|
| short mean (3–21) | 20 of 22 | 41 of 66 | **29 of 66** |
| long mean (34–63) | **0 of 6** | **0 of 18** | **0 of 18** |
| no mean removed | **0 of 2** | **0 of 6** | **0 of 6** |

**Conclusion.** A no-trade band is dominated at every level: band 0.20 gives up
the entire edge (+0.000) to halve turnover, while an EWMA of half-life 5 keeps
59% of it at a seventh of the trading. The reason is what each does to the
level — a band holds the position unchanged on most days and loses track of it;
an EWMA moves every day by less and keeps it. **The control is an EWMA on the
position.**

**And the family separation is not a turnover effect.** The long-mean and
no-removal families clear nothing under *any* control, including no control at
all, so 2.6's result does not depend on how turnover is handled.

---

# Selection

`detrend_sweep.py --control` · **dev + valid only**, scored by the competition metric with
nothing added to it

**The constraint, fixed first (2.7).** The position must be smoothed. That leaves
90 configurations: 15 ways of computing the removed mean × 3 EWMA half-lives × 2
rank windows.

**The rule, pre-registered.** Maximise the smaller of dev and valid — a maximin
rule, requiring both spans to stand up, needing no threshold.

**The ranking:**

| removed mean | EWMA half-life | rank window | dev | valid | maximin |
|---|---|---|---|---|---|
| **ewm3** | **3** | **252** | +0.036 | +0.063 | **0.036** |
| ewm3 | 3 | 504 | +0.035 | +0.064 | 0.035 |
| mean8 | 5 | 252 | +0.034 | +0.043 | 0.034 |
| med8 | 5 | 252 | +0.034 | +0.068 | 0.034 |
| ewm3 | 2 | 252 | +0.033 | +0.118 | 0.033 |
| mean5 | 2 | 504 | +0.033 | +0.216 | 0.033 |
| ewm3 | 2 | 504 | +0.032 | +0.122 | 0.032 |
| mean5 | 3 | 504 | +0.032 | +0.154 | 0.032 |

**The top of this ranking is a tie.** The first eight rows differ by 0.004 in
maximin score, far inside any resolution these spans support, and 29 of the 90
clear both spans. So rather than name one configuration, three rules are applied
and all three are carried forward:

| rule | selects | why this rule |
|---|---|---|
| **A** maximise min(dev, valid) | `ewm3 × half-life 3 × rank 252` | pre-registered; requires both spans to hold, needs no threshold |
| **B** maximise valid subject to dev > 0 | `mean5 × half-life 2 × rank 504` | uses valid for what valid is for, with dev only as a gate |
| **C** maximise mean(dev, valid) | `mean5 × half-life 2 × rank 504` | equal weight to both spans — it picks the same cell as B |
| **D** best flat-window mean | `mean8 × half-life 5 × rank 252` | not a rule over the grid: the three ways of computing the mean tie at the top, and a flat window is the simplest form of the operation |

B and C agree, so the three rules give two configurations; D is added because
the choice between an exponentially weighted, flat-window and median mean is a
tie on the evidence, and a reader reproducing this should see the plainest
version. **A reported result that depends on which of these was used is worth
less than one that does not**, so all are reported.

# The rule

`pipeline.py`

```
Ridge(alpha=1000)  │  rolling 756  │  refit every 84  │  1132 columns
    preprocessing: ffill → standardise (degenerate columns pinned)
                   → clip to ±10 training SD
    ↓  s(t)
v(t) = s(t) − recent mean of s              remove the slow level, once
    ↓
p(t) = percentile of v(t) in a trailing window     turn it into an order
    ↓
z(t) = sqrt(3) · (2·p(t) − 1)
    ↓
tau(t) solved daily on trailing 756 so realised vol ratio = 1.2
    ↓
w*(t) = clip(1 + tau(t)·z(t), 0, 2)
    ↓
w(t)  = EWMA(w*, a few days)                hold turnover near 10% a day
```

Three settings of the two free lengths and the rank window, one per rule in
*Selection*:

| rule | removed mean | rank window | position EWMA |
|---|---|---|---|
| **A** maximin | exponential, half-life 3 | 252 | half-life 3 |
| **B/C** max valid, max mean | flat window of 5 | 504 | half-life 2 |
| **D** flat-window mean | flat window of 8 | 252 | half-life 5 |

## Which parameters the data actually fixed

| parameter | value | separable | where |
|---|---|---|---|
| model | Ridge | yes — transfer dev→valid +0.43 against −0.52 | 2.4 |
| curve shape | linear | **no** — 7 forms inside 0.021 | 2.1 |
| mean exposure `wbar` | 1.00 | **no** — flat over 0.9–1.1 | 2.2 |
| volatility budget | 1.2 | yes | 2.2 |
| scale re-solve cadence | daily | pooled yes (+0.049), per-fold no | 2.2 |
| the removed mean | short, 3–8 | family yes (≤21 against ≥34), within-family **no** | 2.3–2.6 |
| rank window | 252 or 504 | **no** — both clear | 2.6, Selection |
| turnover control | EWMA, half-life 2–5 | control **yes**, half-life **no** | 2.7–2.8 |

Half of them are ties, which is why *Selection* carries three configurations
rather than one: the curve's shape, the mean exposure, the rank window, which
length inside the short family, and which half-life smooths the position are all
"any value in this range" rather than "the data picked this". What the data did
fix is the model, the volatility budget, the solve cadence, the *family* the
removed mean belongs to, and that the turnover control is an EWMA rather than a
band.

## Result

Every row below was selected on dev and valid alone, by the correspondingly
lettered rule in *Selection*. `test` was scored afterwards.

Adjusted Sharpe, against buy-and-hold at 0.415 / 0.968 / 0.302:

| rule | configuration | dev | valid | **test** |
|---|---|---|---|---|
| **A** maximin | `ewm3 × hl3 × rank252` | 0.451 | 1.030 | **0.435** |
| **B/C** max valid, max mean | `mean5 × hl2 × rank504` | 0.448 | 1.183 | **0.448** |
| **D** flat-window mean | `mean8 × hl5 × rank252` | 0.449 | 1.011 | **0.399** |

Against buy-and-hold:

| rule | dev | valid | **test** | turnover (test) | vol ratio (test) |
|---|---|---|---|---|---|
| **A** | +0.036 | +0.063 | **+0.133** | 0.106 | 1.057 |
| **B/C** | +0.033 | +0.216 | **+0.146** | 0.151 | 1.077 |
| **D** | +0.034 | +0.043 | **+0.096** | 0.068 | 1.041 |

All three clear buy-and-hold on all three spans. Average exposure is 1.00
everywhere, turnover 4–15% of notional a day, and both of the metric's penalties
are exactly 1.000 throughout — realised volatility never approaches the 1.2 kink
and the strategy never underperforms the market — so the adjusted Sharpe reported
here *is* the plain Sharpe.

**The strongest statement the evidence supports** is not about any one of these
rows. Of the 90 smoothed configurations, 29 clear both dev and valid; **all 29
are positive on test**, mean +0.129, minimum +0.079. Meanwhile the long-mean and
no-removal families clear nothing under any turnover control, including none at
all.

**Three checks made on dev and valid also held here**, and none of them entered a
decision. 2.6's length sweep keeps its sign in 11 of 12 cells. 2.4's cell-to-cell
transfer stays positive for the Ridge (+0.224 from dev, +0.476 from valid) and
negative for the tree (−0.741 from valid). Under the shipped mapping the tree
reaches +0.012 against the Ridge's +0.096, so the model choice would have been the
same had test been available.

**One thing the constraint cost.** Under the competition metric more turnover
scores better: averaged over the grid, an unconstrained position beats every
smoothed one on both development spans (2.7). Requiring the position to be
smoothed therefore gave up measured score on dev and valid, and it was imposed on
deployability grounds rather than because anything in the data asked for it.

## What this result is not

The edge is small: +0.036 on dev, against a pre-registered threshold of +0.10. It clears buy-and-hold on all three spans for a reason
that replicates across a family of configurations, but it is not a large number,
and it is scored on daily equity-index exposure by a metric that charges nothing
for trading.

Two mechanisms remain open. Why a short mean pays when it orders no better than
removing nothing (2.5), and why the break sits at 21 rather than elsewhere (2.6)
— the account that fitted the LightGBM does not apply to the model that ships.

## The volatility budget is left unspent on purpose

`scale_order.py` · **dev + valid**

**Question.** Smoothing comes after the scale is solved, so realised volatility
falls from the 1.19 the solver aims at to about 1.06 — the penalty never binds
and part of the allowance is unused. Reclaim it?

**Method.** An EWMA is linear, so `EWMA(wbar + tau*z) == wbar + tau*EWMA(z)`.
Solving the scale on the *smoothed* signal aims it at what will actually be held.
Three arms, the third a control that filters the signal but keeps the old scale.

**Result.** All three shipped configurations, dev and valid, against
buy-and-hold:

| config | arm | dev | valid | maximin | turnover (dev) | vol ratio |
|---|---|---|---|---|---|---|
| **A** maximin | smooth after the solve (shipped) | **+0.036** | +0.063 | **0.036** | 0.087 | 1.083 |
| | solve on the smoothed signal | +0.019 | +0.093 | 0.019 | 0.157 | 1.190 |
| | filter the signal, keep the old scale | +0.036 | +0.067 | 0.036 | 0.087 | 1.080 |
| **B/C** max valid | smooth after the solve (shipped) | **+0.033** | +0.216 | **0.033** | 0.131 | 1.090 |
| | solve on the smoothed signal | +0.008 | +0.343 | 0.008 | 0.222 | 1.188 |
| | filter the signal, keep the old scale | +0.033 | +0.220 | 0.033 | 0.131 | 1.088 |
| **D** flat-window mean | smooth after the solve (shipped) | **+0.034** | +0.043 | **0.034** | 0.057 | 1.061 |
| | solve on the smoothed signal | +0.019 | +0.068 | 0.019 | 0.129 | 1.191 |
| | filter the signal, keep the old scale | +0.034 | +0.047 | 0.034 | 0.057 | 1.057 |

**Conclusion.** The algebra holds: the third arm matches the first to three
decimals on dev in every configuration, and the clip binds on 4–5% of days under
the re-solved arm and on none under the shipped one. Re-solving does push the
realised ratio from about 1.06–1.09 to 1.19. But it cuts the maximin score by
half or more in all three configurations while roughly doubling turnover, so it
loses under the same rule everything else was chosen by.

This is 2.2 from the other side. The unpenalised Sharpe saturates near a ratio of
1.3, so a wider swing amplifies whatever is in the signal, noise included. On
valid, where the signal is strong, amplification pays; across dev's 69 blocks it
does not. **The unspent allowance is not free money, and the shipped order
stands.**

The gain sits entirely on valid, where the signal is strong, and the loss sits on
dev's 69 blocks. That is the shape of amplification, not of a better rule.
