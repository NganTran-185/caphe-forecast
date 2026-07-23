# Model Validation Findings

**Model under review:** GARCH(1,1) volatility forecast for arabica coffee
futures (`KC=F`)
**Data through:** 2026-07-21 (1,370 daily observations from 2021-01-04)
**Last run:** 2026-07-22

> Results are re-computed as new data arrives; figures below carry the run date.
> An earlier run (data through ~2026-06) produced a GARCH aggregate improvement
> of −32.1%; the current figure is −28.8%. See §5 on result drift.

---

## 1. Scope

Three questions:

1. **Are the model's assumptions satisfied?** (§2)
2. **Does it beat simpler alternatives — and under which conditions?** (§3)
3. **Where does it break?** (§4)

All forecast evaluation uses **walk-forward backtesting**: the model trains only
on data preceding each forecast window, then the cutoff rolls forward. No
look-ahead bias enters any prediction.

---

## 2. Assumption testing

| Test | Purpose | Statistic | p-value | Verdict |
|---|---|---|---|---|
| ADF — price level | Confirm prices are non-stationary | −1.696 | 0.4333 | Non-stationary, as expected |
| ADF — log returns | GARCH requires stationarity | −25.598 | ≈ 0 | ✅ Assumption holds |
| Engle's LM (10 lags) | Justify the GARCH model class | 104.52 | 6.8 × 10⁻¹⁸ | ✅ ARCH effects present |
| Ljung–Box, squared returns (10 lags) | Corroborate clustering | — | 4.9 × 10⁻²⁵ | ✅ Clustering confirmed |

**Conclusion.** Modelling returns rather than prices is statistically justified:
prices carry a unit root, returns emphatically do not. ARCH effects are present
at overwhelming significance under two independent tests, so a GARCH-class model
is an evidenced choice rather than an assumed one.

---

## 3. Benchmarking

### 3.1 Aggregate — all regimes pooled

54 forecast windows, 21 trading days each.

| Model | MAE | RMSE | vs. baseline |
|---|---|---|---|
| persistence (baseline) | 0.00491 | 0.00652 | — |
| historical 30-day | 0.00466 | 0.00624 | −4.3% |
| EWMA (λ = 0.94) | 0.00427 | 0.00572 | −12.3% |
| **GARCH(1,1)** | **0.00339** | **0.00464** | **−28.8%** |

A clean ordering: each added sophistication buys real accuracy, and GARCH leads.

### 3.2 Disaggregated — by volatility regime

Regimes are defined by the 63-day rolling volatility prevailing **at forecast
time** (25th/75th percentile cut-points: 0.0207 / 0.0244). Classification uses
only information available to the model; cut-points are full-sample quantiles
used for *labelling results*, never as a forecast input.

Window counts: calm 13 · normal 31 · turbulent 10.

| Model | calm | normal | turbulent |
|---|---|---|---|
| persistence (baseline), raw RMSE | 0.00848 | 0.00535 | 0.00688 |
| historical 30-day | −4.0% | −1.5% | −10.4% |
| EWMA (λ = 0.94) | −10.6% | −11.7% | −16.8% |
| **GARCH(1,1)** | **−21.5%** | **−31.8%** | **−39.1%** |

**Finding — the edge strengthens under stress.** The validation concern was that
GARCH might outperform only in easy conditions. The opposite holds: its
advantage rises monotonically from −21.5% (calm) to −39.1% (turbulent). The
model performs best where accurate volatility estimates matter most.

**Mechanism.** Note that persistence's raw error is *worst in calm periods*
(0.00848 vs 0.00688 turbulent). After a quiet stretch, volatility mean-reverts
upward, but persistence keeps projecting the low recent value and is caught out.
GARCH's long-run variance term (ω) is precisely the structure that anchors
against this — so the outperformance has an identifiable cause, not merely an
empirical one.

**Caveat — sample size.** The turbulent bucket contains only 10 forecast
windows. The monotonic pattern is suggestive, not conclusive, and should be
re-examined as more data accumulates.

### 3.3 Price-level models (context)

For completeness, the same walk-forward method applied to forecasting the price
*level* (earlier run):

| Model | RMSE | vs. baseline |
|---|---|---|
| naive (baseline) | 17.66 | — |
| drift | 19.86 | +12.5% worse |
| trend | 25.33 | +43.4% worse |

Nothing beats a naive forecast — consistent with weak-form efficiency. This is
the finding that motivated re-targeting the model at volatility rather than
level. _(Figures predate the current data window; pending re-run.)_

---

## 4. Structural breaks and limitations

### 4.1 Regime distribution

63-day rolling volatility, full sample:

| Percentile | Value |
|---|---|
| 25th | 0.02068 |
| 50th (median) | 0.02249 |
| 75th | 0.02441 |
| 90th | 0.02654 |

### 4.2 Identified episodes

- **Late 2021** — elevated volatility peaking 2021-10-13 at 0.03010, with
  2021-08-03 reaching 0.03135. Consistent with a supply-shock-driven regime.
  _(Causal attribution not yet verified against event records.)_
- **July 2026 — ongoing.** Rolling volatility reached **0.03389 on 2026-07-21,
  the 99.9th percentile of the entire sample.** Nine of the ten
  highest-volatility days on record fall within the preceding two weeks. Closes
  swung from 356.95 (07-09) to 321.30 (07-16) and back to 334.40 (07-20).
  _(Cause not yet identified; stated as a data finding only.)_

### 4.3 Principal limitation — out-of-distribution regimes

GARCH assumes a **stable long-run variance** toward which forecasts revert. The
§3.2 result shows this is not a weakness within *historically observed*
turbulence — the model excels there. The exposure is narrower and sharper:

> When volatility moves **outside the distribution the model was fitted on**,
> the mean-reversion anchor pulls forecasts back toward a historical normal the
> market has left, and the model will tend to **under-forecast**.

The current regime (99.9th percentile, above any sustained prior level) is
exactly such a period. **The backtest cannot speak to this**, because the sample
contains no comparable episode to evaluate against — the limitation is
structural and inferred from model form, not measured.

**Recommendation.** Track live forecast error through the current episode
(Phase 3 monitoring) to obtain direct evidence of behaviour under
out-of-distribution stress.

### 4.4 Other limitations

- Single instrument; no cross-asset or exogenous drivers.
- Fixed GARCH(1,1) specification; alternative orders and distributions
  (e.g. Student-t innovations, EGARCH for asymmetry) untested.
- EWMA λ fixed at the RiskMetrics 0.94 convention rather than fitted to this
  series.
- 21-day horizon only; performance at other horizons unexamined.

---

## 5. Result drift

Aggregate GARCH improvement moved from −32.1% to −28.8% between runs as the
July 2026 data entered the sample. This is expected — backtest results are a
function of the evaluation window — but it demonstrates why point-in-time
figures need date stamps and why periodic re-validation is necessary rather than
optional.

---

## 6. Overall assessment

| Dimension | Assessment |
|---|---|
| Assumptions | ✅ Tested and satisfied |
| Model class justification | ✅ Evidenced (ARCH effects, p < 10⁻¹⁷) |
| Benchmark performance | ✅ Beats three alternatives; edge widens under stress |
| Robustness across regimes | ✅ Within sample · ⚠️ Untested out-of-distribution |
| Monitoring | 🚧 Not yet implemented |

**Summary.** The GARCH(1,1) volatility model is appropriately specified,
statistically justified, and outperforms simpler benchmarks across all observed
regimes. The material open risk is behaviour in volatility regimes outside the
fitted distribution — currently unmeasured, and directly relevant given the
July 2026 environment. Ongoing monitoring is the recommended next control.

---

## Reproducing

```bash
python scripts/validate_assumptions.py   # §2
python scripts/regime_backtest.py        # §3
python scripts/check_vol_dates.py        # §4
```
