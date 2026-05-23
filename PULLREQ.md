# Add probit regression with fixed-effect support

## Summary

This PR adds a `probit` estimator to Regress.jl. The estimator fits a binary-response probit model via Iteratively Reweighted Least Squares (IRLS) and supports high-dimensional fixed effects via `fe(...)` terms in the formula, consistent with the existing OLS/IV/FE API.

## New files

| File | Purpose |
|---|---|
| `src/BinaryModel.jl` | Structs (`BinaryResponse`, `BinaryPredictorQR`, `ILSEstimator`, `BinaryEstimator`) and their full StatsAPI implementations |
| `src/fit_probit.jl` | IRLS fitting loop, formula parsing helpers, probit log-likelihood, step-halving |
| `test/test_probit.jl` | End-to-end smoke test on RWM panel data |
| `test/probit_test/` | Self-contained validation suite comparing `Regress.probit` against R `fixest::feglm` (probit link): R script, Julia script, and automated coefficient/timing comparison |

## Changed files

| File | Change |
|---|---|
| `src/Regress.jl` | Include new files; import `Distributions`/`StatsFuns` at top level; export `probit`, `fit_probit`, `BinaryEstimator` |
| `src/fit.jl` | Add thin `probit(df, formula; kwargs...)` public wrapper |
| `src/utils/fit_common.jl` | Add `verbose::Bool` parameter to `partial_out_fixed_effects!` to suppress per-iteration `@info` logs during IRLS |

## Algorithm

1. **Formula parsing** — `fe(...)` terms are separated from slope terms; the remaining formula is used to build the model matrix `X` and response `y`.
2. **Initialisation** — `BinaryResponse` and `BinaryPredictorQR` objects are constructed; initial deviance is computed.
3. **IRLS loop** — each iteration:
   - Compute scores `g_i` and observed information `h_i` from the probit log-likelihood.
   - Form working response `z = η + g/h` and working weights `h`.
   - Partial out fixed effects from both `X` and `z` (calling `partial_out_fixed_effects!` with `verbose=false`).
   - Solve the resulting WLS problem with `ils_solver` (thin wrapper around the existing `fit_ols_core!`).
   - Recover fixed-effect coefficients via `solve_coefficients!` and update `η`.
   - Apply **step-halving** (up to 26 bisections) if deviance increases.
   - Check convergence: `|Δdeviance| / (0.1 + |deviance_new|) < tolerance`.
4. **Post-fit** — compute fitted probabilities `μ = Φ(η)`, degrees of freedom, and null deviance.

## StatsAPI surface (`BinaryEstimator`)

`coef`, `coefnames`, `coeftable`, `deviance`, `nulldeviance`, `loglikelihood`, `nullloglikelihood`, `nobs`, `dof`, `dof_residual`, `fitted`, `response`, `residuals`, `modelmatrix`, `responsename`, `r2` (McFadden pseudo-R²), `islinear` (→ `false`).

## Usage

```julia
using Regress

m = probit(df, @formula(y ~ x1 + x2 + fe(group));
           beta0 = zeros(2), max_iter = 100, tolerance = 1e-6)

coef(m)
fitted(m)
deviance(m)
r2(m)          # McFadden pseudo-R²
```

## Validation against R `fixest`

`test/probit_test/` contains a self-contained comparison between `fixest::feglm(..., family = binomial(link = "probit"))` and `Regress.probit`. Run with:

```bash
julia --project=. test/probit_test/test_probit_vs_fixest.jl
```

**Test result:**
```
Test Summary:                  | Pass  Total  Time
probit: Regress.jl vs R fixest |    8      8  8.5s
```

**Timing:**

| Estimator | Median seconds |
|---|---:|
| R fixest | 0.035 |
| Regress.jl | 0.2078 |

`Regress.jl / fixest = 5.94×` on this run.

**Coefficient comparison:**

| Coefficient | fixest | Regress.jl |
|---|---:|---:|
| hhninc | -2.028546e-6 | -2.028526e-6 |
| hhkids | -0.0300173 | -0.0300166 |
| educ | -0.0697462 | -0.0697465 |
| married | -0.0295630 | -0.0295618 |

Note: `fixest` drops `age` as collinear after absorbing fixed effects and removes 3,960 fixed effects / 11,014 observations due to pure 0/1 outcomes or singletons. Test compares the coefficient subset common to both estimators.

## Known limitations / TODOs

- `BinaryPredictorQR.X_reduced` is not yet populated with only linearly-independent columns (marked `FIXME`).
- QR factorisation storage in `BinaryPredictorQR` is stubbed out (`TODO`).
- `rr.mu` after convergence does not incorporate the fixed-effect sum `α` in the linear predictor (marked `FIXME` in `fit_probit.jl`).
- Only the probit link is implemented; logit and complementary log-log are natural extensions.

