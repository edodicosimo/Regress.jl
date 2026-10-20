# Changelog

## 0.2.0

### Breaking Changes

- **Requires CovarianceMatrices 0.33.** Regress uses its public extension
  functions for packages that form their own moment matrices, which earlier
  versions lack.

- **Requires Julia 1.12.** CovarianceMatrices 0.33 sets this floor, which rules
  out the 1.10 long-term-support release.

- **`vcov(model)` returns a `CovarianceMatrix`** instead of
  `Symmetric{T,Matrix{T}}`. It is still an `AbstractMatrix`, so indexing,
  `inv`, `\`, and factorizations are unchanged, but the result now carries the
  metadata of the estimate: `estimator(V)` gives the variance estimator,
  `bandwidth(V)` the selected HAC bandwidth, `kernelweights(V)` the per-column
  kernel weights, `information(V)` the full metadata `NamedTuple`, and
  `parent(V)` the bare matrix. Code that stores the result in a
  `Matrix`- or `Symmetric`-typed field needs updating; code that only reads
  values does not.

- **`weakivtest` takes `tol` instead of `eps`.** The old name shadowed
  `Base.eps` in the method body. `weakivtest(m; eps = ...)` now errors.

- **`IVMatrixEstimator` carries the estimator it was fitted with.** The type is
  now `IVMatrixEstimator{T, E, V, C}`, with a leading `estimator::E` field, so
  that leverage and display can dispatch on it. Code that writes the type with
  all its parameters, or constructs one positionally, needs updating;
  `M <: IVMatrixEstimator` bounds and field access by name are unaffected.

- **`LagTerm` is public rather than exported.** It names the type behind
  `lags(...)` and is not meant to be constructed directly, so `using Regress`
  no longer brings it into scope. Use `Regress.LagTerm` where you need the
  name; `lags` itself is unchanged.

- **Invalid input raises `ArgumentError`.** Validation failures previously
  threw bare strings, which `throw` wraps in `ErrorException`, or called
  `error`. This includes accessors on models fitted with `save = :minimal`,
  cluster variables not saved at fit time, `sargan` on an exactly identified
  model, and `weakivtest` with more than one endogenous regressor. Messages are
  unchanged, but `catch`/`@test_throws` clauses matching on the exception type
  need updating.

- **The matrix API rejects offset arrays.** `ols(X, y)` and
  `iv(::TSLS, Z, X, y)` index positionally after converting their arguments,
  so they now declare `Base.require_one_based_indexing`. `view`s and ordinary
  arrays are unaffected; an `OffsetArray` argument errors at the entry point
  instead of producing a misaligned fit.

- **`vcov(DriscollKraay(...), model)` takes no `type` keyword.** The
  small-sample correction is set on the estimator:
  `DriscollKraay(kernel; tis, iis, correction = :HC1)`. Passing `type` throws an
  `ArgumentError`.

- **`momentmatrix` is no longer available after `using Regress`.**
  CovarianceMatrices 0.33 no longer exports it, so the re-export does not
  provide it. Call `CovarianceMatrices.momentmatrix(model)`, or import it with
  `using CovarianceMatrices: momentmatrix`.

- **`CachedCR` with `CR2` or `CR3` throws an `ArgumentError`** on OLS and IV
  models. The cache holds no leverage adjustments, so these returned CR1-type
  results.

- **`CR2` and `CR3` on IV models fitted from matrices give the leverage-adjusted
  cluster variance of the formula fit**, for `vcov` and `aVar`. They returned CR0
  results.

- **Weighted `CR2` and `CR3` follow R `sandwich::vcovCL`** on OLS and IV
  models: each cluster's moments are `X_g'(I - H_gg)^p (w_g .* e_g)` with the
  weighted hat block `H_gg = X_g (X'WX)⁻¹ X_g' W_g`. They weighted the
  per-cluster leverage blocks twice, which drove `I - H_gg` indefinite and
  produced standard errors roughly an order of magnitude too large (and, for
  CR2, an `InexactError` on some data). A model fitted from matrices with a
  zero weight throws an `ArgumentError` for these estimators.

- **One finite-sample factor per estimator, the same on OLS and IV models fitted
  from a formula or from matrices.** The table is in the `vcov` docstring.
  Variances that change:
  - `CR2` and `CR3` no longer multiply the Bell–McCaffrey variance by
    `G/(G-1)·(n-1)/(n-K)`; they equal R `sandwich::vcovCL(type = "HC2"/"HC3")`.
  - `Cluster` is the raw sandwich on every model type. OLS models applied
    `n/(n-K)`.
  - HAC kernels, `EWC` and `VARHAC` apply `n/(n-K)` on IV models, as on OLS
    models (`dofadjust = false` leaves it out).
  - `CR0` on IV models fitted from matrices applies `(n-1)/(n-K)`, and `CR1`
    works there.
  - The `K` of `CR0` and `CR1` on IV models fitted from a formula counts every
    coefficient, so IV `CR1` equals fixest. It counted one fewer.
  - With absorbed fixed effects, the `K` of `CR0` and `CR1` counts a fixed
    effect not nested in the cluster by its levels, and no longer adds an
    intercept. `CR1` equals fixest with one or two fixed effects, on OLS and IV
    models.

- **`vcov`, `stderror` and `aVar` on Regress models throw an `ArgumentError`
  for an unknown keyword.** `vcov` and `stderror` take `dofadjust`, `demean`
  and `prewhite`; `aVar` takes `demean`, `prewhite` and `scale`. `vcov(HC1(),
  model; typo = 1)` was accepted on OLS models, and `vcov(k, model; scale =
  false)` returned a variance off by a factor of `n`.

- **`prewhite = true` with a cluster estimator throws an `ArgumentError`** on
  OLS models. With a single cluster dimension it was ignored.

- **The Kleibergen-Paap statistic follows the model's variance estimator.**
  `first_stage_F_KP(m + vcov(k))` is computed with `k` for `Uncorrelated`, `HC0`,
  `HC1`, `CR0`, `CR1` (any number of cluster dimensions) and the HAC kernels,
  with the finite-sample factor of `vcov` applied to the first-stage regression.
  For any other estimator `+` stores `NaN` and `first_stage_F_KP` throws an
  `ArgumentError`. The statistic was computed once with `HC1` and labeled with
  whatever estimator `+` set. Its robust meat also used the moments
  `Z̃ ⊗ (V − Z̃Π)` in place of `Z̃ ⊗ V`, with `V` the first-stage residuals: on
  `Sales ~ NDI + (Price ~ Pimin)` (Cigar) it was 244.16 against 2885.36. With one
  endogenous regressor the statistic now equals the first-stage Wald F
  (`first_stage_F_robust`, fixest `ivwald1`) under the same estimator.

- **`weakivtest` follows the model's variance estimator.** It supports the
  estimators of the Kleibergen-Paap statistic and throws an `ArgumentError` for
  any other; it treated HC2–HC5, CR2/CR3 and the rest as HC0. Its W uses the
  moment-matrix path and finite-sample factor of `vcov` applied to the
  first-stage regression, which for HC1 and CR1 is Stata `weakivtest`'s W. It
  uses every cluster dimension (only the first before) and resolves clusters
  named by column (`m + vcov(CR1(:State))` threw a `FieldError`). Under
  `Uncorrelated`, W is the iid Ω ⊗ I, so `F_eff = F_robust = F_nonrobust`. HC0
  and CR0 no longer use HC1's degrees of freedom. With absorbed fixed effects, Ω
  and W subtract the fixed-effect levels from the degrees of freedom. `F_robust`
  equals `first_stage_F_robust` and fixest `ivwald1` under the same estimator.

- **The matrix IV fit throws an `ArgumentError` when X̂'X̂ is singular.** It
  used a pseudo-inverse for the variance without a warning. Numerical fallbacks
  (the F statistic's `NaN`, the LIML eigenvalue regularization) apply only to
  `PosDefException`, `SingularException` and `LAPACKException`; any other error
  propagates.

### New Features

- **`lags()` formula term**: `@formula(y ~ lags(x, 12))` expands into a matrix of 12 lag columns. Supports nested transforms (`lags(log(abs(x)), 3)`), interactions (`lags(x, 3) & z`), and composition with other terms. Moved from LocalProjections.jl so both packages share the same implementation.

- **`first_stage_F_iid(m)`**, **`first_stage_F_robust(m)`**, **`first_stage_F_KP(m)`**: New first-stage diagnostic API returning `FirstStageFTest{T, K}` — parametric on value type (`Float64` or `Vector{Float64}`) and variance estimator type. Replaces the ambiguous `first_stage(m)` / `first_stage_f(m)` functions.

- **Kleibergen-Paap rk Wald F-statistic exposed**: The joint KP test statistic is now stored on the model and accessible via `first_stage_F_KP(m)`. Previously computed but discarded.

- **`AbstractTest` type hierarchy**: `AbstractTest` is the new abstract supertype for `FirstStageFTest{T, K}`, `WeakIVTestResult{T}`, `FirstStageResult`, `FirstStageIV`, `WuHausmanResult` and `SarganResult`. Those with a single well-defined p-value implement `StatsAPI.pvalue`; `SarganResult` also implements `dof`. `WeakIVTestResult` implements neither, since it reports against critical values.

- **Public model surface**: `OLSEstimator`, `IVEstimator`, `OLSMatrixEstimator`, `IVMatrixEstimator`, `AbstractIVEstimator`, `esample` and `partial_out` are marked `public`, so `Regress.OLSEstimator` and friends resolve without being exported.

- **`Homoskedastic` type**: Sentinel type used as the variance estimator parameter for IID F-tests.

- **NaN row filtering for `lags()`**: Rows with NaN values produced by `lags()` in the design matrix are automatically excluded from estimation in OLS, TSLS, and K-class models. The `esample` field correctly reflects these exclusions.

- **EWC, DriscollKraay, VARHAC support**: All `Correlated` variance estimators from CovarianceMatrices.jl now work with `model + vcov(...)` (previously only `HAC` was supported).

- **Driscoll-Kraay standard errors on fitted models**: `model + vcov(DriscollKraay(...))`
  now works for OLS and IV models and for the matrix estimators. Previously
  every such call was a `MethodError`. The computation delegates to
  CovarianceMatrices, which scales by the number of time periods rather than
  the number of observations.

- **`show` reports the selected HAC bandwidth**: a model fitted with an
  automatic bandwidth selector prints `Bartlett(auto: 55.71)` rather than
  `Bartlett(auto)`. The bandwidth is a property of the estimate, so it is read
  from `vcov(model)`.

- **`probit(df, formula)`**: binary-response probit by iteratively reweighted
  least squares, absorbing high-dimensional fixed effects declared with
  `fe(...)` terms. Returns a `BinaryEstimator` supporting `coef`, `deviance`,
  `nulldeviance`, `loglikelihood`, `fitted`, `residuals`, `dof`, and `r2` as
  McFadden's pseudo-R². Coefficients agree with GLM's `ProbitLink` to IRLS
  convergence tolerance.

  The estimator reports point estimates only: it computes no standard errors,
  so `stderror`, `vcov` and `coeftable` are not available for it.

### Bug Fixes

- `CR0` and `CR1` with a single cluster vector whose length differs from the
  number of observations throw a `DimensionMismatch` naming both lengths. On
  OLS models (`ols(X, y) + vcov(CR1(g))` and the formula equivalent) they read
  out of bounds and could crash the Julia process.

- The first-stage statistics no longer turn every exception into `NaN`. Only a
  failed factorization (`PosDefException`, `SingularException`,
  `LAPACKException`) gives `NaN`; other errors propagate.

- `first_stage_F_robust` under `CR0` and `CR1` counts the absorbed fixed effects
  in the cluster K, as `vcov` and the Kleibergen-Paap statistic do, and equals
  fixest `ivwald1`. On `Sales ~ NDI + (Price ~ Pimin + Pop) + fe(State)` (Cigar) it
  was 3.5% off under `CR1(:Year)`.

- `UniformSmoother` and `TriangularSmoother` work on every model type, with the
  `n/(n-K)` factor of the other long-run estimators. They threw a `MethodError`.

- `vcov(Uncorrelated(), m)` on OLS models is the iid variance `σ² · (X'X)⁻¹` with
  `σ² = RSS/dof_residual`, as on IV models; weighted fits use the weighted RSS, as
  R `vcov(lm(...))` does. `aVar(Uncorrelated(), m)` works on every model type and
  is the meat of that variance without the degrees-of-freedom factor. Both threw a
  `MethodError` on OLS models, and `aVar` also on IV models.

- `vcov` on Float32 models returns a Float32 matrix for every estimator.
  `HC1`–`HC5`, the cluster estimators and the HAC kernels returned Float64 on
  OLS or IV models.

- Clusters named by column work with every cluster estimator (`CR0`–`CR3`,
  `Cluster`), with keywords, and on `vcov`, `stderror`, `aVar` and `+`.
  `vcov(CR1(:State), model; dofadjust = false)` threw a `MethodError`, and
  `Cluster(:State)` was never looked up. On a model fitted from matrices, a
  column name throws an `ArgumentError` saying that the model stores no
  cluster data.

- On IV models, a keyword passed to `vcov` no longer changes the result. Any
  keyword sent the call to CovarianceMatrices' generic method, so
  `vcov(CR1(s), model; dofadjust = true)` gave standard errors 1.1% smaller
  than `vcov(CR1(s), model)` on the Cigar data.

- `dofadjust = false` on OLS models drops the `n/(n-K)` factor of the HAC,
  EWC, VARHAC and smoother estimators; it was ignored. On IV models it divides
  the `Uncorrelated()` variance by `n` instead of `n - K`. As in
  CovarianceMatrices, it does not change `HC1`–`HC5` or `CR0`–`CR3`, whose
  finite-sample factors define them.

- `adjr2` on formula-fitted OLS models counts the intercept once. It counted
  it twice, so `Sales ~ NDI + Price` on the Cigar data gave 0.207082 where R
  `lm` gives 0.207657. The `:McFadden` variant had the same error. `adjr2` now
  equals R `lm` and fixest `ar2`/`apr2` on OLS models and AER `ivreg` on IV
  models, formula and matrix, with and without an intercept. It also works on
  IV models fitted from matrices, which threw a `MethodError`.

- The model F-statistic of a formula-fitted IV model no longer includes the
  intercept in the Wald test when the intercept is implicit, as in
  `Sales ~ NDI + (Price ~ Pimin)`. It reported 10045.6 where the test of the
  slopes gives 160.36, at fit time and after `+ vcov(...)`, for TSLS and the
  k-class estimators.

- Formula IV fits with `double_precision = false` run, and their coefficients
  and default variance are `Float32`. TSLS threw a `MethodError` from
  `BLAS.syrk!` because the sweep operator's buffer was always `Float64`, and
  LIML and Fuller threw a `MethodError` because their default variance was
  promoted to `Float64`.

- `CachedCR(k, model)` with `CR0` or `CR1` works on every model type, and
  `vcov(CachedCR(k, model), model)` equals `vcov(k, model)` for one- and
  two-way clustering, by vector or, on formula fits, by column name. On
  matrix-fitted OLS and on formula IV models the cached result lacked the
  finite-sample factor (CR1 was 2.1% and 2.2% too small on the Cigar data, CR0
  0.07% too large or small). `CachedCR` on a matrix-fitted OLS model
  threw a method ambiguity, and on a formula IV model a `MethodError`; on a
  formula OLS model with a collinear regressor it threw a `DimensionMismatch`.
  On a formula OLS model the result is a `CovarianceMatrix`, as for `vcov(k,
  model)`.

- `vcov` with a `CachedCR` that CovarianceMatrices built from a model of
  another type (such as a GLM) throws an `ArgumentError` on Regress models.
  It was accepted on formula OLS models.

- `CR2` and `CR3` on OLS models give the leverage-adjusted cluster variance
  `Σ_g X_g'(I - H_gg)^p u_g u_g'(I - H_gg)^p X_g` for any row order and with
  several cluster dimensions. They were wrong when the rows of a cluster were
  not contiguous, which includes every two-way clustering.
  `CovarianceMatrices.momentresiduals` and
  `CovarianceMatrices.momentmatrix` return the weighted residuals and moments
  of a weighted OLS model.
- `CR2` and `CR3` on IV models throw an `ArgumentError` when `I - H_gg` is
  singular for a cluster, and no longer return `NaN` for an observation with a
  zero residual.
- On weighted IV models, every variance computed after the fit was wrong:
  `vcov(estimator, model)`, `model + vcov(estimator)`, and `stderror` with an
  estimator, for all estimators (HC, CR, HAC, `Uncorrelated`, Driscoll-Kraay)
  and for TSLS, LIML, Fuller and k-class fits. The moments combined the
  unweighted residuals with the √w-weighted regressors, so they carried √w
  where they need w. The error was not a scalar factor; on `Cigar.csv` weighted
  by `Pop`, HC1 came out about 1/4900 of the correct value. The variance
  computed at fit time was correct. Weighted IV variances now equal fixest's
  `feols(...; weights)` for HC1 and iid.

- **The intercept moment entered automatic HAC bandwidth selection on IV
  models.** With `Bartlett{NeweyWest}` or `Bartlett{Andrews}`, the OLS path
  takes its kernel weights from the model matrix, giving the constant column
  weight 0, while the IV path passed none and so counted every column of the
  moment matrix. An IV model instrumented by its own regressors selected a
  bandwidth differing from the equivalent OLS model by a factor of two. Both
  now exclude the intercept, as Stata does. Fixed-bandwidth HAC, and every
  other variance estimator, were unaffected.

- **Over-identified TSLS from matrix inputs used the wrong leverage.**
  `leverage(::IVMatrixEstimator)` computed `diag(X̂(X̂'X̂)⁻¹X̂')`, which equals
  the AER/sandwich IV leverage `diag(X·(X̂'X̂)⁻¹·X'·Z·(Z'Z)⁻¹·Z')` only when the
  model is just-identified. HC2/HC3 standard errors from `iv(TSLS(), Z, X, y)`
  therefore differed from those of the same model fitted through the formula
  interface. Both paths now agree; HC0/HC1 and just-identified models were
  never affected.

- **CR2 uses the symmetric square root of `I - H_gg`.** The Bell-McCaffrey
  adjustment is defined by the symmetric root; the Cholesky factor used
  previously preserves the quadratic form `u'(I - H_gg)⁻¹u` but differs from it
  by an orthogonal rotation, which does not cancel in the outer products the
  cluster meat sums. CR2 standard errors change by a fraction of a percent on
  typical data. CR3 is unaffected.

- **CR2/CR3 work on IV models**, with one or several cluster dimensions. Both
  threw `MethodError` when resolving cluster indices. The leverage blocks are
  built from the second-stage regressor matrix, which is what the IV sandwich
  treats as the design. Multiway clustering uses the inclusion–exclusion sum
  over cluster intersections, each term with its own leverage adjustment, as on
  OLS models. Every term uses the second-stage leverage; R `vcovCL` uses
  `hatvalues(ivreg)` for a term whose clusters are single rows, so two-way IV
  results differ from it slightly (relative difference at most 3e-5 on the Cigar data
  clustered by `State` and `Year`).

- **HC4/HC5 on the matrix IV estimator use the standard exponents.**
  `IVMatrixEstimator` raised `1 - h` to `δ` where the definition uses `δ/2`, and
  took an extra square root for HC5, so its adjustments disagreed with the ones
  the other three model types compute.

- **HAC bandwidth selection accounts for kernel weights.** The kernel weights
  that give the intercept column zero weight are now passed to `aVar` through
  the `weights` keyword. The moment matrix's intercept column is not constant,
  so an Andrews or Newey-West bandwidth computed without them differs — by a
  factor of about two on `y ~ x` in the bundled validation data.

- **CovarianceMatrices entry points accept Regress models.** `bread` and
  `leverage` were defined as new functions in Regress's namespace instead of
  extending the CovarianceMatrices generics, so `CovarianceMatrices.vcov(k, m)`
  threw a `MethodError` for every Regress model. `numobs` and `mask` were
  missing for the IV estimators. All are now provided, and the HC2-HC5 and
  CR2/CR3 estimators that dispatch on `leverage` work through the upstream path.

- **`aVar(k, model)` no longer masks collinear coefficients.** `aVar` estimates
  the variance of the moment matrix, where coefficient collinearity has not yet
  entered; rank deficiency is handled by `vcov`, which subsets to the
  full-rank block independently. The masking discarded the `CovarianceMatrix`
  wrapper without affecting any variance. `vcov` results are unchanged,
  including the NaN entries it produces for collinear coefficients.

- **HC3 first-stage F-statistic**: The robust first-stage F with HC3 (and HC2/HC4/HC5) was silently using the HC1 formula. Refactored to delegate to CovarianceMatrices.jl, which handles all variance types correctly.

- **`esample` with `lags()`**: `esample` now correctly marks NaN-filtered rows as `false`, so `sum(m.esample) == nobs(m)` and residuals can be mapped back to the original DataFrame via `res[m.esample] .= residuals(m)`.

- **`lags()` with nested transforms and other terms**: `@formula(y ~ lags(log(abs(x)), 3) + log(x))` previously crashed because `StatsModels.terms(::LagTerm)` returned a `FunctionTerm` instead of leaf terms.

- **`lags()` coefficient names for nested transforms**: `lags(log(abs(x)), 3)` produced names like `l_lag1` instead of `log(abs(x))_lag1`.

- **Fixest validation F-stat references**: Test reference values for `F_nonrobust` were Kleibergen-Paap statistics, not IID F-statistics. Corrected to match the actual SSR-based F-test.

- **`first_stage with vcov` test**: HC1 and HC3 first-stage robust F-statistics were identical due to the HC3 fallthrough bug.

- **Fallback path in `compute_per_endogenous_fstats`**: The fallback when `Xendo_orig` is `nothing` passed `Z_res` as the full design matrix, producing wrong F-statistics. Replaced with an explicit error.

### Refactoring

- **Residual adjustments delegate to CovarianceMatrices.jl**: the HC0 and HC2-HC5
  adjustments now come from `CovarianceMatrices.residual_adjustment` instead of
  being reimplemented for each of the four model types. The CR2/CR3 leverage
  adjustments are computed in Regress. HC1 stays local because upstream divides by
  `n - length(coef(m))`, which cannot see fixed effects absorbed out of the
  design, and the CR finite-sample corrections stay local to keep matching fixest.

- **First-stage robust F via CovarianceMatrices.jl**: Replaced ~250 lines of manual sandwich variance computation (`_compute_meat_inplace!`, `_compute_robust_first_stage_fstats_batched`, `_compute_single_first_stage_fstat`) with `_compute_first_stage_fstats_via_ols`, which constructs lightweight `OLSMatrixEstimator` wrappers around pre-computed first-stage data and delegates to `CovarianceMatrices.vcov`. No refitting; ZZ factorization is shared across endogenous variables.

- **`_filter_nan_rows` shared utility**: Extracted the NaN-filtering logic (previously triplicated in `fit_ols.jl`, `tsls.jl`, `kclass.jl`) into a single function in `fit_common.jl`.

- **`LagTerm` multi-column guard**: `modelcols(::LagTerm)` now throws `ArgumentError` for multi-column inner terms (interactions, categoricals) instead of producing silently wrong results.

- **LocalProjections.jl integration**: `first_stage` and `weakivtest` in LocalProjections.jl now extend the Regress.jl functions instead of defining separate ones.

- **Test import cleanup**: Replaced `using CovarianceMatrices: ...` with `using Regress: ...` in test files where CovarianceMatrices is not directly loadable. Eliminated `Base.==` method redefinition warnings in `test_formula.jl`.

### Documentation

- **`docs/src/iv_fstats.md`**: New document covering all IV diagnostics — IID F, robust Wald F, Kleibergen-Paap rank test, Montiel-Olea-Pflueger weak IV test, Wu-Hausman endogeneity test, and Sargan overidentification test — with exact formulas, computation steps, and API examples.