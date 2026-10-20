##############################################################################
##
## Main User-Facing API: ols() and iv()
##
##############################################################################

"""
    ols(df, formula; kwargs...) -> OLSEstimator

Estimate a linear model using Ordinary Least Squares (OLS).

Supports high-dimensional categorical variables (fixed effects) but not instrumental variables.
For IV models, use `iv(estimator, df, formula)`.

# Arguments
- `df`: a Table (e.g., DataFrame)
- `formula`: A formula created using `@formula(y ~ x1 + x2 + fe(group))`

# Keyword Arguments
- `contrasts::Dict = Dict()`: Contrast codings for categorical variables
- `weights::Union{Nothing, Symbol}`: Column name for weights
- `save::Symbol = :residuals`: Save residuals (`:residuals`), fixed effects (`:fe`), or both (`:all`)
- `save_cluster::Union{Symbol, Vector{Symbol}, Nothing}`: Cluster variables to save for post-estimation vcov
- `dof_add::Integer = 0`: Manual adjustment to degrees of freedom
- `method::Symbol = :cpu`: Computation method (`:cpu`, `:CUDA`, `:Metal`)
- `nthreads::Integer`: Number of threads (default: `Threads.nthreads()` for CPU)
- `double_precision::Bool = true` for CPU, `false` otherwise: Use Float64 vs Float32
- `tol::Real = 1e-6`: Tolerance for fixed effects demeaning
- `maxiter::Integer = 10000`: Maximum iterations for fixed effects
- `drop_singletons::Bool = true`: Drop singleton observations
- `progress_bar::Bool = true`: Show progress bar during estimation
- `subset::Union{Nothing, AbstractVector}`: Select specific rows

# Returns
- `OLSEstimator{T}`: Fitted model (T is Float64 or Float32 depending on `double_precision`)

# Examples
```julia
using DataFrames, RDatasets, Regress

df = dataset("plm", "Cigar")

# Simple OLS
model = ols(df, @formula(Sales ~ NDI + Pop))

# With fixed effects
model = ols(df, @formula(Sales ~ NDI + fe(State) + fe(Year)))

# Post-estimation robust standard errors
vcov(HC3(), model)
vcov(:State, :CR1, model)  # cluster-robust

# With weights
model = ols(df, @formula(Sales ~ NDI), weights = :Pop)
```

# Post-Estimation Variance Calculations

After fitting a model, you can compute different variance-covariance matrices without re-running the regression:

```julia
model = ols(df, @formula(y ~ x1 + x2 + fe(firm_id)))

# Different robust estimators
vcov(HC3(), model)           # Heteroskedasticity-robust (HC3)
vcov(HC1(), model)           # Default (HC1)

# Cluster-robust (using stored cluster variable from fe())
vcov(:firm_id, :CR1, model)

# Two-way clustering
vcov((:firm_id, :year), :CR1, model)

# HAC (time series)
vcov(Bartlett(5), model)

# Standard errors and coefficient table
stderror(HC3(), model)
coeftable(model, :firm_id, :CR1)
```

See also: [`iv`](@ref), [`OLSEstimator`](@ref)
"""
function ols(@nospecialize(df), formula::FormulaTerm; kwargs...)
    has_iv(formula) &&
        throw(ArgumentError("Formula contains instrumental variables. Use `iv(TSLS(), df, formula)` instead."))
    return fit_ols(df, formula; kwargs...)
end

##############################################################################
##
## Matrix-Based OLS API
##
##############################################################################

"""
    ols(X::AbstractMatrix, y::AbstractVector; kwargs...) -> OLSMatrixEstimator

Fit an OLS model from matrix inputs (no formula required).

This is a lightweight, efficient interface for cases where you have pre-computed
design matrices. Compatible with CovarianceMatrices.jl for robust inference.

# Arguments
- `X::AbstractMatrix`: Design matrix (n x k). Include an intercept column if desired.
- `y::AbstractVector`: Response vector (length n)

# Keyword Arguments
- `factorization::Symbol = :auto`: Factorization method
  - `:auto`: Cholesky for k < 100, QR for k >= 100
  - `:chol`: Faster Cholesky factorization
  - `:qr`: More numerically stable QR factorization
- `collinearity::Symbol = :qr`: Collinearity detection method (`:qr` or `:sweep`)
- `tol::Real = 1e-8`: Tolerance for collinearity detection
- `weights::Union{Nothing, AbstractVector} = nothing`: Observation weights
- `fe = nothing`: Fixed effects to absorb: one grouping vector of length n, or a tuple of
  them. `y` and the columns of `X` are demeaned on the groups, as `fe(...)` terms do in
  `ols(df, formula)`, and the absorbed levels enter the degrees of freedom and the
  cluster-robust small-sample correction. Rows are not dropped: a singleton group fits
  its observation exactly. The fixed effects absorb the intercept, so `X` must not
  contain an intercept column.
- `fe_tol::Real = 1e-8`, `fe_maxiter::Integer = 10000`: Tolerance and iteration limit of
  the demeaning; failure to converge throws an error
- `has_intercept::Bool = fe === nothing`: Whether model includes an intercept (for R²
  computation); `true` together with `fe` throws an `ArgumentError`
- `coefnames = nothing`: One name per column of `X`, returned by `coefnames(model)` and
  shown by `coeftable(model)`; the default is `["x1", ..., "xk"]`

# Returns
- `OLSMatrixEstimator{T, P, V, C}`: Fitted model with precomputed HC1 standard errors

# Example
```julia
using Regress, CovarianceMatrices

# Create design matrix with intercept
n = 100
X = hcat(ones(n), randn(n, 2))
beta_true = [1.0, 2.0, -1.5]
y = X * beta_true + 0.1 * randn(n)

# Fit model
model = ols(X, y)

# Extract results
coef(model)           # Coefficients
stderror(model)       # Standard errors (HC1)
r2(model)             # R-squared
residuals(model)      # Residuals
nobs(model)           # Number of observations

# Post-estimation robust standard errors
vcov(HC3(), model)    # HC3 vcov matrix
stderror(HC3(), model)  # HC3 standard errors

# Update model with different vcov
model_hc3 = model + vcov(HC3())
stderror(model_hc3)   # Now uses HC3

# Confidence intervals
confint(model)        # Using precomputed HC1
confint(HC3(), model) # Using HC3
```

# Collinearity Handling
Linearly dependent columns are automatically detected. Coefficients for
collinear columns are set to 0, and their standard errors to NaN.

```julia
# With collinear column
X = hcat(ones(100), randn(100, 2), randn(100, 2)[:, 1])  # Last column duplicated
model = ols(X, randn(100))
coef(model)  # One coefficient will be 0
```

See also: [`ols(df, formula)`](@ref), [`OLSMatrixEstimator`](@ref)
"""
function ols(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
        factorization::Symbol = :auto,
        collinearity::Symbol = :qr,
        tol::Real = 1e-8,
        weights::Union{Nothing, AbstractVector} = nothing,
        fe = nothing,
        fe_tol::Real = 1e-8,
        fe_maxiter::Integer = 10000,
        has_intercept::Bool = fe === nothing,
        coefnames::Union{Nothing, AbstractVector} = nothing)

    # Fit is 1-based: X and y are materialized into `Matrix`/`Vector` and indexed
    # positionally. Reject offset axes up front with a clear message rather than
    # failing deep inside the materialization.
    Base.require_one_based_indexing(X, y)

    # Validate inputs
    n, k = size(X)
    length(y) == n ||
        throw(DimensionMismatch("X has $n rows but y has $(length(y)) elements"))
    names = _matrix_coefnames(coefnames, k)
    fes = _matrix_fixedeffects(fe, n, has_intercept)
    has_fes = !isempty(fes)

    # Validate keywords
    factorization in (:auto, :chol, :qr) ||
        throw(ArgumentError("factorization must be :auto, :chol, or :qr, got :$factorization"))
    collinearity in (:qr, :sweep) ||
        throw(ArgumentError("collinearity must be :qr or :sweep, got :$collinearity"))

    # Determine numeric type
    T = promote_type(eltype(X), eltype(y))
    T <: AbstractFloat || (T = Float64)

    # Convert to Matrix{T} and Vector{T} (materializes views). Demeaning works in
    # place, so with fixed effects the inputs are always copied.
    X_mat = has_fes ? Matrix{T}(X) : convert(Matrix{T}, X)
    y_vec = has_fes ? Vector{T}(y) : convert(Vector{T}, y)
    y_tss = has_fes ? copy(y_vec) : y_vec  # the response before demeaning

    # Handle weights
    has_weights = weights !== nothing
    if has_weights
        length(weights) == n || throw(DimensionMismatch("weights must have length $n"))
        wts_vec = convert(Vector{T}, weights)
    else
        wts_vec = T[]
    end

    # Absorb the fixed effects with the weights, before the √w scaling, as `ols(df, formula)`
    if has_fes
        cols = AbstractVector{T}[y_vec, eachcol(X_mat)...]
        _absorb_matrix_fes!(cols, vcat("y", names), fes,
            has_weights ? Weights(wts_vec) : uweights(n), fe_tol, fe_maxiter, T)
    end

    if has_weights
        sqrtw = sqrt.(wts_vec)
        X_mat = X_mat .* sqrtw
        y_vec = y_vec .* sqrtw
        y_tss = y_tss .* sqrtw
    end

    # Choose factorization
    if factorization == :auto
        factorization = k < 100 ? :chol : :qr
    end

    # Build response object
    mu = similar(y_vec)
    rr = OLSResponse(y_vec, mu, wts_vec, T[], :y)

    # TSS for R²: centered with an intercept or absorbed fixed effects, else sum(y^2)
    tss = has_intercept | has_fes ? sum(abs2, y_tss .- mean(y_tss)) : sum(abs2, y_tss)

    # Fit using unified solver
    pp, basis_coef,
    _ = fit_ols_core!(rr, X_mat, factorization;
        tol = tol, save_matrices = true, collinearity = collinearity)

    # Compute RSS efficiently
    rss = compute_rss(rr.y, rr.mu)

    # Degrees of freedom
    fe_groups, fe_ngroups = _matrix_fe_groups(fes)
    dof_fes = sum(fe_ngroups; init = 0)
    dof_model = sum(basis_coef)
    dof_res = max(1, n - dof_model - dof_fes)

    # R-squared
    r2_val = 1 - rss / tss

    # Compute default HC1 vcov and statistics
    residuals_vcov = rr.y .- rr.mu
    invXX = invchol(pp)

    vcov_matrix = compute_hc1_vcov_direct(
        pp.X, residuals_vcov, invXX, basis_coef,
        n, dof_model, dof_fes, dof_res
    )

    # Standard errors
    se = sqrt.(diag(vcov_matrix))

    # t-statistics and p-values
    coef_vec = copy(pp.beta)
    coef_vec[.!basis_coef] .= zero(T)
    t_stats = coef_vec ./ se
    p_values = 2 .* tdistccdf.(dof_res, abs.(t_stats))

    # Default vcov estimator (HC1)
    default_vcov = CovarianceMatrices.HC1()

    return OLSMatrixEstimator{T, typeof(pp), typeof(default_vcov), typeof(vcov_matrix)}(
        rr, pp, basis_coef,
        n, dof_model, dof_res,
        T(rss), T(tss), T(r2_val), has_intercept, names, fe_groups, fe_ngroups,
        default_vcov, vcov_matrix, se, t_stats, p_values
    )
end

##############################################################################
##
## IV Function - Instrumental Variables
##
##############################################################################

"""
    iv(estimator::AbstractIVEstimator, df, formula; kwargs...) -> IVEstimator

Estimate an instrumental variables model using the specified estimator.

# Arguments
- `estimator::AbstractIVEstimator`: Estimator type (`TSLS()`, `LIML()`, `Fuller()`, `KClass()`)
- `df`: a Table (e.g., DataFrame)
- `formula`: A formula with IV syntax: `@formula(y ~ x + (endo ~ instrument))`

# Keyword Arguments
Same as `ols()`, plus:
- `first_stage::Bool = true`: Compute first-stage F-statistics

# Returns
- `IVEstimator{T}`: Fitted IV model (T is Float64 or Float32 depending on `double_precision`)

# Available Estimators
- `TSLS()`: Two-Stage Least Squares (k = 1)
- `LIML()`: Limited Information Maximum Likelihood (k = k_LIML)
- `Fuller(a)`: Fuller bias-corrected estimator (k = k_LIML - a/(n-L-p)), default a=1
- `KClass(k)`: Generic K-class estimator with user-specified k

# Examples
```julia
using DataFrames, RDatasets, Regress

df = dataset("plm", "Cigar")

# Two-stage least squares
model = iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))

# LIML (better with weak instruments)
model = iv(LIML(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))

# Fuller bias-corrected (approximately median-unbiased)
model = iv(Fuller(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))
model = iv(Fuller(4.0), df, @formula(Sales ~ NDI + (Price ~ Pimin)))  # minimize MSE

# Generic K-class
model = iv(KClass(0.5), df, @formula(Sales ~ NDI + (Price ~ Pimin)))

# With fixed effects
model = iv(TSLS(), df, @formula(Sales ~ (Price ~ Pimin) + fe(State)))

# Post-estimation
vcov(HC3(), model)
coeftable(model)
```

# Post-Estimation Variance Calculations

IV models support the same post-estimation vcov calculations as OLS:

```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ instrument)))

# Heteroskedasticity-robust
vcov(HC3(), model)

# Update model with different vcov
model_hc3 = model + vcov(HC3())

# Cluster-robust (save cluster variable when fitting)
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ inst)), save_cluster = :firm_id)
vcov(CR1(:firm_id), model)

# Two-way clustering
model = iv(TSLS(), df, @formula(...), save_cluster = [:firm_id, :year])
vcov(CR1(:firm_id, :year), model)
```

See also: [`ols`](@ref), [`TSLS`](@ref), [`LIML`](@ref), [`Fuller`](@ref), [`KClass`](@ref), [`IVEstimator`](@ref)
"""
# Check formula before dispatching to specific estimator
function _check_iv_formula(formula::FormulaTerm)
    !has_iv(formula) &&
        throw(ArgumentError("Formula does not contain instrumental variables. Use `ols(df, formula)` instead."))
end

function iv(::TSLS, @nospecialize(df), formula::FormulaTerm; kwargs...)
    _check_iv_formula(formula)
    return fit_tsls(df, formula; kwargs...)
end

function iv(::LIML, @nospecialize(df), formula::FormulaTerm; kwargs...)
    _check_iv_formula(formula)
    return fit_liml(df, formula; kwargs...)
end

function iv(estimator::Fuller, @nospecialize(df), formula::FormulaTerm; kwargs...)
    _check_iv_formula(formula)
    return fit_fuller(df, formula; a = estimator.a, kwargs...)
end

function iv(estimator::KClass, @nospecialize(df), formula::FormulaTerm; kwargs...)
    _check_iv_formula(formula)
    return fit_kclass(df, formula; kappa = estimator.kappa, kwargs...)
end

##############################################################################
##
## Matrix-based IV Function
##
##############################################################################

"""
    iv(estimator, Z::AbstractMatrix, X::AbstractMatrix, y::AbstractVector;
       n_endogenous::Int=1, fe=nothing, has_intercept::Bool=fe === nothing,
       coefnames=nothing) -> IVMatrixEstimator

Estimate an IV model with matrix inputs, by Two-Stage Least Squares (`TSLS()`) or a
k-class estimator (`LIML()`, `Fuller(a)`, `KClass(kappa)`).

# Arguments
- `estimator`: `TSLS()`, `LIML()`, `Fuller(a)` or `KClass(kappa)`
- `Z::AbstractMatrix`: Instrument matrix [Xexo, Zinstr] where Zinstr are excluded
  instruments. The k-class estimators require its first columns to equal those of `X`
  and throw an `ArgumentError` otherwise; TSLS projects on all of `Z`.
- `X::AbstractMatrix`: Regressor matrix, typically [Xexo, Xendo] where Xendo are endogenous variables
- `y::AbstractVector`: Response vector

# Keyword Arguments
- `n_endogenous::Int=1`: Number of endogenous variables (last n_endogenous columns of X)
- `fe=nothing`, `fe_tol=1e-8`, `fe_maxiter=10000`: Fixed effects to absorb, as in
  [`ols(X, y)`](@ref); `y` and the columns of `X` and `Z` are demeaned on the groups.
  `first_stage` and `weakivtest` count the absorbed levels.
- `has_intercept::Bool=fe === nothing`: Whether the model includes an intercept (for R²
  computation); `true` together with `fe` throws an `ArgumentError`
- `coefnames=nothing`: One name per column of `X`, returned by `coefnames(model)` and
  shown by `coeftable(model)`; the default is `["x1", ..., "xk"]`. `first_stage(model)`
  names the endogenous variables and the exogenous regressors from it.

# Returns
- `IVMatrixEstimator{T}`: Fitted IV model supporting `coef()`, `vcov()`, `model + vcov()`, etc.

# Details

The TSLS estimator is computed as:
```math
\\hat{\\beta}_{TSLS} = (X'P_Z X)^{-1} X'P_Z y
```
where ``P_Z = Z(Z'Z)^{-1}Z'`` is the projection matrix onto the instrument space.

A k-class estimator solves
```math
\\hat{\\beta}_\\kappa = (X'X - \\kappa X'M_Z X)^{-1} (X'y - \\kappa X'M_Z y)
```
with ``M_Z = I - P_Z``; ``\\kappa`` is the LIML eigenvalue for `LIML()`, that eigenvalue
minus ``a/(n - L - p)`` for `Fuller(a)` (``L`` excluded instruments, ``p`` exogenous
regressors), and `kappa` for `KClass(kappa)`, as in the formula fits. The sandwich
variance replaces ``\\hat X`` by ``X - \\kappa M_Z X`` and ``(\\hat X'\\hat X)^{-1}`` by
``(X'X - \\kappa X'M_Z X)^{-1}``. `first_stage` and `weakivtest` depend only on the data
and the variance estimator, not on the second-stage estimator.

# Example
```julia
using Regress

# Generate data
n = 200
z = randn(n)           # Instrument
u = randn(n)           # Error
x = 0.5*z + 0.5*u      # Endogenous (correlated with u)
y = 1.0 .+ 2.0*x .+ u  # True coefficient = 2.0

# Build matrices (no intercept case for simplicity)
Z = hcat(ones(n), z)   # [constant, instrument]
X = hcat(ones(n), x)   # [constant, endogenous]

model = iv(TSLS(), Z, X, y; has_intercept=true, n_endogenous=1)
coef(model)  # Should be approximately [1.0, 2.0]

# Robust standard errors
using CovarianceMatrices
vcov(HC1(), model)
model_robust = model + vcov(HC1())
```

See also: [`iv(::TSLS, df, formula)`](@ref), [`IVMatrixEstimator`](@ref)
"""
function iv(estimator::AbstractIVEstimator, Z::AbstractMatrix{<:Real},
        X::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real};
        n_endogenous::Int = 1,
        fe = nothing,
        fe_tol::Real = 1e-8,
        fe_maxiter::Integer = 10000,
        has_intercept::Bool = fe === nothing,
        coefnames::Union{Nothing, AbstractVector} = nothing)

    # Fit is 1-based: Z, X, y are materialized into `Matrix`/`Vector` and indexed
    # positionally. Reject offset axes up front with a clear message rather than
    # failing deep inside the materialization.
    Base.require_one_based_indexing(Z, X, y)

    # Validate inputs
    n = length(y)
    size(X, 1) == n ||
        throw(DimensionMismatch("X has $(size(X,1)) rows but y has $n elements"))
    size(Z, 1) == n ||
        throw(DimensionMismatch("Z has $(size(Z,1)) rows but y has $n elements"))
    n_endogenous >= 0 || throw(ArgumentError("n_endogenous must be non-negative"))
    n_endogenous <= size(X, 2) ||
        throw(ArgumentError("n_endogenous ($n_endogenous) > number of X columns ($(size(X,2)))"))
    names = _matrix_coefnames(coefnames, size(X, 2))
    fes = _matrix_fixedeffects(fe, n, has_intercept)
    has_fes = !isempty(fes)

    # Check order condition
    k_x = size(X, 2)
    k_z = size(Z, 2)
    k_exo = k_x - n_endogenous
    k_instr = k_z - k_exo  # Excluded instruments
    k_instr >= n_endogenous || throw(ArgumentError(
        "Not enough instruments: $k_instr excluded instruments < $n_endogenous endogenous variables"))
    _check_matrix_iv_layout(estimator, Z, X, k_exo)

    # Determine numeric type
    T = promote_type(eltype(X), eltype(Z), eltype(y))
    T <: AbstractFloat || (T = Float64)

    # Convert to concrete types. Demeaning works in place, so with fixed effects the
    # inputs are always copied.
    X_mat = has_fes ? Matrix{T}(X) : convert(Matrix{T}, X)
    Z_mat = has_fes ? Matrix{T}(Z) : convert(Matrix{T}, Z)
    y_vec = has_fes ? Vector{T}(y) : convert(Vector{T}, y)
    y_tss = has_fes ? copy(y_vec) : y_vec  # the response before demeaning

    if has_fes
        cols = AbstractVector{T}[y_vec, eachcol(X_mat)..., eachcol(Z_mat)...]
        colnames = vcat("y", names, ["z$i" for i in axes(Z_mat, 2)])
        _absorb_matrix_fes!(cols, colnames, fes, uweights(n), fe_tol, fe_maxiter, T)
    end

    beta, residuals, X_hat, invXhatXhat = _matrix_iv_fit(
        estimator, Z_mat, X_mat, y_vec, n_endogenous)

    # Degrees of freedom
    k = size(X_mat, 2)
    basis_coef = trues(k)  # No collinearity detection for simplicity
    fe_groups, fe_ngroups = _matrix_fe_groups(fes)
    dof_fes = sum(fe_ngroups; init = 0)
    dof_model = sum(basis_coef)
    dof_res = max(1, n - dof_model - dof_fes)

    # Sum of squares; TSS is centered with an intercept or absorbed fixed effects
    rss = sum(abs2, residuals)
    tss = has_intercept | has_fes ? sum(abs2, y_tss .- mean(y_tss)) : sum(abs2, y_tss)
    r2_val = 1 - rss / tss

    # Post-estimation data for vcov
    postestimation = PostEstimationDataIVMatrix{T}(
        X_mat, Z_mat, X_hat, y_vec, residuals, Matrix{T}(invXhatXhat), n_endogenous
    )

    # Default vcov (HC1)
    default_vcov = CovarianceMatrices.HC1()

    # Compute HC1 variance-covariance matrix
    # Sandwich: V = (n/dof_res) * (X̂'X̂)^{-1} * X̂'diag(e²)X̂ * (X̂'X̂)^{-1},
    # with dof_res = n - k - (absorbed fixed-effect levels)
    scale_hc1 = T(n) / T(dof_res)
    Xe = X_hat .* residuals
    # Form the sandwich as B*B' (B = (X̂'X̂)⁻¹ X̂'diag(e)) so it is PSD even when X̂'X̂ is nearly
    # singular (weak first stage): A*M*A can then have negative diagonal entries from cancellation.
    B_hc1 = invXhatXhat * Xe'
    vcov_matrix = _wrap_vcov(Symmetric(scale_hc1 .* (B_hc1 * B_hc1')), default_vcov, nothing)

    # Standard errors, t-stats, p-values
    se = sqrt.(diag(vcov_matrix))
    t_stats = beta ./ se
    p_values = 2 .* tdistccdf.(dof_res, abs.(t_stats))

    return IVMatrixEstimator{T, typeof(estimator), typeof(default_vcov),
        typeof(vcov_matrix)}(
        estimator,
        beta,
        postestimation,
        basis_coef,
        n,
        dof_model,
        dof_res,
        T(rss),
        T(tss),
        T(r2_val),
        has_intercept,
        names,
        fe_groups,
        fe_ngroups,
        default_vcov,
        vcov_matrix,
        se,
        t_stats,
        p_values
    )
end

# The k-class fit separates the exogenous regressors from the excluded instruments, so
# it needs Z = [Xexo, Zinstr] with the same leading columns as X. TSLS projects on all
# of Z and does not.
_check_matrix_iv_layout(::TSLS, Z, X, k_exo) = nothing
function _check_matrix_iv_layout(estimator::AbstractIVEstimator, Z, X, k_exo)
    view(Z, :, 1:k_exo) == view(X, :, 1:k_exo) || throw(ArgumentError(
        "$(_estimator_name(estimator)) needs Z = [Xexo, Zinstr] and X = [Xexo, Xendo] " *
        "with the same first $k_exo (exogenous) columns"))
    return nothing
end

# The coefficients, residuals, second-stage regressors and bread of a matrix IV fit.
# The sandwich variance is `bread * X̂'diag(e²)X̂ * bread`, with `X̂` the projected
# regressors for TSLS and the k-class adjustment matrix `X - κ·resid(X | Z)` otherwise.
function _matrix_iv_fit(::TSLS, Z_mat, X_mat, y_vec, n_endogenous)
    # Projected regressors X̂ = P_Z X, from an orthonormal basis of the columns of Z
    FZ = _full_rank_qr(Z_mat, "Instrument matrix Z is rank deficient or nearly singular")
    QZ = Matrix(FZ.Q)
    X_hat = QZ * (QZ' * X_mat)

    # TSLS estimator: β = (X̂'X)⁻¹X̂'y = (X̂'X̂)⁻¹X̂'y, since X̂'X = X'P_Z X = X̂'X̂.
    # With X̂[:, p] = QR, β[p] = R⁻¹Q'y and (X̂'X̂)⁻¹[p, p] = R⁻¹R⁻ᵀ.
    FX = _full_rank_qr(X_hat, "X̂'X is singular - check for perfect collinearity")
    Rinv = inv(UpperTriangular(FX.R))
    p = FX.p
    beta = similar(y_vec, size(X_mat, 2))
    beta[p] = Rinv * (Matrix(FX.Q)' * y_vec)
    invXhatXhat = similar(Rinv, size(Rinv))
    invXhatXhat[p, p] = Rinv * Rinv'

    residuals = y_vec - X_mat * beta
    return beta, residuals, X_hat, Symmetric(invXhatXhat)
end

# Pivoted QR of `A`; throws `ArgumentError(msg)` when `A` has numerical rank below its
# column count, that is, when a diagonal entry of R is at most `tol` times the largest,
# the rule `detect_collinearity_qr` applies. A tolerance, rather than a failed Cholesky
# factorization of `A'A`, decides singularity: an exactly collinear `A` can leave a
# small positive pivot after rounding, depending on the BLAS kernel.
function _full_rank_qr(A, msg; tol = 1e-8)
    F = qr(A, ColumnNorm())
    R_diag = abs.(diag(F.R))
    threshold = isempty(R_diag) ? zero(eltype(R_diag)) : maximum(R_diag) * tol
    count(>(threshold), R_diag) == size(A, 2) || throw(ArgumentError(msg))
    return F
end

function _matrix_iv_fit(estimator::Union{LIML, Fuller, KClass}, Z_mat, X_mat, y_vec,
        n_endogenous)
    k_exo = size(X_mat, 2) - n_endogenous
    Xexo = X_mat[:, 1:k_exo]
    Xendo = X_mat[:, (k_exo + 1):end]
    Zinstr = Z_mat[:, (k_exo + 1):end]

    kappa = _kclass_kappa(estimator, y_vec, Xendo, Zinstr, Xexo)
    coef, residuals, invA, Adj = _kclass_fit(y_vec, Xendo, Zinstr, Xexo, kappa)

    # `_kclass_fit` orders the coefficients [endogenous; exogenous]; X is [Xexo, Xendo].
    order = vcat((n_endogenous + 1):(n_endogenous + k_exo), 1:n_endogenous)
    return coef[order], residuals, Adj[:, order], Symmetric(invA[order, order])
end

"""
    probit(df, formula; kwargs...) -> BinaryEstimator

Estimate a Probit model, optionally with fixed effects.
"""
function probit(@nospecialize(df), formula::FormulaTerm;
        beta0 = nothing, max_iter = 100, tolerance = 1e-6)
    return fit_probit(df, formula, beta0, max_iter, tolerance)
end
