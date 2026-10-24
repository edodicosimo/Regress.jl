##############################################################################
##
## OLS Estimation
##
##############################################################################

"""
    _fit_ols_kernel(y, X, wts, fes; kwargs...) -> (fit, fe_solve)

OLS of `y` on `X` with weights `wts`, absorbing the fixed effects `fes`. Both
`ols(df, formula)` and `ols(X, y)` estimate through this function, so equal data and
settings give identical results.

`y` and `X` hold the estimation sample, unweighted. Both are modified in place: they are
demeaned on `fes` and scaled by `√w`, and `X` becomes the stored model matrix.
`factorization = :auto` means Cholesky for fewer than 100 columns and QR otherwise.
Failure of the demeaning to converge throws.

`fit` holds the estimates and their statistics. `fe_solve` holds the fixed-effect solver,
its iteration count and convergence, and copies of `y` and `X` before demeaning when
`save_fes`; the solver's type depends on `method`, so it is kept out of `fit`.
"""
function _fit_ols_kernel(y::Vector{T}, X::Matrix{T}, wts::AbstractWeights,
        fes::Vector{<:FixedEffect};
        response_name::Symbol,
        coef_names::Vector{String},
        has_intercept::Bool,
        has_fe_intercept::Bool,
        factorization::Symbol,
        collinearity::Symbol,
        collinearity_tol::Real,
        fe_tol::Real,
        fe_maxiter::Integer,
        method::Symbol,
        nthreads::Integer,
        progress_bar::Bool,
        save_fes::Bool,
        save_matrices::Bool,
        dof_add::Integer) where {T <: AbstractFloat}
    all(isfinite, wts) || throw(ArgumentError("Weights are not finite"))
    all(isfinite, y) ||
        throw(ArgumentError("Some observations for the dependent variable are not finite"))
    all(isfinite, X) ||
        throw(ArgumentError("Some observations for the exogenous variables are not finite"))

    rr = build_response(y, wts, response_name)
    has_any_intercept = has_intercept | has_fe_intercept
    tss_total = compute_tss_total(rr.y, wts, has_any_intercept)

    if isempty(fes)
        feM, iterations, converged, tss_partial, oldy, oldX = nothing, 0, true, tss_total,
        nothing, nothing
    else
        feM, iterations, converged, tss_partial, oldy, oldX = partial_out_fixed_effects!(
            _fe_columns(rr.y, X), vcat(string(response_name), coef_names), fes, wts,
            method, nthreads, fe_tol, fe_maxiter, progress_bar, save_fes,
            has_intercept, has_fe_intercept, T)
        converged || throw(ErrorException(
            "demeaning on the fixed effects did not converge in $iterations iterations; " *
            "increase fe_maxiter or fe_tol"))
    end

    if !(wts isa UnitWeights)
        sqrtw = sqrt.(wts)
        rr.y .*= sqrtw
        X .*= sqrtw
    end

    if factorization == :auto
        factorization = size(X, 2) < 100 ? :chol : :qr
    end
    pp, basis_coef,
    _ = fit_ols_core!(rr, X, factorization;
        tol = collinearity_tol, save_matrices, collinearity, has_intercept)

    fit = _ols_kernel_inference(rr, pp, basis_coef, fes;
        has_intercept, tss_total, tss_partial, dof_add, save_matrices)
    return fit, (; feM, iterations, converged, oldy, oldX)
end

# Sums of squares, degrees of freedom, R², HC1 vcov and Wald F of a solved OLS fit. A
# separate function so that it compiles for the concrete predictor type.
function _ols_kernel_inference(rr::OLSResponse{T}, pp::OLSLinearPredictor{T},
        basis_coef::BitVector, fes::Vector{<:FixedEffect};
        has_intercept::Bool, tss_total::T, tss_partial::T, dof_add::Integer,
        save_matrices::Bool) where {T}
    n = length(rr.y)
    rss = compute_rss(rr.y, rr.mu)
    ngroups_fes = Int[nunique(fe) for fe in fes]
    dof_fes = _fe_dof(fes)
    dof_model = sum(basis_coef)
    dof_residual = max(1, n - dof_model - dof_fes - dof_add)
    r2 = 1 - rss / tss_total
    r2_within = isempty(fes) ? r2 : 1 - rss / tss_partial

    # HC1 needs the model matrix, which `save = :minimal` discards; the vcov is then zero.
    vcov_estimator = CovarianceMatrices.HC1()
    k = length(basis_coef)
    vcov_matrix = save_matrices ?
                  compute_hc1_vcov_direct(pp.X, rr.y .- rr.mu, invchol(pp), basis_coef,
        n, dof_model, dof_fes, dof_residual) :
                  _wrap_vcov(Symmetric(zeros(T, k, k)), vcov_estimator, nothing)

    se = sqrt.(diag(vcov_matrix))
    coef_vec = copy(pp.beta)
    coef_vec[.!basis_coef] .= zero(T)
    t_stats = coef_vec ./ se
    p_values = 2 .* tdistccdf.(dof_residual, abs.(t_stats))
    F, p = compute_robust_fstat(coef_vec, vcov_matrix, has_intercept, dof_residual)

    if !save_matrices
        clear_response_data!(rr)
        clear_predictor_data!(pp)
    end

    return (; rr, pp, basis_coef, nobs = n, dof_model, dof_fes, ngroups_fes, dof_residual,
        tss_total, tss_partial, rss, r2 = T(r2), r2_within = T(r2_within),
        vcov_estimator, vcov_matrix, se, t_stats, p_values, F = T(F), p = T(p))
end

"""
    fit_ols(df, formula; kwargs...) -> OLSEstimator{T}

Fit a linear model by OLS from a table and a formula; see [`ols`](@ref) for the keywords.
"""
function fit_ols(@nospecialize(df),
        formula::FormulaTerm;
        contrasts::Dict = Dict{Symbol, Any}(),
        weights::Union{Symbol, Nothing} = nothing,
        save::Union{Bool, Symbol} = :residuals,
        save_cluster::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
        dof_add::Integer = 0,
        method::Symbol = :cpu,
        factorization::Symbol = :auto,
        collinearity::Symbol = :sweep,
        collinearity_tol::Real = 1e-8,
        nthreads::Integer = method == :cpu ? Threads.nthreads() : 256,
        double_precision::Bool = method == :cpu,
        fe_tol::Union{Real, Nothing} = nothing,
        fe_maxiter::Union{Integer, Nothing} = nothing,
        tol::Union{Real, Nothing} = nothing,
        maxiter::Union{Integer, Nothing} = nothing,
        drop_singletons::Bool = true,
        progress_bar::Bool = true,
        subset::Union{Nothing, AbstractVector} = nothing)
    fe_tol = _renamed_keyword(fe_tol, tol, :fe_tol, :tol, :ols, 1e-8)
    fe_maxiter = _renamed_keyword(fe_maxiter, maxiter, :fe_maxiter, :maxiter, :ols, 10000)

    save, save_residuals = validate_save_keyword(save)
    nthreads = validate_nthreads(method, nthreads)
    factorization in (:auto, :chol, :qr, :sweep) ||
        throw(ArgumentError("factorization must be :auto, :chol, :qr, or :sweep, got :$factorization"))
    collinearity in (:qr, :sweep) ||
        throw(ArgumentError("collinearity must be :qr or :sweep, got :$collinearity"))

    contrasts = convert(Dict{Symbol, Any}, contrasts)
    df = DataFrame(df; copycols = false)
    T = double_precision ? Float64 : Float32

    ##############################################################################
    ## Prepare Data
    ##############################################################################

    data_prep = prepare_data(df, formula, weights, subset, save, drop_singletons, nthreads)

    # Cluster variables are not kept in :minimal mode
    cluster_data = save == :minimal ? NamedTuple() :
                   extract_cluster_variables(df, data_prep.fe_vars, save_cluster,
        data_prep.esample)

    wts = data_prep.has_weights ?
          Weights(convert(Vector{T}, disallowmissing(view(df[!, weights], data_prep.esample)))) :
          uweights(data_prep.nobs)

    subdf = _create_subdf(df, data_prep.all_vars, data_prep.esample)
    subfes = isempty(data_prep.fes) ? FixedEffect[] :
             FixedEffect[fe[data_prep.esample] for fe in data_prep.fes]

    ##############################################################################
    ## Create Model Matrices
    ##############################################################################

    s = schema(data_prep.formula, subdf, contrasts)
    formula_schema = apply_schema(data_prep.formula, s, OLSEstimator, data_prep.has_fe_intercept)

    y_raw = response(formula_schema, subdf)
    y_raw isa AbstractVector || throw(ArgumentError(
        "the response variable must be numeric (got a matrix — " *
        "check that the dependent variable column is not a String or categorical type)"))
    y = convert(Vector{T}, y_raw)
    X = convert(Matrix{T}, modelmatrix(formula_schema, subdf))
    response_name, coef_names = coefnames(formula_schema)

    # Filter rows with NaN (e.g. from lags() producing NaN for out-of-bounds entries)
    y, (X,), wts, subfes, _,
    has_nan_rows, nan_rows = _filter_nan_rows(y, (X,), wts, subfes)

    return _fit_ols_formula(y, X, wts, subfes;
        data_prep, df, formula_schema, contrasts, cluster_data, save, has_nan_rows,
        nan_rows, response_name = Symbol(response_name),
        coef_names = String[string(name) for name in coef_names],
        factorization, collinearity, collinearity_tol, fe_tol, fe_maxiter, method,
        nthreads, progress_bar, dof_add)
end

# The formula fit from its model matrices onward, compiled for the concrete precision `T`.
function _fit_ols_formula(y::Vector{T}, X::Matrix{T}, wts, subfes;
        data_prep, df, formula_schema, contrasts, cluster_data, save, has_nan_rows,
        nan_rows, response_name, coef_names, factorization, collinearity,
        collinearity_tol, fe_tol, fe_maxiter, method, nthreads, progress_bar,
        dof_add) where {T}
    fit, fe_solve = _fit_ols_kernel(y, X, wts, subfes;
        response_name, coef_names,
        has_intercept = data_prep.has_intercept,
        has_fe_intercept = data_prep.has_fe_intercept,
        factorization, collinearity, collinearity_tol, fe_tol, fe_maxiter, method,
        nthreads, progress_bar,
        save_fes = data_prep.save_fes,
        save_matrices = save != :minimal,
        dof_add)

    # Fixed-effect estimates, from the residual of the undemeaned fit
    augmentdf = DataFrame()
    if data_prep.save_fes && fe_solve.oldy !== nothing
        r = fe_solve.oldX === nothing ? fe_solve.oldy :
            fe_solve.oldy - fe_solve.oldX[:, fit.basis_coef] * fit.pp.beta[fit.basis_coef]
        newfes, _, _ = solve_coefficients!(r, fe_solve.feM; tol = fe_tol,
            maxiter = fe_maxiter)
        for fekey in data_prep.fekeys
            augmentdf[!, fekey] = df[!, fekey]
        end
        for j in eachindex(subfes)
            augmentdf[!, data_prep.feids[j]] = Vector{Union{T, Missing}}(missing, data_prep.nrows)
            augmentdf[data_prep.esample, data_prep.feids[j]] = newfes[j]
        end
    end

    # FE grouping vectors for nesting detection (not kept in :minimal mode)
    fe_groups = save == :minimal || isempty(subfes) ? Vector{Int}[] :
                Vector{Int}[Vector{Int}(fe.refs) for fe in subfes]
    fes = OLSFixedEffects{T}(augmentdf, data_prep.fekeys, cluster_data, fe_groups,
        fit.dof_fes, fit.ngroups_fes, fe_solve.iterations, fe_solve.converged, method)

    esample = data_prep.esample isa Colon ? trues(data_prep.nrows) :
              BitVector(data_prep.esample)
    if has_nan_rows
        esample[findall(esample)[nan_rows]] .= false
    end

    return OLSEstimator(
        fit.rr, fit.pp, fes,
        data_prep.formula_origin, formula_schema, contrasts,
        esample, coef_names, fit.basis_coef,
        fit.nobs, fit.dof_model, fit.dof_fes, fit.dof_residual,
        fit.tss_total, fit.tss_partial, fit.rss, fit.r2, fit.r2_within,
        data_prep.has_intercept,
        fit.vcov_estimator, fit.vcov_matrix, fit.se, fit.t_stats, fit.p_values,
        fit.F, fit.p)
end
