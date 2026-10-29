#######################################
### HELPER FUNCTION TO CLEAN DATA
#######################################
"""
    select_columns(df::DataFrame, formula::FormulaTerm)
        -> (schema, formula_schema, data, X_without_fe, y)

Build model columns and formula metadata from `df`. Return the schema and
schema-applied formula after converting standalone `fe(...)` terms to ordinary
terms, a DataFrame reconstructed from the response and model columns, the
`Matrix{Float64}` design excluding standalone fixed-effect terms, and the
response vector. The returned DataFrame is not a direct subset of `df`.
"""
function select_columns(df::DataFrame, formula::FormulaTerm)
    formula_without_fe = remove_fixedeffects(formula)
    formula = ignore_fe(formula)

    y = modelcols(formula.lhs, df)
    X = modelcols(MatrixTerm(formula.rhs), df)

    X_without_fe = modelcols(MatrixTerm(formula_without_fe.rhs), df)

    schema = StatsModels.schema(formula, df)
    formula_schema = apply_schema(formula, schema)

    response_name = coefnames(formula_schema.lhs)
    Xnames = coefnames(formula_schema.rhs)

    out = DataFrame()

    out[!, response_name] = vec(y)
    for (j, name) in enumerate(Xnames)
        out[!, name] = X[:, j]
    end
    X_final = convert(Matrix{Float64}, X_without_fe)
    return (schema, formula_schema, out, X_final, vec(y))
end

"""
    ignore_fe(f::FormulaTerm) -> FormulaTerm

Return a formula where each `fe(x)` term is converted to the ordinary term `x`.
"""
function ignore_fe(f::FormulaTerm)
    rhs_terms = if hasproperty(f.rhs, :terms)
        f.rhs.terms
    elseif f.rhs isa Tuple
        f.rhs
    else
        (f.rhs,)
    end

    new_rhs = map(rhs_terms) do t
        if t isa FunctionTerm{typeof(fe)}
            term(Symbol(t.args[1]))
        else
            t
        end
    end

    return FormulaTerm(f.lhs, Tuple(new_rhs))
end

"""
    remove_fixedeffects(f::FormulaTerm) -> FormulaTerm

Return a formula with all `fe(...)` terms removed from the right-hand side.
"""
function remove_fixedeffects(f::FormulaTerm)
    rhs_terms = if hasproperty(f.rhs, :terms)
        f.rhs.terms
    elseif f.rhs isa Tuple
        f.rhs
    else
        (f.rhs,)
    end

    new_rhs = filter(rhs_terms) do t
        !(t isa FunctionTerm{typeof(fe)})
    end

    return FormulaTerm(f.lhs, Tuple(new_rhs))
end

"""
    get_coefficient_names_nofe(formula::FormulaTerm, data::DataFrame)

Return the response name and coefficient names after excluding fixed-effect
terms from `formula`.
"""
function get_coefficient_names_nofe(formula::FormulaTerm, data::DataFrame)
    formula = remove_fixedeffects(formula)
    schema = StatsModels.schema(formula, data)
    formula_schema = apply_schema(formula, schema)
    response_name, coef_names = coefnames(formula_schema.lhs), coefnames(formula_schema.rhs)
    coef_names_str = String[string(name) for name in coef_names]
    return (Symbol(response_name), coef_names_str)
end

############################################################
### IRLS workspace for Fisher Scoring
############################################################
"""
    GLMWorkspace(X::Matrix{T})

Allocate buffers for Bernoulli Fisher scoring with `n` observations and `k`
design columns. Buffers are uninitialized until filled by the fitting loop.

- `g`: response residuals `y - mu`.
- `d`: inverse-link derivatives `dmu_deta`.
- `h`: Fisher weights `d^2 / (mu * one_minus_mu)`.
- `z`: working response `eta + g / d`.
- `M`: shared `n × (k + 1)` absorption buffer containing `[z X]`.
- `cols`: views of the columns of `M` for the fixed-effect solver.
- `resid`: temporary storage for recovering fixed-effect contributions;
  unlike `rr.resid`, this is not the response residual vector.

All vector buffers have length `n`. Arithmetic in the formulas is elementwise.
"""
struct GLMWorkspace{T <: AbstractFloat, C <: AbstractVector{T}}
    g::Vector{T}       # response residual y - mu
    d::Vector{T}       # inverse-link derivative dmu_deta
    h::Vector{T}       # Fisher weights dmu_deta^2 / Var(y)
    z::Vector{T}       # working response η + g/d
    M::Matrix{T}       # [working response | design], absorbed in place
    cols::Vector{C}    # views into M's columns for solve_residuals!
    resid::Vector{T}   # z - Xβ, input to the fixed-effect contribution recovery
end

function GLMWorkspace(X::Matrix{T}) where {T <: AbstractFloat}
    n, k = size(X)
    M = Matrix{T}(undef, n, k + 1)
    C = typeof(view(M, :, 1))
    cols = C[view(M, :, j) for j in 1:(k + 1)]
    return GLMWorkspace{T, C}(
        Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n),
        M, cols, Vector{T}(undef, n))
end

############################################################
### IRLS workspace
############################################################
"""
    ProbitWorkspace{T, C}

Legacy workspace for score/observed-information updates, with working response
`eta + g / h`. The current `probit` fitting path uses `GLMWorkspace`
instead. Buffers are allocated once and are initially uninitialized.

The working response and design columns to be absorbed share a single
`(n × (k+1))` buffer `M`: column 1 holds the working response, columns `2:k+1`
hold the design. `cols` is a concretely typed vector of views into `M`, so
`solve_residuals!` demeans them without dynamic dispatch.
"""
struct ProbitWorkspace{T <: AbstractFloat, C <: AbstractVector{T}}
    g::Vector{T}       # per-observation score
    h::Vector{T}       # per-observation observed information (working weights)
    z::Vector{T}       # working response η + g/h
    M::Matrix{T}       # [working response | design], absorbed in place
    cols::Vector{C}    # views into M's columns for solve_residuals!
    resid::Vector{T}   # z - Xβ, input to the fixed-effect contribution recovery
end

function ProbitWorkspace(X::Matrix{T}) where {T <: AbstractFloat}
    n, k = size(X)
    M = Matrix{T}(undef, n, k + 1)
    C = typeof(view(M, :, 1))
    cols = C[view(M, :, j) for j in 1:(k + 1)]
    return ProbitWorkspace{T, C}(
        Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n),
        M, cols, Vector{T}(undef, n))
end

# Views onto the working response and the absorbed design within the shared buffer.
zbuf(ws::Union{ProbitWorkspace, GLMWorkspace}) = view(ws.M, :, 1)
Xdbuf(ws::Union{ProbitWorkspace, GLMWorkspace}) = view(ws.M, :, 2:size(ws.M, 2))

############################################################
### Weighted least squares on the absorbed design
############################################################
"""
    wls_absorbed!(beta, Xd, z, h, basis) -> beta

Solve `min_β Σ hᵢ (zᵢ - Xdᵢ'β)²` and write the coefficients into `beta`.
`Xd` and `z` must already have fixed effects absorbed, if applicable.
`basis` selects retained columns; excluded coefficients are set to zero.

Form the weighted normal equations and solve them with Cholesky factorization.
The retained system must be positive definite. This function allocates temporary
weighted arrays and normal-equation buffers; it does not perform absorption.
"""
function wls_absorbed!(beta::Vector{T}, Xd::AbstractMatrix{T}, z::AbstractVector{T},
        h::Vector{T}, basis::BitVector) where {T <: AbstractFloat}
    hz = h .* z
    hX = h .* Xd
    XtWX = Xd' * hX
    XtWz = Xd' * hz
    # FE-collinear columns (basis[j] == false) are excluded from the solve: pin
    # their row/column to a unit diagonal with a zero right-hand side so their
    # coefficient comes back as zero.
    @inbounds for j in axes(XtWX, 1)
        if !basis[j]
            XtWX[:, j] .= zero(T)
            XtWX[j, :] .= zero(T)
            XtWX[j, j] = one(T)
            XtWz[j] = zero(T)
        end
    end
    copyto!(beta, cholesky!(Symmetric(XtWX)) \ XtWz)
    return beta
end

"""
    _glm_irls!(m, ws, feM, has_fes, max_iter, tolerance) -> basis

Run Bernoulli Fisher scoring in place using `GLMWorkspace` and the optional
fixed-effect solver `feM`. At each iteration, form the working response
`z = eta + (y - mu) / dmu_deta` and weights
`h = dmu_deta^2 / (mu * one_minus_mu)`, elementwise.

Absorb fixed effects when present, solve weighted least squares, recover the
fixed-effect contribution, and update the response with step-halving.
Stop when `abs(Dnew - Dold) / (0.1 + abs(Dnew)) < tolerance` or after
`max_iter` iterations. Reaching the iteration limit does not signal an error.

Return a Boolean mask of retained design columns. Columns nearly annihilated
by fixed-effect absorption are detected on the first iteration and excluded
thereafter; without fixed effects the mask contains only `true` values.
The separate function permits specialization on workspace and solver types.
"""
function _glm_irls!(m::BinaryEstimator, ws::GLMWorkspace, feM, has_fes::Bool,
        max_iter::Integer, tolerance::Real)
    rr = m.rr
    pp = m.pp
    n = length(rr.y)
    k = size(pp.X, 2)
    # Basis of slope columns that survive absorption. A column collinear with the
    # fixed effects collapses to (near) zero after demeaning; it is detected once
    # on the first iteration and excluded thereafter.
    basis = trues(k)
    basis_detected = !has_fes
    alpha = zeros(n)
    sspre = zeros(k)
    zd = zbuf(ws)
    Xd = Xdbuf(ws)

    for _ in 1:max_iter
        inverse_link!(rr)
        rr.resid .= rr.y .- rr.mu
        ws.g .= rr.resid
        ws.d .= rr.dmu_deta
        ws.h .= ws.d .^ 2 ./ (rr.mu .* rr.one_minus_mu)

        # Working response, and the absorption buffer holding [z | X].
        ws.z .= rr.eta .+ ws.g ./ ws.d
        zd .= ws.z
        copyto!(Xd, pp.X)

        if has_fes
            if !basis_detected
                @inbounds for j in 1:k
                    sspre[j] = sum(abs2, view(Xd, :, j))
                end
            end
            FixedEffects.update_weights!(feM, Weights(ws.h))
            solve_residuals!(
                ws.cols, feM; tol = 1e-6, maxiter = 10000, progress_bar = false)
            if !basis_detected
                @inbounds for j in 1:k
                    if sum(abs2, view(Xd, :, j)) < 1e-6 * sspre[j]
                        basis[j] = false
                    end
                end
                basis_detected = true
            end
            # Zero any FE-collinear columns so the WLS coefficient is exactly zero.
            @inbounds for j in 1:k
                basis[j] || (view(Xd, :, j) .= zero(eltype(Xd)))
            end
        end

        wls_absorbed!(pp.beta_new, Xd, zd, ws.h, basis)

        if has_fes
            # The fixed-effect contribution to the linear predictor is the part of
            # the working residual removed by absorption:
            #   α = (z − Xβ) − (z̃ − X̃β)
            # where z̃, X̃ are the demeaned working response and design. Both
            # residuals are already available, so no extra solve is needed.
            mul!(ws.resid, pp.X, pp.beta_new)
            mul!(alpha, Xd, pp.beta_new)
            @inbounds for i in eachindex(alpha)
                alpha[i] = (ws.z[i] - ws.resid[i]) - (zd[i] - alpha[i])
            end
        else
            fill!(alpha, zero(eltype(alpha)))
        end

        update_response!(m, alpha)

        # beta and beta_new are kept as distinct buffers: stephalving! mutates
        # beta_new in place, so beta must be an independent copy.
        converged = abs(rr.deviance_new - rr.deviance) / (0.1 + abs(rr.deviance_new)) <
                    tolerance
        rr.deviance = rr.deviance_new
        copyto!(pp.beta, pp.beta_new)
        converged && break
    end

    return basis
end

############################################################
### FIT BERNOULLI GLM
############################################################
"""
    fit_glmfe(data::DataFrame, formula, beta0, max_iter, tolerance, link) -> BinaryEstimator

Fit a Bernoulli model with the selected probit or logit link by Fisher scoring,
optionally absorbing fixed effects specified by `fe(...)` terms. The response must contain only zeros
and ones. Data preparation currently requires a `DataFrame`.

Each iteration forms working responses and Fisher weights, absorbs fixed
effects, solves weighted least squares, and recovers the fixed-effect
contribution. Workspace buffers and the fixed-effect solver are reused;
solver weights are updated each iteration. Step-halving attempts to reduce
the deviance, with at most 26 halvings per update.

# Arguments
- `link`: `ProbitLink()` or `LogitLink()`.
- `data`: DataFrame containing the response, regressors, and fixed-effect variables.
- `formula`: Model formula, for example `@formula(y ~ x + fe(group))`.
- `beta0`: Initial coefficient vector, or `nothing` for zeros. Its length must
  match the design columns, including the intercept added without fixed effects.
- `max_iter`: Maximum number of Fisher-scoring iterations.
- `tolerance`: Threshold for `abs(Dnew - Dold) / (0.1 + abs(Dnew))`.

Without fixed effects, an explicit intercept is prepended to the design.
With fixed effects, the constant is absorbed. Return a `BinaryEstimator` with
coefficients, response buffers, formula metadata, and parameter counts.
The last iterate is returned even when the iteration limit is reached.
"""
function fit_glmfe(
        @nospecialize(data),
        formula::FormulaTerm,
        beta0::Union{Nothing, Vector},
        max_iter::Integer,
        tolerance::Real,
        link::Link)
    schema, formula_schema, data, X, y = select_columns(data, formula)

    response_name, coef_names_str = get_coefficient_names_nofe(formula, data)

    n = size(data, 1)
    k = count(==(1), y)
    Lnull = k * log(k / n) + (n - k) * log(1 - k / n)
    nulldeviance = -2 * Lnull

    formula, formula_fes = parse_fe(formula)
    fes, feids, fekeys = parse_fixedeffect(data, formula_fes)
    has_fes = has_fe(formula_fes)

    # Fixed effects absorb the constant; without them the model needs an explicit
    # intercept, which the fe-excluded model matrix does not carry.
    if !has_fes
        X = hcat(ones(eltype(X), size(X, 1)), X)
        pushfirst!(coef_names_str, "(Intercept)")
    end

    if beta0 === nothing
        beta0 = zeros(size(X, 2))
    end

    pp = BinaryPredictorQR{Float64}(X, beta0, similar(beta0))

    rr = BinaryResponse(y, pp, response_name, link)

    m = BinaryEstimator{Float64}(
        rr,
        pp,
        formula,
        formula_schema,
        formula_fes,
        n,
        0,
        0,
        nulldeviance,
        has_fes,
        0,
        coef_names_str,
        trues(length(coef_names_str))
    )
    m.deviance = deviance(m)

    ###############################################
    ############ ESTIMATION LOOP ##################
    ###############################################
    ws = GLMWorkspace(X)
    # Build the fixed-effect solver once; only its weights change per iteration.
    feM = has_fes ?
          AbstractFixedEffectSolver{Float64}(fes, Weights(ones(n)), Val{:cpu}) : nothing
    basis = _glm_irls!(m, ws, feM, has_fes, max_iter, tolerance)

    ################################
    ## Summary statistics
    ################################
    dof_fes = _fe_dof(fes)
    m.basis_coef = basis
    m.n_parameters = sum(basis)
    m.fixed_effects_dof = dof_fes

    return m
end

"""
    refresh_response!(rr, X, beta, alpha) -> deviance

Set `rr.eta = X * beta + alpha`, then update `mu`, `one_minus_mu`,
`dmu_deta`, `resid`, and `deviance_contributions` using the selected inverse link
and Bernoulli deviance. Reuse the response buffers and return the sum of the
per-observation deviance contributions.

Do not assign `rr.deviance` or `rr.deviance_new`; the caller stores the returned
value so an accepted deviance remains available during candidate evaluation.
"""
function refresh_response!(rr::BinaryResponse{T}, X, beta, alpha) where {T}
    mul!(rr.eta, X, beta)
    rr.eta .+= alpha
    inverse_link!(rr)
    rr.resid .= rr.y .- rr.mu
    rr.deviance_contributions .= deviance_bernoulli.(rr.y, rr.mu, rr.one_minus_mu)
    return sum(rr.deviance_contributions)
end

"""
    stephalving!(m::BinaryEstimator, alpha)

While the candidate deviance exceeds the accepted deviance, replace
`pp.beta_new` with its midpoint with `pp.beta` and refresh the response.
Keep the supplied fixed-effect contribution `alpha` fixed.

Attempt at most 26 halvings. Update the candidate coefficients, response
buffers, and `rr.deviance_new`, leaving `pp.beta` and `rr.deviance` unchanged.
A non-increasing deviance is not guaranteed if the halving limit is reached.
"""
function stephalving!(m::BinaryEstimator, alpha)
    rr = m.rr
    pp = m.pp
    steps = 0
    while rr.deviance < rr.deviance_new && steps < 26
        pp.beta_new .= (pp.beta .+ pp.beta_new) ./ 2
        rr.deviance_new = refresh_response!(rr, pp.X, pp.beta_new, alpha)
        steps += 1
    end
end

"""
    update_response!(m, alpha)

Evaluate `pp.beta_new` with the supplied fixed-effect contribution `alpha`.
Refresh the linear predictor, probabilities, derivative, residuals, and
per-observation deviance contributions, and store their sum in `rr.deviance_new`.

Call `stephalving!`, which attempts up to 26 halvings if the deviance increases.
Leave acceptance of the candidate coefficients and deviance to the caller.
"""
function update_response!(m, alpha)
    rr = m.rr
    pp = m.pp
    rr.deviance_new = refresh_response!(rr, pp.X, pp.beta_new, alpha)
    stephalving!(m, alpha)
end

"""
    inverse_link!(rr::BinaryResponse{T, ProbitLink}) -> rr

Update `rr.mu`, `rr.one_minus_mu`, and `rr.dmu_deta` from `rr.eta` in place.
Use the standard normal CDF, complementary CDF, and PDF, respectively.
The complementary CDF avoids cancellation when `mu` is close to one.
"""
function inverse_link!(rr::BinaryResponse{T, ProbitLink}) where {T}
    rr.mu .= normcdf.(rr.eta)
    rr.one_minus_mu .= StatsFuns.normccdf.(rr.eta)
    rr.dmu_deta .= exp.(normlogpdf.(rr.eta))
    return rr
end

"""
    inverse_link!(rr::BinaryResponse{T, LogitLink}) -> rr

Update the response probabilities and inverse-link derivative for the logit
link: `mu = logistic(eta)`, `one_minus_mu = logistic(-eta)`, and
`dmu_deta = mu * one_minus_mu`, elementwise. The complement formula avoids
subtraction from one. Return the response object.

Fill the existing `rr.mu`, `rr.one_minus_mu`, and `rr.dmu_deta` buffers in place.
"""
function inverse_link!(rr::BinaryResponse{T, LogitLink}) where {T}
    rr.mu .= StatsFuns.logistic.(rr.eta)
    rr.one_minus_mu .= StatsFuns.logistic.(-rr.eta)
    rr.dmu_deta .= rr.mu .* rr.one_minus_mu
    return rr
end

"""
    link_probit(mu) -> eta

Evaluate the probit link `eta = Φ⁻¹(mu)` for a scalar probability `mu` in
`[0, 1]`, where `Φ` is the standard normal CDF. The endpoints `0` and `1`
map to `-Inf` and `Inf`, respectively.
"""
function link_probit(mu)
    return quantile(Normal(), mu)
end

"""
    link_logit(mu) -> eta

Evaluate the logit link `eta = log(mu / (1 - mu))` for a scalar probability
`mu` in `[0, 1]`. The endpoints `0` and `1` map to `-Inf` and `Inf`,
respectively. Use `log1p(-mu)` to evaluate the logarithm of the complement.
"""
function link_logit(mu)
    return log(mu) - log1p(-mu)
end

"""
    var_bernoulli(rr) -> variances

Return a vector of Bernoulli variances `rr.mu .* rr.one_minus_mu`.
The probability buffers must be updated before calling this function.
"""
function var_bernoulli(rr::BinaryResponse)
    return rr.mu .* rr.one_minus_mu
end

"""
    deviance_bernoulli(y, mu, one_minus_mu) -> deviance

Return the deviance contribution of one Bernoulli observation `y` (zero or
one), given its success probability `mu` and complement `one_minus_mu`.
The contribution is `-2 * log(mu)` when `y == 1` and
`-2 * log(one_minus_mu)` when `y == 0`. Sum the contributions to obtain
the total deviance.

Accepting the complement separately allows callers to supply a value computed
with `ccdf`. Branching on `y` avoids `0 * log(0)` at probability endpoints;
an observation assigned zero probability has infinite deviance.
"""
function deviance_bernoulli(y, mu, one_minus_mu)
    if y == 1
        return -2 * log(mu)
    elseif y == 0
        return -2 * log(one_minus_mu)
    end
    throw(DomainError(y, "Bernoulli response must be zero or one"))
end

"""
    fit_probit(data, formula, beta0, max_iter, tolerance) -> BinaryEstimator

Compatibility wrapper for `fit_glmfe` with `ProbitLink()`.
See [`glmfe`](@ref) for the public formula-first interface.
"""
function fit_probit(data, formula, beta0, max_iter, tolerance)
    return fit_glmfe(data, formula, beta0, max_iter, tolerance, ProbitLink())
end
