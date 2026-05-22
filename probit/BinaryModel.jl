using Regress
using Regress: AbstractRegressModel
using StatsAPI
using StatsBase

# ── Response ─────────────────────────────────────────────────────────────────

"""
    BinaryResponse{T <: AbstractFloat}

Working storage for binary model response. Holds observed data, fitted values,
and per-observation scores/hessians/log-likelihoods updated at each IRLS iteration.

# Fields
- `y`: observed binary response vector
- `distribution`: assumed link distribution (e.g. Normal for probit)
- `v`: per-observation `(score, hessian, log-likelihood)` tuples
- `deviance`, `deviance_new`: current and candidate deviance (convergence check)
- `eta`: linear predictor 
- `mu`: fitted probabilities (inverse-link of η)
- `wts`: observation weights
- `offset`: optional offset vector (empty = no offset)
- `response_name`: symbol name of the response variable
"""
mutable struct BinaryResponse{T <: AbstractFloat}
    y::Vector{T}
    distribution::Distribution
    v::Vector{Tuple{T,T,T}}
    deviance::T
    deviance_new::T
    eta::Vector{T}
    mu::Vector{T}
    wts::Vector{T}
    offset::Vector{T}
    response_name::Symbol
end

function StatsAPI.deviance(rr::BinaryResponse)
    total_log_likelihood = sum(getindex.(rr.v, 3))
    return -2 * total_log_likelihood
end

# ── Predictor ─────────────────────────────────────────────────────────────────

"""
    BinaryPredictorQR{T <: AbstractFloat, W <: AbstractWeights}

QR-based predictor for binary models. Stores the design matrix, coefficient
estimates, and demeaned working variables used in each IRLS step.

# Fields
- `X`: full design matrix
- `X_reduced`: non-collinear columns only (FIXME: currently same as X)
- `beta`: current coefficient estimates
- `deltaBeta`: coefficient update from last iteration
- `beta_new`: scratch space for candidate coefficients
- `weights`: observation weights
- `tildaX`: demeaned design matrix (after FE absorption)
- `z`, `tildaz`: working response and its demeaned version
"""
mutable struct BinaryPredictorQR{T <: AbstractFloat, W <: AbstractWeights}
    X::Matrix{T}
    X_reduced::Matrix{T}  #FIXME Non collinear columns only
    beta::Vector{T}
    deltaBeta::Vector{T}
    beta_new::Vector{T}
    weights::W
    tildaX::Matrix{T}
    z::Vector{T}
    tildaz::Vector{T}
    # qr::LinearAlgebra.QRCompactWY{T, Matrix{T}} #TODO add QR factorization
end

# ── ILS inner model ───────────────────────────────────────────────────────────

"""
    ILSEstimator{T <: AbstractFloat, P <: Regress.OLSLinearPredictor{T}}

Iterated Least Squares sub-model wrapping an OLS response and predictor.
Used as the inner linear step of IRLS binary model fitting.
`basis_coef` marks which columns are linearly independent.
"""
struct ILSEstimator{T <: AbstractFloat, P <: Regress.OLSLinearPredictor{T}} <:
       AbstractRegressModel
    rr::Regress.OLSResponse{T}
    pp::P
    basis_coef::BitVector
end

function StatsAPI.coef(m::ILSEstimator)
    beta = copy(m.pp.beta)
    beta[.!m.basis_coef] .= zero(eltype(beta))
    return beta
end

basis_coef(m::ILSEstimator) = m.basis_coef

# ── Fitted model ──────────────────────────────────────────────────────────────

"""
    BinaryEstimator{T <: AbstractFloat}

Fitted binary regression model (e.g. probit, logit). Combines a `BinaryResponse`
and `BinaryPredictorQR` with formula metadata and summary statistics produced
after IRLS convergence.
"""
struct BinaryEstimator{T <: AbstractFloat} <: AbstractRegressModel
    rr::BinaryResponse{T}
    pp::BinaryPredictorQR{T}

    formula::FormulaTerm
    formula_schema::FormulaTerm

    n_observations::Int
    n_parameters::Int
    rss::Float64
    tss::Float64
    has_fixed_effects::Bool
    fixed_effects_dof::Int

    coefnames::Vector{String}
    basis_coef::BitVector
end

has_iv(::BinaryEstimator) = false
has_fe(m::BinaryEstimator) = Regress.has_fe(m.formula)

basis_coef(m::BinaryEstimator) = m.basis_coef

StatsAPI.islinear(::BinaryEstimator) = false
StatsAPI.coefnames(m::BinaryEstimator) = m.coefnames
StatsAPI.responsename(m::BinaryEstimator) = m.rr.response_name

function StatsAPI.coef(m::BinaryEstimator)
    m.pp.beta
end

function StatsAPI.fitted(m::BinaryEstimator)
    m.rr.mu
end

function StatsAPI.response(m::BinaryEstimator)
    m.rr.y
end

function StatsAPI.residuals(m::BinaryEstimator)
    m.rr.y - m.rr.mu
end

function StatsAPI.modelmatrix(m::BinaryEstimator)
    m.pp.X
end

function coeftable(m::BinaryEstimator)
    CoefTable(
        [coef(m)],
        ["Estimate"],
        coefnames(m)
    )
end

function Base.show(io::IO, ::MIME"text/plain", m::BinaryEstimator)
    show(io, MIME"text/plain"(), coeftable(m))
end
