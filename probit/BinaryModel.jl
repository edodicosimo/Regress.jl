using Regress
using Regress: AbstractRegressModel
using StatsAPI
using StatsBase

mutable struct BinaryResponse{T <: AbstractFloat}
    y::Vector{T} #observed response vector
    distribution::Distribution #assumed distribution of the response
    v::Vector{Tuple{T,T,T}} # Tuple containing (score, hessian, likelihood) for each observation
    deviance::T
    deviance_new::T # scrape space pre allocated to store the new deviance at each iteration
    eta::Vector{T}
    mu::Vector{T} #fitted values
    wts::Vector{T} #weights
    offset::Vector{T} # Offset (empty = no offset, for GLM compatibility)
    response_name::Symbol
end 

function StatsAPI.deviance(rr::BinaryResponse)
    total_log_likelihood = sum(getindex.(rr.v,3))
    deviance = -2 * total_log_likelihood
end

mutable struct BinaryPredictorQR{T <: AbstractFloat, W <: AbstractWeights}
    X::Matrix{T}
    X_reduced::Matrix{T} #FIXME Non collinear columns only #vector of columns used for the demeaning
    beta::Vector{T}            # coefficient estimates before last cycle update
    deltaBeta::Vector{T}
    beta_new::Vector{T}   #temporary allocation for computation
    weights::W
    tildaX::Matrix{T} #to put demeaned X
    z::Vector{T}
    tildaz::Vector{T}
    # qr::LinearAlgebra.QRCompactWY{T, Matrix{T}} # QR factorization of X_reduced #TODO add QR factorization 
end



##########
struct ILSEstimator{T <: AbstractFloat, P <: Regress.OLSLinearPredictor{T}} <:
       AbstractRegressModel
    rr::Regress.OLSResponse{T}              # Response object
    pp::P                           # Predictor object (Chol or QR)
    basis_coef::BitVector           # Which coefficients are not collinear
end

function StatsAPI.coef(m::ILSEstimator)
    beta = copy(m.pp.beta)
    beta[.!m.basis_coef] .= zero(eltype(beta))
    return beta
end

basis_coef(m::ILSEstimator) = m.basis_coef


########

struct BinaryEstimator{T <: AbstractFloat} <: AbstractRegressModel
    rr :: BinaryResponse{T}
    pp :: BinaryPredictorQR{T}

    # Formula and metadata
    formula::FormulaTerm
    formula_schema::FormulaTerm

    n_observations::Int      
    n_parameters::Int        
    rss::Float64             
    tss::Float64             
    has_fixed_effects::Bool  
    fixed_effects_dof::Int   
    
    # Coefficient metadatas
    coefnames::Vector{String}
    basis_coef::BitVector
end

has_iv(::BinaryEstimator) = false
has_fe(m::BinaryEstimator) = Regress.has_fe(m.formula)


basis_coef(m::BinaryEstimator) = m.basis_coef

StatsAPI.islinear(::BinaryEstimator) = false
StatsAPI.coefnames(m::BinaryEstimator) = m.coefnames #return coefficient names NO FE variables
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

function Base.show(io::IO,::MIME"text/plain",m::BinaryEstimator)
    show(io, MIME"text/plain"(), coeftable(m))
end
