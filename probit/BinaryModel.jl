using Regress
using Regress: AbstractRegressModel
using StatsAPI
using StatsBase

mutable struct BinaryResponse{T <: AbstractFloat}
    y::Vector{T} #the independent variable 
    mu::Vector{T} #fitted values
    wts::Vector{T} #weights
    offset::Vector{T} # Offset (empty = no offset, for GLM compatibility)
    response_name::Symbol
end 

mutable struct BinaryPredictorQR{T}
    X::Matrix{T}
    X_reduced::Matrix{T} #FIXME Non collinear columns only
    beta::Vector{T}            # Coefficient estimates (full, with NaN)
    # qr::LinearAlgebra.QRCompactWY{T, Matrix{T}} # QR factorization of X_reduced #TODO add QR factorization 
end


struct BinaryEstimator{T <: AbstractFloat} <: AbstractRegressModel
    rr :: BinaryResponse{T}
    pp :: BinaryPredictorQR{T}
    formula::FormulaTerm
    n_observations::Int      
    n_parameters::Int        
    rss::Float64             
    tss::Float64             
    has_fixed_effects::Bool  
    fixed_effects_dof::Int   
    
    # Coefficient metadatas
    coefnames::Vector{String}
end


has_iv(::BinaryEstimator) = false
has_fe(m::BinaryEstimator) = Regress.has_fe(m.formula)
StatsAPI.islinear(::BinaryEstimator) = false
StatsAPI.coefnames(m::BinaryEstimator) = m.coefnames #return coefficient names NO FE variables

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
