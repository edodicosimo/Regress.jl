using Distributions
using DataFrames
import Regress
using Regress: fe
using StatsModels
using StatsFuns
using LinearAlgebra
using Base.Threads
include("BinaryModel.jl")


"""
    select_columns(df::DataFrame, formula::FormulaTerm) -> df::DataFrame, X::Matrix, y::Vector
Select from a dataframe only the columns specified by the formula
"""
function select_columns(df::DataFrame, formula::FormulaTerm)
    formulanofe = remove_fixedeffects(formula)
    formula = ignore_fe(formula)
    
    y = modelcols(formula.lhs,df)
    X = modelcols(formula.rhs,df)
    
    Xnofe = modelcols(formulanofe.rhs,df)

    schema = StatsModels.schema(formula, df)
    f_s = apply_schema(formula,schema)

    yname = coefnames(f_s.lhs) 
    Xnames = coefnames(f_s.rhs)

    out = DataFrame()

    out[!, yname] = vec(y)
    for (j,name) in enumerate(Xnames)
        out[!, name] = X[j]
    end
    return (out, reduce(hcat,Xnofe), vec(y))
    
end

"""
    ignore_fe(f::FormulaTerm) -> FormulaTerm
given a formula return the same formula but with the fe terms treated as non-fe
es: ignore_re(@formula(y ~ x + fe(z))) -> @formula(y ~ x + z) 
"""
function ignore_fe(f::FormulaTerm)
    rhs_terms = f.rhs isa Tuple ? collect(f.rhs) : collect(f.rhs.terms)

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
    remove_fe(f::FormulaTerm) -> FormulaTerm
given a formula removes the term that are as such fe(term)
"""
function remove_fixedeffects(f::FormulaTerm)
    rhs_terms = f.rhs isa Tuple ? collect(f.rhs) : collect(f.rhs.terms)

    new_rhs = filter(rhs_terms) do t
        !(t isa FunctionTerm{typeof(fe)})
    end

    return FormulaTerm(f.lhs, Tuple(new_rhs))
end

function replace_lhs(f::FormulaTerm, new_lhs::Symbol)
    return Term(new_lhs) ~ f.rhs
end

"""
    save_fe(f::FormulaTerm) -> Vector{Symbol}
From a formula return a vector of symbols that contains the fixed effect terms
"""
function save_fe(f::FormulaTerm)
    rhs_terms = f.rhs isa Tuple ? collect(f.rhs) : collect(f.rhs.terms)
    fes = filter(rhs_terms) do term
        (term isa FunctionTerm{typeof(fe)})
    end
    fesymbol = map(fes) do term
        return term.args[1].sym
    end
    return fes,fesymbol
end

function get_coefficient_names_nofe(formula::FormulaTerm, data::DataFrame)
    formula = remove_fixedeffects(formula)
    schema = StatsModels.schema(formula, data)
    f_s = apply_schema(formula,schema)
    response_name, coef_names = coefnames(f_s.lhs), coefnames(f_s.rhs)
    coef_names_str = String[string(name) for name = coef_names] 
end

function drop_term(f::FormulaTerm, sym::String)
    sym = Symbol(sym)
    rhs_terms = filter(t -> t != term(sym), collect(f.rhs))
    return f.lhs ~ sum(rhs_terms)
end
"""
    fit_probit(data, formula, beta0, max_iter, tolerance) -> BinaryEstimator

Estimate a probit model with fixed effects using Iteratively Reweighted Least Squares (IRLS).

# Arguments
- `data::DataFrame`: Input dataset. 
- `formula::FormulaTerm`: A `StatsModels.jl` formula created using `@formula(y ~ x1 + x2 + fe(group))`.
- `beta0::Vector` : Initial guess for the coefficient vector β.
- `max_iter::Int`  : Maximum number of iterations allowed.
- `tolerance::Real` : Convergence threshold based on the norm of successive β updates.

# Returns 

"""
function fit_probit(
    @nospecialize(data),
    formula::FormulaTerm,
    beta0::Vector,
    max_iter::Integer,
    tolerance::Real #if the difference between the old beta and the new one is below the tolerance stop 
)
    ###############################################
    ###### FORMULA PARSING AND DATA CLEANING ######
    ###############################################

    #parse the formula and return a dataframe with only the needed columns, X::Matrix, y::Vector.
    data, X, y = select_columns(data, formula)

    # store coefficient names for model summary, ignroes fe variables
    coef_names_str = get_coefficient_names_nofe(formula, data)

    #initialize beta with the user inputed values
    beta = beta0

    #initialize alpha to all 0 and append it to the dataframe
    alpha = zeros(size(X,1))
    
    # vector to store fitted values, now empty
    hatY = Vector{Float64}()

    # total sum of squares
    tss = sum((y .- mean(y)).^2)

    # formula parsing
    formula, formula_fes = Regress.parse_fe(formula)
    fes, feids, fekeys = Regress.parse_fixedeffect(data, formula_fes)

    #Initialize variables
    eta = X * beta
    v = log_likelihood_probit.(y,eta)
    total_log_likelihood = sum(getindex.(v,3)) 
    deviance = -2 * total_log_likelihood

    ###############################################
    ############ ESTIMATION LOOP ##################
    ###############################################
    i = 0
    for _ in 1:max_iter
        i += 1

        #compute the score (gi) and Hessian (hi) of the likelihood wrt eta
        gi = getindex.(v, 1)
        hi = getindex.(v, 2)
        
        # compute working response and append to the df 
        zi = eta .+ (gi./hi) 
        
        # create a vector of vectors with zi and then all the columns of X, to then pass it to partialout
        # this will be modified in place
        cols = Vector{AbstractVector{Float64}}(collect(eachcol(X)))
        pushfirst!(cols, zi)

        feM, iterations,
        converged,
        tss_partial,
        oldz,
        oldX= Regress.partial_out_fixed_effects!(
            cols,
            coef_names_str,
            fes,
            Weights(hi),
            :cpu, # TODO make this an argument,
            Threads.nthreads(),
            1e-6,
            10000,
            true,
            true, #we need to always save fixed effects,
            true, 
            true, 
            Float64
        ) # this modifies X and z in place

        betanew = Regress.coef(
            Regress.ols(
                X,
                zi,
                weights= hi
            )
        )

        PO = [feM, iterations,
        converged,
        tss_partial,
        oldz,
        oldX]
        newfes, b, c = Regress.solve_coefficients!(
            oldz - oldX * betanew,
            feM;
            tol = 1e-6,
            maxiter = 1000
        )

        X = oldX
        
        alpha = stack(newfes)

        # Compute eta(t) = X * beta + alpha(t) using current iteration alpha but original X
        alpha_sum = alpha isa AbstractVector ? alpha : vec(sum(alpha, dims = 2))
        eta = X * betanew .+ alpha_sum

        v = log_likelihood_probit.(y,eta)
        total_log_likelihood = sum(getindex.(v,3))
        deviance_new = -2 * total_log_likelihood

        # step halving
        steps = 0
        while deviance < deviance_new && steps < 26
            betanew = (beta .+ betanew) ./2
            eta = X * betanew .+ alpha_sum
            v = log_likelihood_probit.(y,eta)
            total_log_likelihood = sum(getindex.(v,3))
            deviance_new = -2 * total_log_likelihood
            steps += 1
        end
        
        if norm(deviance_new - deviance) / (0.1 + norm(deviance_new))  < tolerance
            beta = betanew
            break
        end
        deviance = deviance_new
        beta = betanew
        
    end
 rr = BinaryResponse{Float64}(
        y,
        hatY, #FIXME non so se ci va yhat qua, cosa sono i valori fittati nel probit?
        Vector{Float64}(),
        Vector{Float64}(),
        :simboloToFix #FIXME

    )
    pp = BinaryPredictorQR{Float64}(
        X,
        Matrix{Float64}(undef,0,0),
        beta
    )
    estimator = BinaryEstimator{Float64}(
        rr,
        pp,
        formula,
        size(data,1),
        0,
        0.0,
        tss,
        true,
        0,
        coef_names_str
    )
    return estimator
end



"""
An helper function for fit_probit, it computes the score and the Hessian
Input:  
- a pdf
- a cdf
- a value of y_i  
computes the score of the log likelihood wrt eta and the hessian matrix 
"""
function log_likelihood_probit(y,eta)
    if y == 1
        g_i = exp(normlogpdf(eta)-normlogcdf(eta))
        h_i = g_i^2 + eta * g_i
        di = normlogcdf(eta)
    else
        g_i = - exp(normlogpdf(eta)-normlogccdf(eta)) 
        h_i = g_i^2 + eta*g_i
        di = normlogccdf(eta)
    end
    return (g_i, h_i,di)
end


