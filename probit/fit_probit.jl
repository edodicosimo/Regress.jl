using Distributions
using DataFrames
import Regress
using Regress: fe
using StatsModels
using StatsFuns
using LinearAlgebra
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
    fes = map(fes) do term
        return term.args[1].sym
    end
    return fes
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

    # store coefficient names for model sumamry
    coef_names_str = get_coefficient_names_nofe(formula, data)

    #initialize beta with the user inputed values
    beta = beta0

    #initialize alpha and append it to the dataframe
    data.alpha = zeros(size(X,1))
    
    # vector to store fitted values, now empty
    hatY = Vector{Float64}()

    # total sum of squares
    tss = sum((y .- mean(y)).^2)

    # a vector of symbols that contains the fixed effect terms in the formula
    fesymbols = save_fe(formula)

    ###############################################
    ############ ESTIMATION LOOP ##################
    ###############################################

    for _ in 1:max_iter

        # eta is the working variable wrt which we compute the ml stats
        eta = X * beta + data.alpha 
        
        #compute the score and Hessian of the likelihood wrt eta
        v = log_likelihood_probit.(y,eta) 
        
        # append the score and hessian to the dataframe
        gi = getindex.(v, 1)
        data.gi = gi
        hi = getindex.(v, 2)
        data.hi = hi
       
        # compute working response and append to the df 
        data.z_i = eta .+ (gi./hi) 
        
        # run WLS fixed effects regression zi ~ rhs
        m = Regress.ols(
            data,
            replace_lhs(formula,:z_i);
            weights = :hi,
            save = :fe,
        )

        # the estimated coefficient
        beta_new = Regress.coef(m) 
        
        # these are the estimated fixed effects
        #dataframe che per ogni dimensione fe ha due colonne: nome e fe_{nome} 
        alpha_new = Regress.fe(m; keepkeys = true) 

        #names of the fe columns already in the data df
        old_cols = setdiff(
            intersect(propertynames(data), propertynames(alpha_new)),
            fesymbols
        )

        #remove old fe columns
        select!(data, Not(old_cols))

        #join fe into the data df
        data = leftjoin(data, alpha_new, on = fesymbols)
        select!(data, Not(:alpha))              # rimuove la vecchia alpha
        rename!(data, :fe_id => :alpha)         # rinomina la nuova
        dropmissing!(data)
        collinear_cols = coefnames(m)[.!m.basis_coef]
        if length(collinear_cols) > 0
            select!(data,Not(collinear_cols))
            keep = .!in.(coefnames(m), Ref(collinear_cols))
            beta = beta[keep]
            beta_new = coef(m)[keep]
            for term in collinear_cols
                formula = drop_term(formula,term)
            end
            coef_names_str = get_coefficient_names_nofe(formula, data)
        end

        _, X, y = select_columns(data, formula)
        
        
        if norm(beta - beta_new) < tolerance
            # store the fitted values 
            hatY = Regress.predict(m,data)
            break
        end
        
        beta = beta_new
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
    else
        g_i = - exp(normlogpdf(eta)-normlogccdf(eta)) 
        h_i = g_i^2 + eta*g_i
    end
    return (g_i, h_i)
end

