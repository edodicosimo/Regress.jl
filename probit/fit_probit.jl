using Distributions
using DataFrames
using Regress
using Regress: fe
using StatsModels
using StatsFuns
using LinearAlgebra


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

function fit_probit(
    data,
    formula,
    beta0,
    max_iter,
    tolerance #if the difference between the old beta and the new one is below the tolerance stop 
)
    #1. parse the formula and return a dataframe with only the needed columns, X::Matrix, y::Vector.
    data, X, y = select_columns(data, formula)

    #initialize beta with the user inputed values
    beta = beta0

    #initialize alpha and append it to the dataframe
    data.alpha = zeros(size(X,1))
    
    iteration = 0
    for _ in 1:max_iter

        #2. compute eta 
        eta = X * beta + data.alpha 
        
        #compute the score and Hessian of the likelihood wrt eta
        v = log_likelihood_probit.(y,eta) 

        gi = getindex.(v, 1)
        data.gi = gi
        hi = getindex.(v, 2)
        data.hi = hi

        data.z_i = eta .+ (gi./hi) 
        
        m = Regress.ols(
            data,
            replace_lhs(formula,:z_i);
            weights = :hi,
            save = :fe,
        )

        beta_new = coef(m)                       # coefficienti delle x
        alpha_new = Regress.fe(m; keepkeys = true) # fixed effects stimati, con chiave c
        # hat_y = predict(m, data)       
        # data = dropmissing(rename!(leftjoin(data, unique(alpha_new, :id), on=:id), :fe_id => :alpha_new))
        data = leftjoin(data, unique(alpha_new, :id), on=:id)

        select!(data, Not(:alpha))              # rimuove la vecchia alpha
        rename!(data, :fe_id => :alpha)         # rinomina la nuova
        dropmissing!(data)
        _, X, y = select_columns(data, formula)
        
        
        diagnostic = (norm(gi),maximum(abs.(gi)))
        if norm(beta - beta_new) < tolerance
            return(beta_new,"stopped because difference between new and old β < $(tolerance) at iteration $(iteration)",data,diagnostic)
        end
        iteration += 1
        beta = beta_new
    end

    return (beta,iteration,data)
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


