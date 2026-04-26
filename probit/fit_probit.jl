using Distributions
using DataFrames
using Regress
using Regress: fe
using StatsModels: @formula
using StatsFuns
using LinearAlgebra



function fit_probit(
    df,
    beta0,
    max_iter,
    tolerance #if the difference between the old beta and the new one is below the tolerance stop 
)
    rwm_data = df
    beta = beta0

    rwm_data.alpha_new .= 0

    for _ in 1:max_iter

        y = Matrix(rwm_data[:,Cols("visit_dummy")])
        X_i = rwm_data[:,Cols("age", "hhninc", "hhkids", "educ", "married")] #
        X_i = Matrix(X_i) #
        alpha = rwm_data.alpha_new
        select!(rwm_data, Not(:alpha_new))

        eta = X_i * beta + alpha 

        v = log_likelihood_probit.(y,eta) 

        gi = getindex.(v, 1)
        hi = getindex.(v, 2)

        z_i = eta .+ (gi./hi) 
        id = Matrix(rwm_data[:,Cols("id")])
        df = DataFrame( #FIXME funziona solo con 5 regressori, va generalizzato
            z = vec(z_i),
            y = vec(y),
            h = vec(hi),
            x1 = X_i[:,1],
            x2 = X_i[:,2],
            x3 = X_i[:,3],
            x4 = X_i[:,4],
            x5 = X_i[:,5],
            id = vec(id)
        )

        m = Regress.ols(
            df,
            @formula(z ~ x1 + x2 + x3 + x4 + x5 + fe(id));
            weights = :h,
            save = :fe,
        )

        beta_new = coef(m)                       # coefficienti delle x
        alpha_new = Regress.fe(m; keepkeys = true) # fixed effects stimati, con chiave c
        hat_y = predict(m, df)       
        rwm_data = dropmissing(rename!(leftjoin(rwm_data, unique(alpha_new, :id), on=:id), :fe_id => :alpha_new))

        if norm(beta - beta_new) < tolerance
            beta = beta_new
            break
        end

        beta = beta_new
    end

    return beta
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
        h_i = g_i^2 - eta*g_i
    end
    return (g_i, h_i)
end


