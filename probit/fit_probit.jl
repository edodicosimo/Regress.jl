using Distributions
using DataFrames
using Regress
using Regress: fe
using StatsModels: @formula
using StatsFuns
using LinearAlgebra

# for iter in 1:max_iter

#     eta = X * beta + alpha

#     p   = Phi(eta)
#     pdf = phi(eta)

#     g = score_probit_wrt_eta(y, p, pdf)
#     h = neg_hessian_probit_wrt_eta(y, p, pdf)

#     z = eta + g / h

#     z_tilde = demeaning(z, fe, weights=h)
#     X_tilde = demeaning(X, fe, weights=h)

#     beta_new = weighted_least_squares(X_tilde, z_tilde, weights=h)

#     r = z - X * beta_new
#     alpha_new = recover_fixed_effects(r, fe, weights=h)

#     if norm(beta_new - beta) < tol
#         beta = beta_new
#         alpha = alpha_new
#         break
#     end

#     beta  = beta_new
#     alpha = alpha_new

# end

# return beta, alpha

function fit_probit(
    y,
    X,
    fe, #TODO serve un identificatore degli effetti fissi, cioè il cluster di appartenenza (firm, year...)
    beta0,
    alpha0,
    max_iter,
    tolerance #if the difference between the old beta and the new one is below the tolerance stop 
)
    beta = beta0
    alpha = alpha0
    for i in 1:max_iter
        eta = X * beta + alpha

        v = log_likelihood_probit.(y,eta) #TODO check if the results here are correct

        gi = getindex.(v, 1)
        hi = getindex.(v, 2)

        z_i = eta .+ (gi./hi) 
        df = DataFrame(
            z = z_i,
            y = y_i,
            h = hi,
            x1 = X_i[:,1],
            x2 = X_i[:,2],
            x3 = X_i[:,3],
            x4 = X_i[:,4],
            x5 = X_i[:,5],
            c = ci
        )

        m = Regress.ols(
            df,
            @formula(z ~ x1 + x2 + x3 + x4 + x5 + fe(c));
            weights = :h,
            save = :fe,
        )

        beta_new = coef(m)                       # coefficienti delle x
        alpha_new = Regress.fe(m; keepkeys = true) # fixed effects stimati, con chiave c
        ŷ = predict(m, df)           
    end

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
        g_i = exp(normlogpdf(eta)-normlogccdf(eta)) 
        h_i = g_i^2 - eta*g_i
    end
    return (g_i, h_i)
end


