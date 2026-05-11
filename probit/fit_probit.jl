using Distributions
using DataFrames
import Regress
using Regress: fe
using StatsModels
using StatsFuns
using LinearAlgebra
using Base.Threads
include("BinaryModel.jl")

#######################################
### HELPER FUNCTION TO CLEAN DATA
#######################################
"""
    select_columns(df::DataFrame, formula::FormulaTerm) -> data, X, y

Return a reduced data frame, model matrix, and response vector using only the
variables referenced by `formula`.
"""
function select_columns(df::DataFrame, formula::FormulaTerm)
    formula_without_fe = remove_fixedeffects(formula)
    formula = ignore_fe(formula)
    
    y = modelcols(formula.lhs,df)
    X = modelcols(formula.rhs,df)
    
    X_without_fe = modelcols(formula_without_fe.rhs,df)

    schema = StatsModels.schema(formula, df)
    formula_schema = apply_schema(formula,schema)

    response_name = coefnames(formula_schema.lhs) 
    Xnames = coefnames(formula_schema.rhs)

    out = DataFrame()

    out[!, response_name] = vec(y)
    for (j,name) in enumerate(Xnames)
        out[!, name] = X[j]
    end
    return (schema, formula_schema,out, reduce(hcat,X_without_fe), vec(y))
    
end

"""
    ignore_fe(f::FormulaTerm) -> FormulaTerm

Return a formula where each `fe(x)` term is converted to the ordinary term `x`.
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
    remove_fixedeffects(f::FormulaTerm) -> FormulaTerm

Return a formula with all `fe(...)` terms removed from the right-hand side.
"""
function remove_fixedeffects(f::FormulaTerm)
    rhs_terms = f.rhs isa Tuple ? collect(f.rhs) : collect(f.rhs.terms)

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
    formula_schema = apply_schema(formula,schema)
    response_name, coef_names = coefnames(formula_schema.lhs), coefnames(formula_schema.rhs)
    coef_names_str = String[string(name) for name = coef_names] 
    return (Symbol(response_name),coef_names_str)
end


############################################################
### OLS SOLVER
############################################################
"""
    ils_solver(X, y; factorization = :auto, collinearity = :qr, tol = 1e-8,
               weights = nothing, has_intercept = true) -> ILSEstimator

Fit the weighted least-squares step used by iterative least squares (ILS).
This is a lightweight OLS solver for internal probit iterations, returning
coefficients and the non-collinear coefficient mask without inference results.
"""
function ils_solver(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
        factorization::Symbol = :auto,
        collinearity::Symbol = :qr,
        tol::Real = 1e-8,
        weights::Union{Nothing, AbstractVector} = nothing,
        has_intercept::Bool = true)

    # Validate inputs
    n, k = size(X)
    length(y) == n ||
        throw(DimensionMismatch("X has $n rows but y has $(length(y)) elements"))

    # Validate keywords
    factorization in (:auto, :chol, :qr) ||
        throw(ArgumentError("factorization must be :auto, :chol, or :qr, got :$factorization"))
    collinearity in (:qr, :sweep) ||
        throw(ArgumentError("collinearity must be :qr or :sweep, got :$collinearity"))

    # Determine numeric type
    T = promote_type(eltype(X), eltype(y))
    T <: AbstractFloat || (T = Float64)

    # Convert to Matrix{T} and Vector{T} (materializes views)
    X_mat = convert(Matrix{T}, X)
    y_vec = convert(Vector{T}, y)

    # Handle weights
    has_weights = weights !== nothing
    if has_weights
        length(weights) == n || throw(DimensionMismatch("weights must have length $n"))
        wts_vec = convert(Vector{T}, weights)
        sqrtw = sqrt.(wts_vec)
        X_mat = X_mat .* sqrtw
        y_vec = y_vec .* sqrtw
    else
        wts_vec = T[]
    end

    # Choose factorization
    if factorization == :auto
        factorization = k < 100 ? :chol : :qr
    end

    # Build response object
    mu = similar(y_vec)
    rr = Regress.OLSResponse(y_vec, mu, wts_vec, T[], :y)

    # Fit using unified solver
    pp, basis_coef,
    _ = Regress.fit_ols_core!(rr, X_mat, factorization;
        tol = tol, save_matrices = true, collinearity = collinearity)


    return ILSEstimator{T, typeof(pp)}(
        rr, pp, basis_coef
    )
end

############################################################
### Probit specific likelihood helper
############################################################
"""
    log_likelihood_probit(y, eta)

Return the score, observed information, and log-likelihood contribution for a
single probit observation with response `y` and linear predictor `eta`.
"""
function log_likelihood_probit(y,eta)
    if y == 1
        gi = exp(normlogpdf(eta)-normlogcdf(eta))
        hi = gi^2 + eta * gi
        di = normlogcdf(eta)
    else
        gi = - exp(normlogpdf(eta)-normlogccdf(eta)) 
        hi = gi^2 + eta*gi
        di = normlogccdf(eta)
    end
    return (gi, hi,di)
end


############################################################
### FIT PROBIT
############################################################
"""
    fit_probit(data, formula, beta0, max_iter, tolerance) -> BinaryEstimator

Estimate a binary-response probit model, optionally absorbing fixed effects
specified with `fe(...)` terms in `formula`.

The estimator uses iterative reweighted least squares. At each iteration, fixed
effects are partialled out before solving the weighted least-squares update for
the slope coefficients.

# Arguments
- `data`: Input table containing the response, regressors, and fixed-effect
  variables.
- `formula::FormulaTerm`: A `StatsModels.jl` formula, for example
  `@formula(y ~ x1 + x2 + fe(group))`.
- `beta0::Vector`: Initial coefficient vector for the non-fixed-effect
  regressors.
- `max_iter::Integer`: Maximum number of IRLS iterations.
- `tolerance::Real`: Convergence tolerance for the deviance update.

# Returns
- `BinaryEstimator`: Fitted model containing the response, fitted
  probabilities, coefficient estimates, model matrix, formula, and coefficient
  names.
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
    schema, formula_schema, data, X, y = select_columns(data, formula)

    # store coefficient names for model summary, ignroes fe variables
    response_name, coef_names_str = get_coefficient_names_nofe(formula, data)

    #initialize alpha to all 0 and append it to the dataframe
    alpha = zeros(size(X,1))
    
    # vector to store fitted values, now empty
    fitted_probabilities = Vector{Float64}()

    # total sum of squares
    tss = sum((y .- mean(y)).^2)

    # formula parsing
    formula, formula_fes = Regress.parse_fe(formula)
    fes, feids, fekeys = Regress.parse_fixedeffect(data, formula_fes)


    ## Instantiate predictor object
    pp = BinaryPredictorQR{Float64,Weights}(
            X,similar(X),
            beta0,similar(beta0),
            similar(beta0),Weights(ones(length(y))),
            similar(X),similar(y), similar(y)
        )
    

    #Initialize variables
    eta = pp.X * pp.beta
    v = log_likelihood_probit.(y,eta)
    total_log_likelihood = sum(getindex.(v,3)) 
    deviance = -2 * total_log_likelihood
    basis_coef_mask = trues(length(beta0))
    beta = copy(pp.beta)

    ###############################################
    ############ ESTIMATION LOOP ##################
    ###############################################
    i = 0
    for _ in 1:max_iter
        println(i)
        X̃, X = pp.tildaX, pp.X
        z̃, z = pp.tildaz,pp.z
        copyto!(X̃,X)
        #compute the score (gi) and Hessian (hi) of the likelihood wrt eta
        gi = getindex.(v, 1)
        hi = getindex.(v, 2)
        
        # compute working response 
        z̃ .= eta .+ (gi./hi) 
        copyto!(z,z̃)
        
        # create a vector of vectors with zi and then all the columns of X, to then pass it to partialout
        # this will be modified in place
        cols = Vector{AbstractVector{Float64}}(collect(eachcol(X̃))) #this is a view so it does not allocate
        pushfirst!(cols, z̃) 

        feM, _,_,_,_,_ = Regress.partial_out_fixed_effects!(
            cols,
            coef_names_str,
            fes,
            Weights(hi),
            :cpu, # TODO make this an argument,
            Threads.nthreads(),
            1e-6,
            10000,
            true,
            false,
            true, 
            true, 
            Float64
        ) # this modifies X and z in place

        wls = ils_solver(
                X̃,
                z̃,
                weights= hi
            )

        betanew = Regress.coef(wls)


        newfes, _ , _ = Regress.solve_coefficients!(
            z - X * betanew,
            feM;
            tol = 1e-6,
            maxiter = 1000
        )
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
        pp.beta = betanew
        i += 1
    end

    ################################
    ## Summary statistics
    ################################

    fitted_probabilities = normcdf.(eta)



    ################################
    ## Construct Return Objects
    ################################

    rr = BinaryResponse{Float64}(
        convert(Vector{Float64}, y),
        fitted_probabilities,
        Vector{Float64}(),
        Vector{Float64}(),
        response_name
    )
    estimator = BinaryEstimator{Float64}(
        rr,
        pp,
        formula,
        formula_schema,
        size(data,1),
        0,
        0.0,
        tss,
        true,
        0,
        coef_names_str,
        trues(length(coef_names_str))
    )
    return estimator
end
