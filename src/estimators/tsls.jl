##############################################################################
##
## Two-Stage Least Squares (TSLS) Estimation
##
##############################################################################

##############################################################################
## Helper functions for IV estimation (extracted for faster compilation)
##############################################################################

"""
    _iv_collinearity_detection!(Xexo, Xendo, Z, XexoXexo, XendoXendo, ZZ,
                                 has_intercept, coefnames_endo)

Multi-stage collinearity detection for IV models.
Returns updated matrices and basis vectors.
"""
function _iv_collinearity_detection!(
        Xexo::Matrix{T}, Xendo::Matrix{T}, Z::Matrix{T},
        XexoXexo::Symmetric{T}, XendoXendo::Symmetric{T}, ZZ::Symmetric{T},
        has_intercept::Bool, coefnames_endo::Vector
) where {T}
    perm = nothing

    # First pass: check collinearity within Xendo
    XendoXendo_check = copy(XendoXendo.data)
    basis_endo = basis!(Symmetric(XendoXendo_check); has_intercept = false)
    if !all(basis_endo)
        Xendo = Xendo[:, basis_endo]
        XendoXendo = Symmetric(XendoXendo.data[basis_endo, basis_endo])
    end

    # Compute cross-products
    XexoZ = Xexo' * Z
    XexoXendo = Xexo' * Xendo
    ZXendo = Z' * Xendo

    # Second pass: joint collinearity check for (Xexo, Z, Xendo)
    k_exo, k_z, k_endo = size(Xexo, 2), size(Z, 2), size(Xendo, 2)
    XexoZXendo = build_block_symmetric(
        [XexoXexo.data, XexoZ, XexoXendo, ZZ.data, ZXendo, XendoXendo.data],
        [k_exo, k_z, k_endo]
    )

    basis_all = basis!(XexoZXendo; has_intercept = has_intercept)
    basis_Xexo = basis_all[1:k_exo]
    basis_Z = basis_all[(k_exo + 1):(k_exo + k_z)]
    basis_endo_small = basis_all[(k_exo + k_z + 1):end]

    # If adding Xexo and Z makes Xendo collinear, recategorize as exogenous
    if !all(basis_endo_small)
        Xexo = hcat(Xexo, Xendo[:, .!basis_endo_small])
        Xendo = Xendo[:, basis_endo_small]

        # Recompute cross-products for new Xexo
        XexoXexo = compute_crossproduct(Xexo)
        XexoZ = Xexo' * Z
        XexoXendo = Xexo' * Xendo
        ZXendo = Z' * Xendo
        XendoXendo = compute_crossproduct(Xendo)

        # Track permutation for coefficient reordering
        basis_endo2 = trues(length(basis_endo))
        basis_endo2[basis_endo] = basis_endo_small
        ans = collect(1:length(basis_endo))
        ans = vcat(ans[.!basis_endo2], ans[basis_endo2])
        perm = vcat(1:(size(Xexo, 2) - count(.!basis_endo_small)),
            (size(Xexo, 2) - count(.!basis_endo_small)) .+ ans)

        out = join(coefnames_endo[.!basis_endo2], " ")
        @info "Endogenous vars collinear with ivs. Recategorized as exogenous: $(out)"

        # Third pass after recategorization
        k_exo, k_z, k_endo = size(Xexo, 2), size(Z, 2), size(Xendo, 2)
        XexoZXendo = build_block_symmetric(
            [XexoXexo.data, XexoZ, XexoXendo, ZZ.data, ZXendo, XendoXendo.data],
            [k_exo, k_z, k_endo]
        )
        basis_all = basis!(XexoZXendo; has_intercept = has_intercept)
        basis_Xexo = basis_all[1:k_exo]
        basis_Z = basis_all[(k_exo + 1):(k_exo + k_z)]
    end

    # Apply basis to matrices
    if !all(basis_Xexo)
        Xexo = Xexo[:, basis_Xexo]
        XexoXexo = Symmetric(XexoXexo.data[basis_Xexo, basis_Xexo])
        XexoXendo = XexoXendo[basis_Xexo, :]
    end
    if !all(basis_Z)
        Z = Z[:, basis_Z]
        ZZ = Symmetric(ZZ.data[basis_Z, basis_Z])
        ZXendo = ZXendo[basis_Z, :]
    end
    XexoZ = XexoZ[basis_Xexo, basis_Z]

    # Compute final basis for coefficients
    basis_endo2 = trues(length(basis_endo))
    basis_endo2[basis_endo] = basis_endo_small
    basis_coef = vcat(basis_Xexo, basis_endo[basis_endo2])

    return (
        Xexo = Xexo, Xendo = Xendo, Z = Z,
        XexoXexo = XexoXexo, XendoXendo = XendoXendo, ZZ = ZZ,
        XexoZ = XexoZ, XexoXendo = XexoXendo, ZXendo = ZXendo,
        basis_endo = basis_endo, basis_endo_small = basis_endo_small,
        basis_Xexo = basis_Xexo, basis_Z = basis_Z,
        basis_coef = basis_coef, perm = perm
    )
end

"""
    _iv_first_stage_fe(Xexo, Xendo, iv_fes, wts, method, nthreads, maxiter, tol, progress_bar, T)

FE-based first-stage regression using Frisch-Waugh-Lovell.
"""
function _iv_first_stage_fe(
        Xexo::Matrix{T}, Xendo::Matrix{T},
        iv_fes::Vector{FixedEffect},
        wts::AbstractWeights,
        method::Symbol, nthreads::Int,
        maxiter::Int, tol::Real, progress_bar::Bool
) where {T}
    k_exo, k_endo = size(Xexo, 2), size(Xendo, 2)
    nobs = size(Xexo, 1)

    # Create FE solver for instrument FEs
    iv_feM = AbstractFixedEffectSolver{T}(iv_fes, wts, Val{method})

    # Step 1: Demean Xexo by instrument FEs
    Xexo_demeaned = copy(Xexo)
    if k_exo > 0
        Xexo_cols = [Xexo_demeaned[:, j] for j in 1:k_exo]
        solve_residuals!(
            Xexo_cols, iv_feM; maxiter = maxiter, tol = tol, progress_bar = progress_bar)
        for j in 1:k_exo
            Xexo_demeaned[:, j] .= Xexo_cols[j]
        end
    end

    # Step 2: Demean Xendo by instrument FEs
    Xendo_demeaned = copy(Xendo)
    Xendo_cols = [Xendo_demeaned[:, j] for j in 1:k_endo]
    solve_residuals!(
        Xendo_cols, iv_feM; maxiter = maxiter, tol = tol, progress_bar = progress_bar)
    for j in 1:k_endo
        Xendo_demeaned[:, j] .= Xendo_cols[j]
    end

    # Step 3: Within-FE regression of Xendo on Xexo
    if k_exo > 0
        XexoXexo_within = compute_crossproduct(Xexo_demeaned)
        XexoXendo_within = Xexo_demeaned' * Xendo_demeaned
        gamma = XexoXexo_within \ XexoXendo_within
        Xendo_within_res = Xendo_demeaned .- Xexo_demeaned * gamma
    else
        gamma = zeros(T, 0, k_endo)
        Xendo_within_res = Xendo_demeaned
    end

    # Step 4: Predicted endogenous = original Xendo - within residuals
    Xendo_hat = Xendo .- Xendo_within_res

    return (newZ = Xexo, Pi = gamma, Xendo_hat = Xendo_hat,
        XexoZ = zeros(T, k_exo, 0),
        Z = Matrix{T}(undef, nobs, 0),
        ZZ = Symmetric(zeros(T, 0, 0)),
        ZXendo = zeros(T, 0, k_endo))
end

"""
    _iv_first_stage_fstats_standard(...)

Compute first-stage F-statistics for standard IV.
"""
function _iv_first_stage_fstats_standard(
        Xendo::Matrix{T}, newZ::Matrix{T}, Pi::Matrix{T},
        XexoXexo::Symmetric{T}, XexoZ::Matrix{T}, ZZ::Symmetric{T},
        nobs::Int, dof_fes::Int,
        endo_names::Vector{String}, k_exo::Int, has_intercept::Bool
) where {T}
    n = size(Xendo, 1)
    k_endo = size(Xendo, 2)
    k_z = size(ZZ, 1)

    # Compute residuals for first-stage F-stat: Xendo_res = Xendo - newZ * Pi
    # Pre-allocate and compute in one step
    Xendo_res = copy(Xendo)
    BLAS.gemm!('N', 'N', -one(T), newZ, Pi, one(T), Xendo_res)

    # Partial out Z w.r.t. Xexo: Z_res = Z - Xexo * Pi2
    XexoZ_aug = build_block_symmetric([XexoXexo.data, XexoZ, ZZ.data], [k_exo, k_z])
    Pi2 = ls_solve!(XexoZ_aug, k_exo)

    # Extract Z from newZ without allocating a new matrix
    # newZ = [Xexo, Z], so Z = newZ[:, (k_exo+1):end]
    # Compute Z_res = Z - Xexo * Pi2 using views where possible
    Xexo_view = view(newZ, :, 1:k_exo)
    Z_res = newZ[:, (k_exo + 1):end]  # This creates a copy (needed for gemm!)
    BLAS.gemm!('N', 'N', -one(T), Xexo_view, Pi2, one(T), Z_res)

    # Extract the relevant part of Pi (instruments only)
    Pip = Pi[(k_exo + 1):end, :]
    k_endo = size(Xendo, 2)

    # Compute joint first-stage F-statistic (Kleibergen-Paap)
    # HR1's factor n / dof_residual of the first-stage regression
    F_kp,
    p_kp = compute_first_stage_fstat(
        Xendo_res, Z_res, Pip,
        CovarianceMatrices.HR1(),
        nobs / max(1, nobs - size(newZ, 2) - dof_fes)
    )

    # Compute per-endogenous F-statistics
    F_kp_per_endo,
    p_kp_per_endo = compute_per_endogenous_fstats(
        Xendo_res, Z_res, Pip,
        CovarianceMatrices.HR1(), dof_fes;
        Xendo_orig = Xendo, newZ = newZ
    )

    # Store first-stage data - reuse existing matrices where possible
    # Pip, Xendo_res, Z_res are already computed and not needed elsewhere
    # Xendo and newZ are passed by reference, only copy if we must preserve them
    first_stage_data = FirstStageData{T}(
        Pip, Xendo_res, Z_res,
        endo_names, k_exo, Xendo, newZ, has_intercept
    )

    return (F_kp = F_kp, p_kp = p_kp,
        F_kp_per_endo = F_kp_per_endo, p_kp_per_endo = p_kp_per_endo,
        first_stage_data = first_stage_data)
end

"""
    _iv_first_stage_fstats_fe(...)

Compute first-stage F-statistics for FE-based IV.
"""
function _iv_first_stage_fstats_fe(
        Xendo::Matrix{T}, Xendo_hat::Matrix{T},
        iv_fes::Vector{FixedEffect},
        newZ::Matrix{T}, nobs::Int, k_exo::Int,
        endo_names::Vector{String}, has_intercept::Bool
) where {T}
    k_endo = size(Xendo, 2)
    Xendo_res = Xendo .- Xendo_hat
    n_iv_fe_groups = sum(nunique(fe) for fe in iv_fes; init = 0)

    F_kp_per_endo = T[]
    p_kp_per_endo = T[]

    for j in 1:k_endo
        rss_j = sum(abs2, Xendo_res[:, j])
        tss_j = sum(abs2, Xendo[:, j] .- mean(Xendo[:, j]))
        ess_j = tss_j - rss_j

        df1 = n_iv_fe_groups - 1
        df2 = nobs - n_iv_fe_groups - k_exo

        if df1 > 0 && df2 > 0
            F_j = (ess_j / df1) / (rss_j / df2)
            p_j = fdistccdf(df1, df2, F_j)
            push!(F_kp_per_endo, F_j)
            push!(p_kp_per_endo, p_j)
        else
            push!(F_kp_per_endo, T(NaN))
            push!(p_kp_per_endo, T(NaN))
        end
    end

    # Joint F
    if k_endo == 1
        F_kp, p_kp = F_kp_per_endo[1], p_kp_per_endo[1]
    else
        F_kp = k_endo / sum(1 ./ F_kp_per_endo)
        df1 = n_iv_fe_groups - 1
        df2 = nobs - n_iv_fe_groups - k_exo
        p_kp = fdistccdf(df1, df2, F_kp)
    end

    # Reuse existing matrices - Xendo_res is already a copy, Xendo and newZ can be shared
    first_stage_data = FirstStageData{T}(
        zeros(T, 0, k_endo), Xendo_res, zeros(T, nobs, 0),
        endo_names, k_exo, Xendo, newZ, has_intercept
    )

    return (F_kp = F_kp, p_kp = p_kp,
        F_kp_per_endo = F_kp_per_endo, p_kp_per_endo = p_kp_per_endo,
        first_stage_data = first_stage_data)
end

"""
    _iv_compute_inference(Xhat, residuals_raw, invXhatXhat, basis_coef, coef,
                          nobs, dof_model, dof_fes, dof_residual, has_intercept, T)

Compute vcov matrix, standard errors, t-stats, p-values, and robust F-stat.
"""
function _iv_compute_inference(
        Xhat::Matrix{T}, residuals_raw::Vector{T},
        invXhatXhat::Symmetric{T}, basis_coef::BitVector, coef::Vector{T},
        nobs::Int, dof_model::Int, dof_fes::Int, dof_residual::Int,
        has_intercept::Bool
) where {T}
    # Compute HC1 vcov
    vcov_matrix = compute_hc1_vcov_direct_iv(
        Xhat, residuals_raw, invXhatXhat, basis_coef,
        nobs, dof_model, dof_fes, dof_residual
    )

    # Standard errors
    se = sqrt.(diag(vcov_matrix))

    # t-stats and p-values
    coef_full = copy(coef)
    coef_full[.!basis_coef] .= zero(T)
    t_stats = coef_full ./ se
    p_values = 2 .* tdistccdf.(dof_residual, abs.(t_stats))

    # Robust F-statistic
    F_stat_robust,
    p_val_robust = compute_robust_fstat(coef_full, vcov_matrix, has_intercept, dof_residual)

    return (vcov_matrix = vcov_matrix, se = se, t_stats = t_stats,
        p_values = p_values, F_stat_robust = F_stat_robust,
        p_val_robust = p_val_robust)
end

# β = (X̂'X̂)⁻¹X̂'y and (X̂'X̂)⁻¹ from a pivoted QR of X̂, which works at the condition
# number of X̂ rather than of X̂'X̂. Throws if X̂ is rank deficient.
function _iv_second_stage_qr(Xhat::Matrix{T}, y::Vector{T}) where {T}
    k = size(Xhat, 2)
    F = _full_rank_qr(Xhat, "X̂'X is singular - check for perfect collinearity")
    Rinv = inv(UpperTriangular(F.R))
    p = F.p
    beta = Vector{T}(undef, k)
    beta[p] = Rinv * (F.Q' * y)[1:k]
    invXX = Matrix{T}(undef, k, k)
    invXX[p, p] = Rinv * Rinv'
    return beta, Symmetric(invXX)
end

##############################################################################
##
## TSLS kernel
##
##############################################################################

"""
    _iv_prepare!(y, Xexo, Xendo, Z, wts, fes; kwargs...) -> (prep, fe_solve)

The steps that the TSLS and k-class kernels share before estimation: check that the
data are finite, compute the total sum of squares, demean `y`, `Xexo`, `Xendo` and `Z`
on `fes` in place (throwing if the demeaning does not converge), scale them by `√w`, and
find collinear columns on the cross-products by the sweep operator. A column that the
earlier ones explain is dropped; an endogenous regressor that the instruments explain is
moved to the exogenous block, with an `@info`.

`prep` holds the reduced `Xexo`, `Xendo` and `Z`, their cross-products, the result of
`_iv_collinearity_detection!` (`coll`), the names of the remaining endogenous
regressors, and the sums of squares. `fe_solve` is as for [`_fit_ols_kernel`](@ref).
"""
function _iv_prepare!(y::Vector{T}, Xexo::Matrix{T}, Xendo::Matrix{T}, Z::Matrix{T},
        wts::AbstractWeights, fes::Vector{<:FixedEffect};
        response_name::Symbol, coefnames_exo::Vector{String},
        coefnames_endo::Vector{String}, coefnames_iv::Vector{String},
        has_intercept::Bool, has_fe_intercept::Bool, fe_tol::Real, fe_maxiter::Integer,
        method::Symbol, nthreads::Integer, progress_bar::Bool,
        save_fes::Bool) where {T <: AbstractFloat}
    _check_iv_finite(y, Xexo, Xendo, Z, wts)
    k_exo_in, k_endo_in = size(Xexo, 2), size(Xendo, 2)
    tss_total = tss(y, has_intercept | has_fe_intercept, wts)

    if isempty(fes)
        feM, iterations, converged, tss_partial, oldy, oldX = nothing, 0, true, tss_total,
        nothing, nothing
    else
        feM, iterations, converged, tss_partial, oldy, oldcols = partial_out_fixed_effects!(
            _fe_columns(y, Xexo, Xendo, Z),
            vcat(string(response_name), coefnames_exo, coefnames_endo, coefnames_iv),
            fes, wts, method, nthreads, fe_tol, fe_maxiter, progress_bar, save_fes,
            has_intercept, has_fe_intercept, T)
        converged || throw(ErrorException(
            "demeaning on the fixed effects did not converge in $iterations iterations; " *
            "increase fe_maxiter or fe_tol"))
        oldX = oldcols === nothing ? nothing : oldcols[:, 1:(k_exo_in + k_endo_in)]
    end

    if !(wts isa UnitWeights)
        sqrtw = sqrt.(wts)
        y .*= sqrtw
        Xexo .*= sqrtw
        Xendo .*= sqrtw
        Z .*= sqrtw
    end

    coll = _iv_collinearity_detection!(Xexo, Xendo, Z,
        compute_crossproduct(Xexo), compute_crossproduct(Xendo), compute_crossproduct(Z),
        has_intercept, coefnames_endo)
    endo_names = String[coefnames_endo[i]
                        for i in findall(coll.basis_endo)[coll.basis_endo_small]]
    ngroups_fes = Int[nunique(fe) for fe in fes]

    prep = (; Xexo = coll.Xexo, Xendo = coll.Xendo, Z = coll.Z,
        XexoXexo = coll.XexoXexo, ZZ = coll.ZZ, XexoZ = coll.XexoZ, coll, endo_names,
        tss_total, tss_partial, ngroups_fes, dof_fes = _fe_dof(fes))
    return prep, (; feM, iterations, converged, oldy, oldX)
end

# First-stage F statistics and first-stage data of an IV fit: per endogenous regressor
# (Kleibergen-Paap and iid) and joint (Kleibergen-Paap). Empty and NaN when
# `first_stage` is false or no regressor is endogenous.
function _iv_first_stage_stats(first_stage::Bool, Xendo::Matrix{T}, Xendo_hat, newZ,
        Pi, XexoZ, ZZ, prep, iv_fes, n, has_intercept) where {T}
    k_exo, k_endo = size(prep.Xexo, 2), size(Xendo, 2)
    if !(first_stage && k_endo > 0)
        return (; first_stage_data = empty_first_stage_data(T),
            F_first_stage_nonrobust = T[], p_first_stage_nonrobust = T[],
            F_first_stage_robust = T[], p_first_stage_robust = T[],
            F_kp = T(NaN), p_kp = T(NaN))
    end
    fstats = if isempty(iv_fes)
        _iv_first_stage_fstats_standard(Xendo, newZ, Pi, prep.XexoXexo, XexoZ, ZZ, n,
            prep.dof_fes, prep.endo_names, k_exo, has_intercept)
    else
        _iv_first_stage_fstats_fe(Xendo, Xendo_hat, iv_fes, newZ, n, k_exo,
            prep.endo_names, has_intercept)
    end
    F_nonrobust, p_nonrobust, _,
    _ = _compute_first_stage_f_iid(fstats.first_stage_data, n, prep.dof_fes)
    return (; fstats.first_stage_data,
        F_first_stage_nonrobust = F_nonrobust, p_first_stage_nonrobust = p_nonrobust,
        F_first_stage_robust = fstats.F_kp_per_endo,
        p_first_stage_robust = fstats.p_kp_per_endo,
        F_kp = T(fstats.F_kp), p_kp = T(fstats.p_kp))
end

# The coefficients of an IV fit in formula order, with NaN for dropped columns, and the
# column order that puts the estimated columns of `[Xexo, Xendo]` (after collinear columns
# were dropped and endogenous ones moved) back in formula order. `endo_cols` are the
# positions of the remaining endogenous columns in that order.
function _iv_formula_order(coll, coef_reduced, k_exo)
    basis_coef = coll.basis_coef
    coef = expand_coef_nan(coef_reduced, basis_coef)
    order = collect(eachindex(coef_reduced))
    if coll.perm !== nothing
        _invperm = invperm(coll.perm)
        coef = coef[_invperm]
        order = sortperm(coll.perm[findall(basis_coef)])
        basis_coef = basis_coef[_invperm]
    end
    endo_cols = sort(invperm(order)[(k_exo + 1):end])
    return coef, basis_coef, order, endo_cols
end

"""
    _fit_tsls_kernel(y, Xexo, Xendo, Z, wts, fes; kwargs...) -> (fit, fe_solve)

TSLS of `y` on `[Xexo, Xendo]` with instruments `[Xexo, Z]` and weights `wts`, absorbing
the fixed effects `fes`. Both `iv(TSLS(), df, formula)` and `iv(TSLS(), Z, X, y)`
estimate through this function, so equal data and settings give identical results.

The inputs hold the estimation sample, unweighted, and are modified in place; see
[`_iv_prepare!`](@ref) for the demeaning, weighting and collinearity steps. The first and
second stages are solved by QR. `iv_fes` are fixed effects used as excluded instruments.

`fit` holds the coefficients and every stored matrix in the column order of
`[Xexo, Xendo]`, whatever columns were moved or dropped; `fit.endo_cols` are the columns
of `fit.X` that remain endogenous, and the leading columns of `fit.Z` are the other
columns of `fit.X`, in order.
"""
function _fit_tsls_kernel(y::Vector{T}, Xexo::Matrix{T}, Xendo::Matrix{T}, Z::Matrix{T},
        wts::AbstractWeights, fes::Vector{<:FixedEffect};
        iv_fes::Vector{FixedEffect} = FixedEffect[], has_intercept::Bool,
        dof_add::Integer, first_stage::Bool, fe_tol::Real, fe_maxiter::Integer,
        method::Symbol, nthreads::Integer, progress_bar::Bool,
        kwargs...) where {T <: AbstractFloat}
    prep, fe_solve = _iv_prepare!(y, Xexo, Xendo, Z, wts, fes; has_intercept, fe_tol,
        fe_maxiter, method, nthreads, progress_bar, kwargs...)
    Xexo, Xendo, Z = prep.Xexo, prep.Xendo, prep.Z
    n = length(y)

    if isempty(iv_fes)
        size(Z, 2) >= size(Xendo, 2) ||
            throw(ArgumentError("Model not identified. There must be at least as many instruments as endogenous variables"))
        # `_iv_prepare!` dropped the collinear columns of [Xexo, Z], so the QR needs no
        # column pivoting.
        newZ = hcat(Xexo, Z)
        Pi = qr(newZ) \ Xendo
        Xendo_hat = newZ * Pi
        XexoZ, ZZ = prep.XexoZ, prep.ZZ
    else
        fs = _iv_first_stage_fe(Xexo, Xendo, iv_fes, wts, method, nthreads, fe_maxiter,
            fe_tol, progress_bar)
        newZ, Pi, Xendo_hat = fs.newZ::Matrix{T}, fs.Pi::Matrix{T}, fs.Xendo_hat::Matrix{T}
        XexoZ, ZZ = fs.XexoZ::Matrix{T}, fs.ZZ::Symmetric{T, Matrix{T}}
    end

    Xhat = hcat(Xexo, Xendo_hat)
    X = hcat(Xexo, Xendo)
    coef_reduced, invXhatXhat = _iv_second_stage_qr(Xhat, y)
    residuals = y - X * coef_reduced

    fs_stats = _iv_first_stage_stats(first_stage, Xendo, Xendo_hat, newZ, Pi, XexoZ, ZZ,
        prep, iv_fes, n, has_intercept)
    coef, basis_coef, order, endo_cols = _iv_formula_order(prep.coll, coef_reduced,
        size(Xexo, 2))
    if !issorted(order)
        Xhat, X = Xhat[:, order], X[:, order]
        invXhatXhat = Symmetric(Matrix(invXhatXhat)[order, order])
    end

    return _iv_kernel_inference(; coef, basis_coef, X, Xhat, bread = invXhatXhat,
        crossx = cholesky(Symmetric(Xhat' * Xhat)), y, Z = newZ, endo_cols, residuals,
        prep, dof_add, has_intercept, fs_stats, kappa = T(NaN)),
    fe_solve
end

# Degrees of freedom, sums of squares and HC1 inference of an IV fit whose stored
# matrices are in formula order; `Xhat` and `bread` are the moment regressors and the
# bread of the sandwich (`Adj` and `inv(A)` for the k-class estimators).
function _iv_kernel_inference(; coef::Vector{T}, basis_coef, X, Xhat, bread, crossx, y,
        Z, endo_cols, residuals, prep, dof_add, has_intercept, fs_stats,
        kappa) where {T}
    n = length(y)
    dof_model = size(X, 2)
    dof_residual = max(1, n - dof_model - prep.dof_fes - dof_add)
    rss = sum(abs2, residuals)
    r2_within = 1 - rss / prep.tss_partial
    inf = _iv_compute_inference(Xhat, residuals, bread, basis_coef, coef,
        n, dof_model, prep.dof_fes, dof_residual, has_intercept)
    return (; coef, basis_coef, X, Xhat, bread, crossx, y, Z, endo_cols, residuals,
        kappa, nobs = n, dof_model, prep.dof_fes, prep.ngroups_fes, dof_residual, rss,
        prep.tss_total, prep.tss_partial, r2_within = T(r2_within),
        vcov_matrix = inf.vcov_matrix, se = inf.se, t_stats = inf.t_stats,
        p_values = inf.p_values, F = T(inf.F_stat_robust), p = T(inf.p_val_robust),
        fs_stats...)
end

# Throws unless the response, regressors, instruments and weights of an IV fit are finite.
function _check_iv_finite(y, Xexo, Xendo, Z, wts)
    all(isfinite, wts) || throw(ArgumentError("Weights are not finite"))
    all(isfinite, y) ||
        throw(ArgumentError("Some observations for the dependent variable are not finite"))
    all(isfinite, Xexo) ||
        throw(ArgumentError("Some observations for the exogenous variables are not finite"))
    all(isfinite, Xendo) ||
        throw(ArgumentError("Some observations for the endogenous variables are not finite"))
    all(isfinite, Z) ||
        throw(ArgumentError("Some observations for the instrumental variables are not finite"))
    return nothing
end

##############################################################################
##
## Formula entry point
##
##############################################################################

"""
    fit_tsls(df, formula; kwargs...) -> IVEstimator{T}

Fit an instrumental variables model by TSLS from a table and a formula; see
[`iv`](@ref) for the keywords.
"""
function fit_tsls(@nospecialize(df),
        formula::FormulaTerm;
        contrasts::Dict = Dict{Symbol, Any}(),
        weights::Union{Symbol, Nothing} = nothing,
        save::Union{Bool, Symbol} = :residuals,
        save_cluster::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
        dof_add::Integer = 0,
        method::Symbol = :cpu,
        nthreads::Integer = method == :cpu ? Threads.nthreads() : 256,
        double_precision::Bool = method == :cpu,
        fe_tol::Union{Real, Nothing} = nothing,
        fe_maxiter::Union{Integer, Nothing} = nothing,
        tol::Union{Real, Nothing} = nothing,
        maxiter::Union{Integer, Nothing} = nothing,
        drop_singletons::Bool = true,
        progress_bar::Bool = true,
        subset::Union{Nothing, AbstractVector} = nothing,
        first_stage::Bool = true)
    fe_tol = _renamed_keyword(fe_tol, tol, :fe_tol, :tol, :iv, 1e-8)
    fe_maxiter = _renamed_keyword(fe_maxiter, maxiter, :fe_maxiter, :maxiter, :iv, 10000)
    prep = _iv_formula_data(df, formula; contrasts, weights, save, save_cluster, method,
        nthreads, double_precision, drop_singletons, subset)
    return _fit_tsls_formula(prep.y, prep.Xexo, prep.Xendo, prep.Z, prep.wts, prep.subfes;
        prep, fe_tol, fe_maxiter, method, nthreads = prep.nthreads, progress_bar,
        dof_add, first_stage)
end

# The formula TSLS fit from its model matrices onward, compiled for the concrete `T`.
function _fit_tsls_formula(y::Vector{T}, Xexo::Matrix{T}, Xendo::Matrix{T}, Z::Matrix{T},
        wts, subfes; prep, fe_tol, fe_maxiter, method, nthreads, progress_bar, dof_add,
        first_stage) where {T}
    data_prep = prep.data_prep
    fit, fe_solve = _fit_tsls_kernel(y, Xexo, Xendo, Z, wts, subfes;
        iv_fes = prep.iv_fes, _iv_kernel_names(prep)..., fe_tol, fe_maxiter, method,
        nthreads, progress_bar, has_intercept = data_prep.has_intercept,
        has_fe_intercept = data_prep.has_fe_intercept, save_fes = data_prep.save_fes,
        dof_add, first_stage)
    return _iv_formula_result(TSLS(), fit, fe_solve, prep, wts, subfes, fe_tol,
        fe_maxiter)
end

# The kernel keywords that name the columns of an IV formula fit.
function _iv_kernel_names(prep)
    return (; response_name = Symbol(prep.response_name), prep.coefnames_exo,
        prep.coefnames_endo, prep.coefnames_iv)
end

# The `IVEstimator` of a formula fit from the kernel's output.
function _iv_formula_result(estimator::AbstractIVEstimator, fit, fe_solve, prep, wts,
        subfes, fe_tol, fe_maxiter)
    data_prep = prep.data_prep
    T = eltype(fit.coef)
    kclass = !(estimator isa TSLS)
    postestimation_data = PostEstimationDataIV(
        fit.Xhat, fit.X, fit.y, fit.crossx, fit.bread, wts,
        prep.cluster_data, fit.basis_coef, fit.first_stage_data,
        kclass ? fit.Xhat : Matrix{T}(undef, 0, 0), fit.kappa, fit.Z,
        Symmetric(cholesky(Symmetric(fit.Z' * fit.Z)) \ I),
        Vector{Int}[fe.refs for fe in subfes], data_prep.fekeys, fit.ngroups_fes)

    return IVEstimator{
        T, typeof(estimator), typeof(CovarianceMatrices.HC1()),
        typeof(postestimation_data), typeof(fit.vcov_matrix)}(
        estimator, fit.coef,
        _iv_esample(prep), _iv_unweighted_residuals(fit.residuals, wts),
        prep.save_residuals,
        _iv_saved_fes(fit, fe_solve, prep, subfes, fe_tol, fe_maxiter),
        postestimation_data,
        data_prep.fekeys, prep.coef_names, prep.response_name,
        data_prep.formula_origin, prep.formula_schema, prep.contrasts,
        fit.nobs, fit.dof_model, fit.dof_fes, fit.dof_residual,
        fit.rss, fit.tss_total,
        fe_solve.iterations, fe_solve.converged, fit.r2_within,
        CovarianceMatrices.HC1(), fit.vcov_matrix, fit.se, fit.t_stats, fit.p_values,
        fit.F, fit.p,
        fit.F_first_stage_nonrobust, fit.p_first_stage_nonrobust,
        fit.F_first_stage_robust, fit.p_first_stage_robust,
        fit.F_kp, fit.p_kp)
end

# Residuals on the scale of the data, from those of the √w-scaled fit.
_iv_unweighted_residuals(r, wts) = wts isa UnitWeights ? r : r ./ sqrt.(wts)

# The estimation sample over the rows of the data, without the rows dropped for NaN.
function _iv_esample(prep)
    data_prep = prep.data_prep
    esample = data_prep.esample isa Colon ? trues(data_prep.nrows) :
              BitVector(data_prep.esample)
    if prep.has_nan_rows
        esample[findall(esample)[prep.nan_rows]] .= false
    end
    return esample
end

# Fixed-effect estimates, from the residual of the undemeaned fit, as a DataFrame over
# the rows of the data (empty unless the fixed effects are saved).
function _iv_saved_fes(fit, fe_solve, prep, subfes, fe_tol, fe_maxiter)
    data_prep = prep.data_prep
    augmentdf = DataFrame()
    (data_prep.save_fes && fe_solve.oldy !== nothing) || return augmentdf
    b = fit.basis_coef
    r = fe_solve.oldy - fe_solve.oldX[:, b] * fit.coef[b]
    newfes, _, _ = solve_coefficients!(r, fe_solve.feM; tol = fe_tol, maxiter = fe_maxiter)
    for fekey in data_prep.fekeys
        augmentdf[!, fekey] = prep.df[!, fekey]
    end
    for j in eachindex(subfes)
        augmentdf[!, data_prep.feids[j]] = Vector{Union{eltype(r), Missing}}(missing,
            data_prep.nrows)
        augmentdf[data_prep.esample, data_prep.feids[j]] = newfes[j]
    end
    return augmentdf
end

"""
    _iv_formula_data(df, formula; kwargs...) -> NamedTuple

The estimation sample, weights, fixed effects and model matrices of an IV formula fit
(shared by TSLS and the k-class estimators), with rows that hold NaN dropped.
"""
function _iv_formula_data(@nospecialize(df), formula::FormulaTerm; contrasts, weights,
        save, save_cluster, method, nthreads, double_precision, drop_singletons, subset)
    save, save_residuals = validate_save_keyword(save)
    nthreads = validate_nthreads(method, nthreads)
    contrasts = convert(Dict{Symbol, Any}, contrasts)
    df = DataFrame(df; copycols = false)
    T = double_precision ? Float64 : Float32

    data_prep = prepare_data(df, formula, weights, subset, save, drop_singletons, nthreads)
    cluster_data = extract_cluster_variables(df, data_prep.fe_vars, save_cluster,
        data_prep.esample)
    wts = data_prep.has_weights ?
          Weights(convert(Vector{T}, disallowmissing(view(df[!, weights], data_prep.esample)))) :
          uweights(data_prep.nobs)
    subdf = _create_subdf(df, data_prep.all_vars, data_prep.esample)
    subfes = isempty(data_prep.fes) ? FixedEffect[] :
             FixedEffect[fe[data_prep.esample] for fe in data_prep.fes]

    s = schema(data_prep.formula, subdf, contrasts)
    formula_schema = apply_schema(data_prep.formula, s, IVEstimator, data_prep.has_fe_intercept)
    y_raw = response(formula_schema, subdf)
    y_raw isa AbstractVector || throw(ArgumentError(
        "the response variable must be numeric (got a matrix — " *
        "check that the dependent variable column is not a String or categorical type)"))
    y = convert(Vector{T}, y_raw)
    Xexo = convert(Matrix{T}, modelmatrix(formula_schema, subdf))
    response_name, coefnames_exo = coefnames(formula_schema)

    formula_endo_schema = apply_schema(data_prep.formula_endo,
        schema(data_prep.formula_endo, subdf, contrasts), StatisticalModel)
    Xendo = convert(Matrix{T}, modelmatrix(formula_endo_schema, subdf))
    _, coefnames_endo = coefnames(formula_endo_schema)

    if data_prep.formula_iv != FormulaTerm(ConstantTerm(0), ConstantTerm(0))
        formula_iv_schema = apply_schema(data_prep.formula_iv,
            schema(data_prep.formula_iv, subdf, contrasts), StatisticalModel)
        Z = convert(Matrix{T}, modelmatrix(formula_iv_schema, subdf))
        _, coefnames_iv = coefnames(formula_iv_schema)
    else
        Z = Matrix{T}(undef, data_prep.nobs, 0)
        coefnames_iv = String[]
    end

    iv_fes = FixedEffect[]
    if data_prep.has_fe_iv
        iv_fes, _, _ = parse_fixedeffect(subdf, data_prep.formula_iv_fe)
    end

    # The schema used by predict covers the exogenous and the endogenous regressors.
    schema_terms = collect(AbstractTerm, eachterm(formula_schema.rhs))
    for term in eachterm(formula_endo_schema.rhs)
        term != ConstantTerm(0) && push!(schema_terms, term)
    end
    formula_schema = FormulaTerm(formula_schema.lhs, MatrixTerm(Tuple(schema_terms)))

    y, (Xexo, Xendo, Z), wts, subfes, _,
    has_nan_rows, nan_rows = _filter_nan_rows(y, (Xexo, Xendo, Z), wts, subfes)

    _strings(v) = String[string(x) for x in _vec(v)]
    return (; df, data_prep, cluster_data, contrasts, formula_schema, save_residuals,
        nthreads, y, Xexo, Xendo, Z, wts, subfes, iv_fes, has_nan_rows, nan_rows,
        response_name, coefnames_exo = _strings(coefnames_exo),
        coefnames_endo = _strings(coefnames_endo), coefnames_iv = _strings(coefnames_iv),
        coef_names = vcat(_strings(coefnames_exo), _strings(coefnames_endo)))
end

# Coefficient names come back from StatsModels as a single name or a vector of them.
_vec(v::AbstractVector) = v
_vec(v) = [v]
