@testitem "matrix OLS - accessors and factorizations" tags = [:ols, :matrix] begin
    using Regress, StatsBase

    n = 200
    t = range(0, 1; length = n)
    X = hcat(ones(n), cos.(2π .* t), sin.(2π .* t))
    beta = [1.0, 2.0, -1.5]
    noise = 0.05 .* cos.(7π .* t)   # deterministic, not collinear with X
    y = X * beta .+ noise

    ref = Regress.ols(X, y)

    # The matrix API accepts :auto/:chol/:qr (not :sweep, which is formula-only).
    for fac in (:auto, :chol, :qr)
        m = Regress.ols(X, y; factorization = fac)
        @test coef(m) ≈ coef(ref)
        @test coef(m) ≈ beta atol = 0.02
        @test length(residuals(m)) == n
        @test length(fitted(m)) == n
        @test fitted(m) .+ residuals(m) ≈ y
        @test size(modelmatrix(m)) == (n, 3)
        @test nobs(m) == n
        @test dof(m) == 3
        @test dof_residual(m) == n - 3
        @test response(m) === y
        @test 0.0 < r2(m) < 1.0
        @test isfinite(adjr2(m))
        @test all(isfinite, vcov(m))
        @test vcov(m) ≈ vcov(ref) rtol = 1e-6
    end
end

@testitem "matrix OLS - vcov re-specification via + operator" tags = [:ols, :matrix] begin
    using Regress, StatsBase
    using CovarianceMatrices

    n = 200
    t = range(0, 1; length = n)
    X = hcat(ones(n), cos.(2π .* t), sin.(2π .* t))
    y = X * [0.5, 1.0, -0.3] .+ 0.1 .* (t .- 0.5)
    m = Regress.ols(X, y)

    m3 = m + vcov(HC3())
    @test stderror(m3) ≈ stderror(HC3(), m)
    @test stderror(m3) != stderror(m)          # HC1 default differs from HC3
    @test coef(m3) == coef(m)                   # coefficients unchanged
    @test size(confint(m3)) == (3, 2)
end

@testitem "matrix OLS - confint variants" tags = [:ols, :matrix] begin
    using Regress, StatsBase
    using CovarianceMatrices

    n = 200
    t = range(0, 1; length = n)
    X = hcat(ones(n), cos.(2π .* t), sin.(2π .* t))
    y = X * [0.5, 1.0, -0.3] .+ 0.1 .* (t .- 0.5)
    m = Regress.ols(X, y)

    ci_default = confint(m)
    ci_hc3 = confint(HC3(), m)
    ci_90 = confint(m; level = 0.90)

    @test size(ci_default) == (3, 2)
    @test size(ci_hc3) == (3, 2)
    @test size(ci_90) == (3, 2)
    # A 90% interval is narrower than the default 95% interval.
    @test all((ci_90[:, 2] .- ci_90[:, 1]) .< (ci_default[:, 2] .- ci_default[:, 1]))
end

@testitem "matrix OLS - collinear columns" tags = [:ols, :matrix] begin
    using Regress, StatsBase

    n = 200
    t = range(0, 1; length = n)
    x1 = cos.(2π .* t)
    x2 = sin.(2π .* t)
    X = hcat(ones(n), x1, x2, x2)   # last column duplicates x2
    y = X[:, 1:3] * [1.0, 2.0, -0.5]

    m = Regress.ols(X, y)
    c = coef(m)
    @test any(==(0.0), c)             # collinear column coefficient set to zero
    @test any(isnan, stderror(m))     # its standard error is NaN
end

@testitem "OLS save=:minimal clears data" tags = [:ols] begin
    using Regress, DataFrames, CSV, StatsBase

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    m = Regress.ols(df, @formula(Sales ~ NDI + CPI); save = :minimal)
    @test !Regress.has_matrices(m)
    # Coefficients and standard errors survive :minimal.
    @test all(isfinite, coef(m))
    @test all(isfinite, stderror(m))
    # Data-dependent accessors report the data is gone.
    @test_throws "not stored" response(m)
    @test_throws "not stored" fitted(m)
    @test_throws "not stored" modelmatrix(m)
end

@testitem "matrix models - cluster vector length must match the rows" tags = [
    :ols, :iv, :matrix, :vcov] begin
    using Regress, StatsBase
    using CovarianceMatrices

    n = 200
    t = range(0, 1; length = n)
    z = cos.(5π .* t)
    x = 0.8 .* z .+ 0.1 .* sin.(3π .* t)
    X = hcat(ones(n), x)
    y = X * [0.5, 1.0] .+ 0.1 .* sin.(11π .* t)
    Z = hcat(ones(n), z)
    mols = Regress.ols(X, y)
    miv = Regress.iv(Regress.TSLS(), Z, X, y; has_intercept = true, n_endogenous = 1)

    for m in (mols, miv), len in (n - 1, n + 1), E in (CR0, CR1, CR2, CR3)
        k = E(repeat(1:10; outer = cld(len, 10))[1:len])
        msg = Regex("cluster variables? ha(s|ve) $len observations.* $n")
        @test_throws msg vcov(k, m)
        @test_throws msg stderror(k, m)
        @test_throws msg m + vcov(k)
    end
    # The single-cluster CR0/CR1 path of matrix OLS aggregates without a moment matrix.
    @test_throws DimensionMismatch aVar(CR1(rand(1:10, n - 1)), mols)
    # Several cluster dimensions
    g1, g2 = repeat(1:10; outer = 20), repeat(1:20; inner = 10)
    for m in (mols, miv)
        @test_throws DimensionMismatch vcov(CR1(g1[1:(n - 1)], g2[1:(n - 1)]), m)
    end
end

@testitem "matrix models - coefficient names" tags = [:ols, :iv, :matrix] begin
    using Regress, StatsBase
    using CovarianceMatrices
    using Regress: first_stage

    n = 200
    t = range(0, 1; length = n)
    w = cos.(2π .* t)
    z = sin.(2π .* t)
    v = 0.3 .* cos.(5π .* t)
    x = 0.8 .* z .+ v
    y = 1.0 .+ 2.0 .* x .+ 0.5 .* w .+ 0.4 .* v .+ 0.05 .* sin.(9π .* t)

    # OLS: default names, given names, names survive `+ vcov`
    X = hcat(ones(n), w, x)
    m0 = Regress.ols(X, y)
    @test coefnames(m0) == ["x1", "x2", "x3"]
    m = Regress.ols(X, y; coefnames = ["(Intercept)", "w", "x"])
    @test coefnames(m) == ["(Intercept)", "w", "x"]
    @test coef(m) == coef(m0)
    @test stderror(m) == stderror(m0)
    @test coefnames(m + vcov(HC3())) == coefnames(m)
    @test Regress.ols(X, y; coefnames = [:a, :b, :c]) |> coefnames == ["a", "b", "c"]
    ct = coeftable(m)
    @test ct.rownms == ["w", "x", "(Intercept)"]   # intercept is listed last
    @test occursin("w", sprint(show, ct))
    @test_throws "coefnames has 2 names but the design matrix has 3 columns" Regress.ols(
        X, y; coefnames = ["a", "b"])

    # IV: names on the model, `+ vcov`, coeftable, show and the first stage
    Z = hcat(ones(n), w, z)
    i0 = Regress.iv(Regress.TSLS(), Z, X, y)
    @test coefnames(i0) == ["x1", "x2", "x3"]
    mi = Regress.iv(Regress.TSLS(), Z, X, y; coefnames = ["(Intercept)", "w", "x"])
    @test coefnames(mi) == ["(Intercept)", "w", "x"]
    @test coef(mi) == coef(i0)
    @test coefnames(mi + vcov(HC3())) == coefnames(mi)
    @test coeftable(mi).rownms == ["w", "x", "(Intercept)"]
    @test occursin("x = ", sprint(show, MIME"text/plain"(), mi))
    @test_throws DimensionMismatch Regress.iv(Regress.TSLS(), Z, X, y; coefnames = ["a"])

    fs = first_stage(mi)
    @test fs.endogenous_names == ["x"]
    @test coefnames(fs.models[1]) == ["(Intercept)", "w", "inst_1"]
    fs_z = first_stage(mi; instrument_names = ["z"])
    @test coefnames(fs_z.models[1]) == ["(Intercept)", "w", "z"]
    @test coef(fs_z.models[1]) == coef(fs.models[1])
    @test fs_z.F_nonrobust == fs.F_nonrobust
    @test first_stage(i0).endogenous_names == ["x3"]
    @test_throws "instrument_names has 2 names but the model has 1 excluded instruments" first_stage(
        mi; instrument_names = ["z", "q"])
    @test_throws "endogenous_names has 2 names but the model has 1 endogenous variables" first_stage(
        mi; endogenous_names = ["x", "q"])
end

@testitem "matrix models - absorbed fixed effects" tags = [:ols, :iv, :matrix, :fe] begin
    using Regress, DataFrames, StableRNGs
    using Regress: fe

    rng = StableRNG(1)
    G, T = 40, 15
    n = G * T
    id = repeat(1:G, inner = T)
    t = repeat(1:T, outer = G)
    pair = (id .+ 1) .÷ 2   # id is nested in pair; t is not
    x = randn(rng, n) .+ 0.3 .* id ./ G
    w = randn(rng, n)
    z = x .+ randn(rng, n)
    y = x .- 0.5 .* w .+ randn(rng, G)[id] .+ randn(rng, n)
    wt = 0.5 .+ rand(rng, n)
    df = DataFrame(; y, x, w, z, id, t, pair, wt)
    opts = (; tol = 1e-12, drop_singletons = false, save_cluster = [:id, :pair, :t])
    xcopy = copy(x)

    # OLS, one fixed effect: coefficients, HC1, R², and CR1 with the fixed effect
    # nested in the cluster (id, pair) and not nested (t)
    mf = Regress.ols(df, @formula(y ~ x + w + fe(id)); opts...)
    mm = Regress.ols([x w], y; fe = id, coefnames = ["x", "w"])
    @test x == xcopy
    @test coef(mm) ≈ coef(mf)
    @test stderror(mm) ≈ stderror(mf)
    @test dof_residual(mm) == dof_residual(mf) == n - 2 - G
    @test r2(mm) ≈ r2(mf)
    @test adjr2(mm) ≈ adjr2(mf)
    for c in (:id, :pair, :t)
        @test stderror(mm + vcov(CR1(df[!, c]))) ≈ stderror(mf + vcov(CR1(c)))
    end

    # Two fixed effects, and weights
    mf2 = Regress.ols(df, @formula(y ~ x + w + fe(id) + fe(t)); opts...)
    mm2 = Regress.ols([x w], y; fe = (id, t))
    @test coef(mm2) ≈ coef(mf2)
    @test stderror(mm2) ≈ stderror(mf2)
    @test dof_residual(mm2) == dof_residual(mf2)
    @test stderror(mm2 + vcov(CR1(pair))) ≈ stderror(mf2 + vcov(CR1(:pair)))
    mfw = Regress.ols(df, @formula(y ~ x + w + fe(id)); weights = :wt, opts...)
    mmw = Regress.ols([x w], y; fe = id, weights = wt)
    @test coef(mmw) ≈ coef(mfw)
    @test stderror(mmw) ≈ stderror(mfw)

    # TSLS, with first-stage and weak-instrument statistics
    mfi = Regress.iv(TSLS(), df, @formula(y ~ w + (x ~ z) + fe(id)); opts...)
    mmi = Regress.iv(TSLS(), [w z], [w x], y; fe = id, coefnames = ["w", "x"])
    @test coef(mmi) ≈ coef(mfi)
    @test stderror(mmi) ≈ stderror(mfi)
    @test dof_residual(mmi) == dof_residual(mfi)
    for c in (:id, :pair, :t)
        mmc = mmi + vcov(CR1(df[!, c]))
        mfc = mfi + vcov(CR1(c))
        @test stderror(mmc) ≈ stderror(mfc)
        @test first_stage(mmc).F_robust ≈ first_stage(mfc).F_robust
        @test weakivtest(mmc).F_eff ≈ weakivtest(mfc).F_eff
    end
    fsm, fsf = first_stage(mmi), first_stage(mfi)
    @test fsm.F_nonrobust ≈ fsf.F_nonrobust
    @test fsm.df2 == fsf.df2

    # Without `fe`, nothing is absorbed
    @test dof_residual(Regress.ols([ones(n) x w], y)) == n - 3

    @test_throws "the fixed effects absorb the intercept" Regress.ols(
        [x w], y; fe = id, has_intercept = true)
    @test_throws "fixed effect 1 has 10 observations but y has $n" Regress.ols(
        [x w], y; fe = id[1:10])
    @test_throws "fixed effect 1 has missing values" Regress.ols(
        [x w], y; fe = Union{Int, Missing}[id[1:(end - 1)]; missing])
    @test_throws "did not converge" Regress.ols(
        [x w], y; fe = (id, pair .+ t), fe_maxiter = 1, fe_tol = 1e-30)
end

@testitem "matrix models - k-class IV" tags = [:iv, :kclass, :matrix, :fe] begin
    using Regress, DataFrames, StableRNGs
    using Regress: fe

    rng = StableRNG(1)
    G, T = 40, 15
    n = G * T
    id = repeat(1:G, inner = T)
    t = repeat(1:T, outer = G)
    z1, z2, w, u = randn(rng, n), randn(rng, n), randn(rng, n), randn(rng, n)
    x = 0.5 .* z1 .+ 0.3 .* z2 .+ 0.5 .* u .+ 0.2 .* w
    y = 1 .+ 2 .* x .- 0.5 .* w .+ u .+ randn(rng, G)[id]
    df = DataFrame(; y, x, w, z1, z2, id, t)

    # Fuller(1) on this design has kappa just below 1, and KClass(0.8) is below 1; both warn
    for est in (LIML(), Fuller(1.0), KClass(0.8))
        fit(f) = est isa LIML ? (@test_logs f()) : (@test_logs (:warn, r"kappa < 1") f())
        mf = fit(() -> Regress.iv(est, df, @formula(y ~ w + (x ~ z1 + z2));
            save_cluster = [:id]))
        mm = fit(() -> Regress.iv(est, [ones(n) w z1 z2], [ones(n) w x], y;
            coefnames = ["(Intercept)", "w", "x"]))
        @test mm.estimator == est
        @test coef(mm) ≈ coef(mf)
        @test stderror(mm) ≈ stderror(mf)
        @test r2(mm) ≈ r2(mf)
        @test stderror(mm + vcov(CR1(id))) ≈ stderror(mf + vcov(CR1(:id)))
        @test stderror(mm + vcov(HC3())) ≈ stderror(mf + vcov(HC3()))
        @test stderror(mm + vcov(Bartlett(3))) ≈ stderror(mf + vcov(Bartlett(3)))
        @test first_stage(mm).F_robust ≈ first_stage(mf).F_robust
        @test weakivtest(mm).F_eff ≈ weakivtest(mf).F_eff

        # Absorbed fixed effects
        opts = (; tol = 1e-12, drop_singletons = false, save_cluster = [:t])
        mfe = fit(() -> Regress.iv(est, df, @formula(y ~ w + (x ~ z1 + z2) + fe(id));
            opts...))
        mme = fit(() -> Regress.iv(est, [w z1 z2], [w x], y; fe = id))
        @test coef(mme) ≈ coef(mfe)
        @test stderror(mme) ≈ stderror(mfe)
        @test dof_residual(mme) == dof_residual(mfe)
        @test stderror(mme + vcov(CR1(t))) ≈ stderror(mfe + vcov(CR1(:t)))
    end

    # Just identified: LIML is TSLS
    liml = Regress.iv(LIML(), [ones(n) w z1], [ones(n) w x], y)
    tsls = Regress.iv(TSLS(), [ones(n) w z1], [ones(n) w x], y)
    @test coef(liml) ≈ coef(tsls)
    @test stderror(liml) ≈ stderror(tsls)

    # The k-class fits need the exogenous columns first in both Z and X
    @test_throws "LIML needs Z = [Xexo, Zinstr] and X = [Xexo, Xendo] with the same first 2" Regress.iv(
        LIML(), [w ones(n) z1], [ones(n) w x], y)
end
