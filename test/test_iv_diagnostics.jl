@testitem "IV diagnostics: first_stage_f" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using Regress: fe

    csv_path = joinpath(@__DIR__, "data", "basic_validation_df.csv")
    df = CSV.read(csv_path, DataFrame)
    df.a = categorical(df.a)

    # m11: y ~ (x ~ z1 + z2), no FE — 1 endo, overidentified
    m11 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2)))
    fs11 = first_stage_f(m11)
    @test fs11.F_per_endo[1] ≈ 69.71611 atol = 0.01
    @test fs11.df1 == 2
    @test fs11.df2 == 497

    # m21: y ~ (x ~ z1 + z2) + fe(a), FE — 1 endo, overidentified
    m21 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2) + fe(a)))
    fs21 = first_stage_f(m21)
    @test fs21.F_per_endo[1] ≈ 60.65408 atol = 0.01
    @test fs21.df1 == 2
    @test fs21.df2 == 398

    # m12: y ~ (x + x2 ~ z1 + z2), no FE — 2 endo, exactly identified
    m12 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x + x2 ~ z1 + z2)))
    fs12 = first_stage_f(m12)
    @test fs12.F_per_endo[1] ≈ 69.71611 atol = 0.01  # x
    @test fs12.F_per_endo[2] ≈ 0.95367 atol = 0.01   # x2
    @test fs12.df1 == 2
    @test fs12.df2 == 497

    # m22: y ~ (x + x2 ~ z1 + z2) + fe(a), FE — 2 endo, exactly identified
    m22 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x + x2 ~ z1 + z2) + fe(a)))
    fs22 = first_stage_f(m22)
    @test fs22.F_per_endo[1] ≈ 60.65408 atol = 0.01  # x
    @test fs22.F_per_endo[2] ≈ 0.43385 atol = 0.01   # x2
    @test fs22.df1 == 2
    @test fs22.df2 == 398
end

@testitem "IV diagnostics: wu_hausman" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using Regress: fe

    csv_path = joinpath(@__DIR__, "data", "basic_validation_df.csv")
    df = CSV.read(csv_path, DataFrame)
    df.a = categorical(df.a)

    # m11: 1 endo, no FE
    m11 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2)))
    wh11 = wu_hausman(m11)
    @test wh11.stat ≈ 15.03388 atol = 0.01
    @test wh11.p ≈ 1.198e-4 atol = 1e-5
    @test wh11.df1 == 1
    @test wh11.df2 == 497

    # m21: 1 endo, FE
    m21 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2) + fe(a)))
    wh21 = wu_hausman(m21)
    @test wh21.stat ≈ 8.29012 atol = 0.01
    @test wh21.p ≈ 0.004201 atol = 1e-4
    @test wh21.df1 == 1
    @test wh21.df2 == 398

    # m12: 2 endo, no FE
    m12 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x + x2 ~ z1 + z2)))
    wh12 = wu_hausman(m12)
    @test wh12.stat ≈ 7.56721 atol = 0.01
    @test wh12.p ≈ 5.792e-4 atol = 1e-4
    @test wh12.df1 == 2
    @test wh12.df2 == 495

    # m22: 2 endo, FE
    m22 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x + x2 ~ z1 + z2) + fe(a)))
    wh22 = wu_hausman(m22)
    @test wh22.stat ≈ 4.37336 atol = 0.01
    @test wh22.p ≈ 0.013223 atol = 1e-3
    @test wh22.df1 == 2
    @test wh22.df2 == 396
end

@testitem "IV diagnostics: sargan" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using Regress: fe

    csv_path = joinpath(@__DIR__, "data", "basic_validation_df.csv")
    df = CSV.read(csv_path, DataFrame)
    df.a = categorical(df.a)

    # m11: 1 endo, 2 instruments — overidentified
    m11 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2)))
    s11 = sargan(m11)
    @test s11.stat ≈ 0.19677 atol = 0.01
    @test s11.p ≈ 0.657342 atol = 0.01
    @test s11.df == 1

    # m21: 1 endo, 2 instruments, FE — overidentified
    m21 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2) + fe(a)))
    s21 = sargan(m21)
    @test s21.stat ≈ 0.53469 atol = 0.01
    @test s21.p ≈ 0.464644 atol = 0.01
    @test s21.df == 1

    # m12: 2 endo, 2 instruments — exactly identified, should error
    m12 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x + x2 ~ z1 + z2)))
    @test_throws ArgumentError sargan(m12)
    @test_throws "Sargan test requires overidentification" sargan(m12)

    # m22: 2 endo, 2 instruments, FE — exactly identified, should error
    m22 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x + x2 ~ z1 + z2) + fe(a)))
    @test_throws ArgumentError sargan(m22)
    @test_throws "Sargan test requires overidentification" sargan(m22)
end

@testitem "AbstractTest interface" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV
    using Regress: AbstractTest, first_stage_f, first_stage, first_stage_F_iid,
                   first_stage_F_robust, first_stage_F_KP, wu_hausman, sargan,
                   weakivtest
    using StatsAPI: pvalue, dof

    csv_path = joinpath(@__DIR__, "data", "basic_validation_df.csv")
    df = CSV.read(csv_path, DataFrame)

    # Overidentified single-endogenous model so every diagnostic is available.
    m = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2)))

    # Every statistical test-result type subtypes AbstractTest.
    for t in (first_stage_f(m), first_stage(m), first_stage_F_iid(m),
        first_stage_F_robust(m), first_stage_F_KP(m), wu_hausman(m),
        sargan(m), weakivtest(m))
        @test t isa AbstractTest
    end

    # Single-statistic tests report a scalar p-value equal to their `p` field.
    @test pvalue(wu_hausman(m)) == wu_hausman(m).p
    @test pvalue(sargan(m)) == sargan(m).p

    # Per-endogenous first-stage tests report a vector aligned with the endogenous
    # variables; the composite results report their robust p-values.
    @test pvalue(first_stage_f(m)) == first_stage_f(m).p_per_endo
    @test pvalue(first_stage(m)) == first_stage(m).p_robust
    @test pvalue(first_stage_F_iid(m)) == first_stage_F_iid(m).p
    @test pvalue(first_stage_F_robust(m)) == first_stage_F_robust(m).p

    # WeakIVTestResult reports against critical values, so it has no p-value.
    @test !hasmethod(pvalue, Tuple{typeof(weakivtest(m))})

    # `dof` is a scalar only where unambiguous (Sargan's χ² df); F-based tests
    # carry df1/df2 instead and implement no scalar `dof`.
    @test dof(sargan(m)) == sargan(m).df
    @test !hasmethod(dof, Tuple{typeof(wu_hausman(m))})
    @test !hasmethod(dof, Tuple{typeof(first_stage_F_iid(m))})

    # Matrix-path first-stage result also participates in the interface.
    mm = Regress.iv(Regress.TSLS(), Matrix(df[:, [:z1, :z2]]),
        reshape(df.x, :, 1), df.y; n_endogenous = 1)
    fsi = first_stage(mm)
    @test fsi isa AbstractTest
    @test pvalue(fsi) == fsi.p_robust
end

@testitem "Kleibergen-Paap statistic follows the variance estimator" tags = [
    :iv, :validation] begin
    using Regress
    using DataFrames, CSV, LinearAlgebra
    using Regress: fe

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    n = nrow(df)
    kp(m, k) = first_stage_F_KP(m + vcov(k))

    # With one endogenous regressor the KP rk Wald F is the first-stage Wald F of the
    # excluded instruments. References: R fixest 0.14.1
    # `fitstat(feols(Sales ~ NDI | Price ~ Pimin + Pop, df, vcov = v), ~ivwald1)`.
    m = Regress.iv(Regress.TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin + Pop)),
        save_cluster = [:State, :Year])
    for (k, ref) in ((Uncorrelated(), 2622.461860257), (HC1(), 1464.27486829565),
        (CR1(:State), 280.848595798773), (CR1(:State, :Year), 280.440867252914))
        fs = kp(m, k)
        @test fs.stat ≈ ref rtol = 1e-10
        @test fs.vcov_estimator == k
        @test fs.p ≈ Regress.chisqccdf(2, 2 * fs.stat)
    end
    # The statistic computed at fit time is the HC1 one.
    @test first_stage_F_KP(m).stat ≈ kp(m, HC1()).stat rtol = 1e-12
    @test first_stage_F_KP(m).vcov_estimator == HC1()
    # LIML shares the first stage.
    ml = Regress.iv(Regress.LIML(), df, @formula(Sales ~ NDI + (Price ~ Pimin + Pop)), save_cluster = :State)
    @test kp(ml, CR1(:State)).stat ≈ 280.848595798773 rtol = 1e-10
    # The HAC kernels and CachedCR equal the robust first-stage F, computed from the full
    # first-stage regression rather than from the partialled-out moments.
    for k in (Bartlett(3), Parzen(5), CachedCR(CR1(:State), m))
        @test kp(m, k).stat ≈ first_stage_F_robust(m + vcov(k)).stat rtol = 1e-10
    end

    # Absorbed fixed effect, with clusters nested in it (State) and not (Year).
    # References: fixest `feols(Sales ~ NDI | State | Price ~ Pimin + Pop, ...)`.
    mf = Regress.iv(
        Regress.TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin + Pop) + fe(State)),
        save_cluster = [:State, :Year])
    for (k, ref) in ((Uncorrelated(), 2776.23219867604), (HC1(), 1435.11064681269),
        (CR1(:State), 332.781459426826), (CR1(:Year), 1146.43018399374))
        @test kp(mf, k).stat ≈ ref rtol = 1e-10
        # The robust first-stage F counts the absorbed fixed effect in its cluster K too.
        @test only(first_stage_F_robust(mf + vcov(k)).stat) ≈ ref rtol = 1e-10
    end
    for k in (CR0(:Year), Bartlett(3))
        @test only(first_stage_F_robust(mf + vcov(k)).stat) ≈ kp(mf, k).stat rtol = 1e-10
    end

    # Two endogenous regressors, by hand: θ = F Π G⁻¹ with F'F = Z̃'Z̃ and G'G = V'V, the
    # instruments Z̃ and first-stage residuals V with the intercept partialled out, and
    # the KP statistic for rank(Π) = 1 with meat S of the moments Z̃ ⊗ V.
    m2 = Regress.iv(Regress.TSLS(), df, @formula(Sales ~ (Price + NDI ~ Pimin + Pop)),
        save_cluster = [:State, :Year])
    Z = Matrix(df[:, [:Pimin, :Pop]])
    X = Matrix(df[:, [:Price, :NDI]])
    W = [ones(n) Z]
    V = X - W * (W \ X)
    Zt = Z .- sum(Z; dims = 1) ./ n
    Π = Zt \ X
    function kp_by_hand(S)
        F = cholesky(Symmetric(Zt' * Zt)).U
        G = cholesky(Symmetric(V' * V)).U
        θ = F * Π / G
        U, _, Vθ = svd(θ; full = true)
        Vt = Vθ'
        u = U[2:2, 2:2]
        a = U[:, 2:2] * (u \ sqrt(u * u'))
        b = sqrt(Vt[2, 2]^2) * (Vt[2, 2] \ Vt[1:2, 2]')
        L = kron(b, a')
        λ = L * vec(θ)
        K = kron(G, F)'
        Σ = L * (K \ (K \ S)') * L'
        return only(λ' * (Σ \ λ)) / 2
    end
    M = hcat(Zt .* V[:, 1], Zt .* V[:, 2])
    cluster_meat(g) = (s = [sum(M[g .== c, :]; dims = 1) for c in unique(g)];
        sum(si' * si for si in s))
    K = 3  # first-stage coefficients: intercept, Pimin, Pop
    G = length(unique(df.State))
    S1 = G / (G - 1) * (n - 1) / (n - K) * cluster_meat(df.State)
    @test kp(m2, CR1(:State)).stat ≈ kp_by_hand(S1) rtol = 1e-10
    Gmin = min(G, length(unique(df.Year)))
    S2 = Gmin / (Gmin - 1) * (n - 1) / (n - K) *
         (cluster_meat(df.State) + cluster_meat(df.Year) -
          cluster_meat(string.(df.State, "-", df.Year)))
    @test kp(m2, CR1(:State, :Year)).stat ≈ kp_by_hand(S2) rtol = 1e-10
    @test kp(m2, HC1()).stat ≈ kp_by_hand(n / (n - K) * (M' * M)) rtol = 1e-10
    # fixest `fitstat(feols(Sales ~ 1 | Price + NDI ~ Pimin + Pop, df, vcov = "hetero"),
    # ~kpr)` applies n/(n−K) twice: once in the meat and again to the variance.
    @test kp(m2, HC1()).stat ≈ 16.872201216016 * n / (n - K) rtol = 1e-10

    # Estimators for which the statistic is not defined: `+` succeeds, and asking for the
    # statistic throws.
    for k in (HC3(), CR2(:State), DriscollKraay(Bartlett(3), df.Year, df.State))
        mk = m + vcov(k)
        @test isnan(mk.F_kp)
        @test_throws "The Kleibergen-Paap statistic is not defined for $(nameof(typeof(k)))" first_stage_F_KP(mk)
    end
end

@testitem "First-stage statistics propagate unexpected errors" tags = [:iv] begin
    using Regress
    using DataFrames, CSV, CovarianceMatrices

    struct NotAnEstimator <: CovarianceMatrices.AbstractAsymptoticVarianceEstimator end
    Regress._kp_supported(::NotAnEstimator) = true

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    m = Regress.iv(Regress.TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin + Pop)))
    fsd = m.postestimation.first_stage_data
    @test_throws MethodError Regress.compute_first_stage_fstat(
        fsd.Xendo_res, fsd.Z_res, fsd.Pi, NotAnEstimator(), 1.0)
    @test_throws MethodError Regress._compute_first_stage_fstats_via_ols(
        fsd.newZ, fsd.Xendo_orig, 2, NotAnEstimator(), 0)
end
