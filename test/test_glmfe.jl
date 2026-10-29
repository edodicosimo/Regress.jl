@testitem "glmfe probit and logit match GLM" tags = [:probit, :glmfe, :fe] begin
    using Regress, DataFrames, StableRNGs
    using Regress: fe
    import GLM
    using StatsAPI: coef, fitted, deviance, loglikelihood

    rng = StableRNG(29102026)
    n = 1800
    x = randn(rng, n)
    group = repeat(1:6, inner=300)
    group_effect = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
    eta = 0.2 .+ 0.6 .* x .+ group_effect[group]
    for (link, reference_link) in ((Regress.ProbitLink(), GLM.ProbitLink()),
                                  (Regress.LogitLink(), GLM.LogitLink()))
        probabilities = link isa Regress.ProbitLink ? Regress.normcdf.(eta) :
                        Regress.StatsFuns.logistic.(eta)
        y = Int.(rand(rng, n) .< probabilities)
        df = DataFrame(; y, x, group, groupcat=string.(group))
        for fixed in (false, true)
            formula = fixed ? @formula(y ~ x + fe(group)) : @formula(y ~ x)
            reference_formula = fixed ? @formula(y ~ x + groupcat) : @formula(y ~ x)
            m = glmfe(formula, df; link, max_iter=1000, tolerance=1e-10)
            reference = GLM.glm(reference_formula, df, GLM.Binomial(), reference_link)
            @test m.rr.link === link
            @test isconcretetype(fieldtype(typeof(m), :rr))
            @test coef(m)[fixed ? 1 : 2] ≈ coef(reference)[2] atol=1e-5
            @test all(isapprox.(fitted(m), fitted(reference); atol=1e-5))
            @test deviance(m) ≈ deviance(reference) atol=1e-6
            @test loglikelihood(m) ≈ loglikelihood(reference) atol=1e-6
            @test m.rr.one_minus_mu ≈ 1 .- fitted(m)
            @test m.rr.resid ≈ y .- fitted(m)
            if !fixed && link isa Regress.ProbitLink
                legacy = Regress.probit(df, formula; max_iter=1000, tolerance=1e-10)
                @test coef(legacy) == coef(m)
                @test fitted(legacy) == fitted(m)
                @test coef(glmfe(formula, df; max_iter=1000, tolerance=1e-10)) == coef(m)
            end
        end
    end
end

@testitem "glmfe inverse links update buffers in place" tags = [:glmfe] begin
    using Regress
    using Regress: BinaryResponse, BinaryPredictorQR, inverse_link!
    for link in (Regress.ProbitLink(), Regress.LogitLink())
        pp = BinaryPredictorQR{Float64}(ones(3, 1), [0.0], [0.0])
        rr = BinaryResponse([0, 1, 1], pp, :y, link)
        buffers = (rr.mu, rr.one_minus_mu, rr.dmu_deta)
        rr.eta .= [-40.0, 0.0, 40.0]
        @test inverse_link!(rr) === rr
        @test rr.mu === buffers[1]
        @test rr.one_minus_mu === buffers[2]
        @test rr.dmu_deta === buffers[3]
        @test rr.mu[2] == rr.one_minus_mu[2] == 0.5
        @test rr.dmu_deta[2] ≈ (link isa Regress.ProbitLink ? inv(sqrt(2pi)) : 0.25)
        @test all(isfinite, rr.mu)
        @test all(isfinite, rr.one_minus_mu)
        @test all(isfinite, rr.dmu_deta)
    end
end
