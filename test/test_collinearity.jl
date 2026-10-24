@testitem "collinearity with fixed effects" tags = [:ols, :fe, :vcov] begin
    using DataFrames, CSV, Regress, Random, Statistics
    using Regress: fe
    using StatsBase: coef, stderror
    using CovarianceMatrices: CR1

    # read the data
    csvfile = CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv"))
    df = DataFrame(csvfile)

    # create a bigger dataset
    sort!(df, [:State, :Year])
    nstates = maximum(df.State)
    dflarge = copy(df)
    for i in 1:100
        dfnew = copy(df)
        dfnew.State .+= i .* nstates
        append!(dflarge, dfnew)
    end

    # create a dummy variable that is collinear with the State-FE
    dflarge.highstate = dflarge.State .< median(dflarge.State)

    # create a second 'high-dimensional' categorical variable
    Random.seed!(1234)
    dflarge.catvar = rand(1:200, nrow(dflarge))

    rr = Regress.ols(
        dflarge, @formula(Price ~ highstate + Pop + fe(Year) + fe(catvar) + fe(State)),
        save_cluster = :State, fe_tol = 1e-6)

    # test that the collinear coefficient is zero and the standard error is NaN
    @test coef(rr)[1] ≈ 0.0
    @test isnan(stderror(CR1(:State), rr)[1])
end

@testitem "collinearity - small data" tags = [:ols] begin
    using DataFrames, Regress
    using StatsBase: stderror

    df = DataFrame(
        [36.9302 44.5105;
         39.4935 44.5044;
         38.946 44.5072;
         37.8005 44.5098;
         37.2613 44.5103;
         35.3885 44.5109;],
        :auto
    )
    rr = Regress.ols(df, @formula(x1 ~ x2))
    @test all(!isnan, stderror(rr))
end

@testitem "sweep collinearity detection drops later columns" tags = [:ols] begin
    using Regress, LinearAlgebra, StableRNGs

    rng = StableRNG(3)
    n = 200
    x1, x2, x3 = randn(rng, n), 1e3 .* randn(rng, n), randn(rng, n)
    X = [ones(n) x1 x2 2 .* x2 x3 x1 .+ x2]

    # The column that the earlier ones explain is dropped, as in sweep_solve!.
    for has_intercept in (true, false)
        basis, _ = Regress.detect_collinearity(X; method = :sweep, has_intercept)
        @test basis == [true, true, true, false, true, false]
        aug = Matrix(X' * X)
        b, _, _ = Regress.sweep_solve!(aug, X' * x3, has_intercept; tol = 1e-8)
        @test b == basis
    end

    # On a full-rank matrix the sweep leaves -inv(A) in the upper triangle.
    A = Matrix(X[:, [1, 2, 3, 5]]' * X[:, [1, 2, 3, 5]])
    S = copy(A)
    basis, rank = Regress.sweep_collinear!(S, 1e-8)
    @test all(basis) && rank == 4
    @test Symmetric(S, :U) ≈ -inv(A) rtol = 1e-10

    # Both detection methods find the same rank; :sweep keeps formula order.
    y = x1 .+ randn(rng, n)
    m = Regress.ols(X, y)
    @test isnan(stderror(m)[4]) && isnan(stderror(m)[6])
    @test sum(Regress.ols(X, y; collinearity = :qr).basis_coef) == 4
end
