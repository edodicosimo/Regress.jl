@testitem "Invalid input throws ArgumentError" tags = [:ols, :iv] begin
    using Regress, DataFrames, CSV, StatsBase, CovarianceMatrices

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    # Data dropped by save = :minimal
    mols = Regress.ols(df, @formula(Sales ~ NDI); save = :minimal)
    @test_throws ArgumentError residuals(mols)
    @test_throws "Response vector not stored. Model was fit with save=:minimal" residuals(mols)
    miv = Regress.iv(Regress.TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)); save = :minimal)
    @test_throws ArgumentError vcov(HC1(), miv)
    @test_throws "Model does not have residuals stored" vcov(HC1(), miv)

    # Cluster variable not saved at fit time
    ols_f = Regress.ols(df, @formula(Sales ~ NDI + Price))
    iv_f = Regress.iv(Regress.TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))
    for m in (ols_f, iv_f)
        @test_throws ArgumentError vcov(CR1(:State), m)
        @test_throws "Cluster variable :State not found in model" vcov(CR1(:State), m)
    end
end

@testitem "matrix IV drops collinear columns as the formula fit does" tags = [:iv, :matrix] begin
    using Regress, DataFrames, CSV, StatsBase

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    n = nrow(df)

    # A repeated instrument is dropped.
    X = hcat(ones(n), df.NDI)
    m_dup = Regress.iv(Regress.TSLS(), hcat(ones(n), df.Pimin, df.Pimin), X, df.Sales)
    m_one = Regress.iv(Regress.TSLS(), hcat(ones(n), df.Pimin), X, df.Sales)
    @test coef(m_dup) == coef(m_one)
    @test stderror(m_dup) == stderror(m_one)

    # An endogenous regressor that repeats an exogenous one (exactly or to 1e-12) becomes
    # exogenous and is dropped as collinear; the fit is then OLS on the other columns.
    Z = hcat(ones(n), df.Pimin, df.Pop)
    m_ols = Regress.ols(X, df.Sales)
    for Xdup in (hcat(X, df.NDI), hcat(X, df.NDI .* (1 .+ 1e-12 .* sin.(1:n))))
        m = Regress.iv(Regress.TSLS(), Z, Xdup, df.Sales)
        @test coef(m)[1:2] ≈ coef(m_ols) rtol = 1e-10
        @test stderror(m)[1:2] ≈ stderror(m_ols) rtol = 1e-10
        @test coef(m)[3] == 0 && isnan(stderror(m)[3])
        @test dof_residual(m) == dof_residual(m_ols)
    end
end
