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

    # Singular matrices in the matrix IV fit
    n = nrow(df)
    X = hcat(ones(n), df.NDI)
    Zdup = hcat(ones(n), df.Pimin, df.Pimin)
    @test_throws ArgumentError Regress.iv(Regress.TSLS(), Zdup, X, df.Sales)
    @test_throws "Instrument matrix Z is rank deficient" Regress.iv(Regress.TSLS(), Zdup, X, df.Sales)
    Xdup = hcat(ones(n), df.NDI, df.NDI)
    Z = hcat(ones(n), df.Pimin, df.Pop)
    @test_throws ArgumentError Regress.iv(Regress.TSLS(), Z, Xdup, df.Sales)
    @test_throws "X̂'X is singular" Regress.iv(Regress.TSLS(), Z, Xdup, df.Sales)
end
