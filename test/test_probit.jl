@testitem "Probit coefficients - RWM saved results" tags = [:probit, :fe] begin
    using Regress, CSV, DataFrames
    using Regress: fe, probit
    using StatsAPI: coef

    rwm_data = CSV.read(
        joinpath(@__DIR__, "data/rwm.data"),
        DataFrame;
        header = false,
        delim = ' ',
        ignorerepeated = true
    )

    rename!(rwm_data, [
        :id, :female, :year, :age, :hsat, :handdum, :handper,
        :hhninc, :hhkids, :educ, :married, :haupts, :reals,
        :fachhs, :abitur, :univ, :working, :bluec, :whitec,
        :self, :beamt, :docvis, :hospvis, :public, :addon
    ])

    rwm_data[!, :visit_dummy] = ifelse.(rwm_data.docvis .> 0, 1, 0)

    model = probit(
        rwm_data,
        @formula(visit_dummy ~ age + hhninc + hhkids + educ + married + fe(id) + fe(year));
        beta0 = nothing,
        max_iter = 1000,
        tolerance = 1e-6
    )

    expected_coef = [
        0.0,
        -2.028526040781942e-6,
        -0.030016603889186432,
        -0.06974646303214804,
        -0.02956180984977514
    ]

    @test coef(model) ≈ expected_coef atol = 1e-4
end

@testitem "Probit coefficients - RWM model variants" tags = [:probit, :fe] begin
    using Regress, CSV, DataFrames
    using Regress: fe, probit
    using StatsAPI: coef

    rwm_data = CSV.read(
        joinpath(@__DIR__, "data/rwm.data"),
        DataFrame;
        header = false,
        delim = ' ',
        ignorerepeated = true
    )

    rename!(rwm_data, [
        :id, :female, :year, :age, :hsat, :handdum, :handper,
        :hhninc, :hhkids, :educ, :married, :haupts, :reals,
        :fachhs, :abitur, :univ, :working, :bluec, :whitec,
        :self, :beamt, :docvis, :hospvis, :public, :addon
    ])

    rwm_data[!, :visit_dummy] = ifelse.(rwm_data.docvis .> 0, 1, 0)

    model_id_fe = probit(
        rwm_data,
        @formula(visit_dummy ~ age + hhninc + hhkids + educ + married + fe(id));
        beta0 = nothing,
        max_iter = 1000,
        tolerance = 1e-6
    )
    expected_id_fe_coef = [
        0.0625323,
        -3.43276e-6,
        -0.0482711,
        -0.0721936,
        -0.0327782
    ]
    @test coef(model_id_fe) ≈ expected_id_fe_coef atol = 1e-4

    model_no_fe = probit(
        rwm_data,
        @formula(visit_dummy ~ age + hhninc + hhkids + educ + married);
        beta0 = nothing,
        max_iter = 1000,
        tolerance = 1e-6
    )
    expected_no_fe_coef = [
        0.0142774,
        -1.1008e-5,
        -0.126499,
        -0.0211008,
        0.0548625
    ]
    @test coef(model_no_fe) ≈ expected_no_fe_coef atol = 1e-4

    model_educ = probit(
        rwm_data,
        @formula(visit_dummy ~ educ);
        beta0 = nothing,
        max_iter = 1000,
        tolerance = 1e-6
    )
    @test coef(model_educ) ≈ [0.0261837] atol = 1e-4
end
