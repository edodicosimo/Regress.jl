@testitem "Probit Fisher weights and working response" tags = [:probit] begin
    using Regress, DataFrames
    using Regress: fit_probit, ProbitWorkspaceForFS, BinaryResponse, _probit_irls!
    using StatsAPI: deviance

    df = DataFrame(x = repeat([-1.0, 0.0, 1.0], inner = 4),
        y = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    m = fit_probit(df, @formula(y ~ x), nothing, 100, 1e-10)
    @test deviance(m) ≈ deviance(m.rr)
    fill!(m.pp.beta, 0.0)
    m.rr = BinaryResponse(df.y, m.pp, :y)
    ws = ProbitWorkspaceForFS(m.pp.X)
    @test ws isa ProbitWorkspaceForFS

    _probit_irls!(m, ws, nothing, false, 1, 1e-10)

    # At eta = 0, mu = 1/2 and phi(eta) = 1/sqrt(2pi).
    derivative = inv(sqrt(2pi))
    @test ws.g ≈ df.y .- 0.5
    @test ws.d ≈ fill(derivative, length(df.y))
    @test ws.h ≈ fill(2 / pi, length(df.y))
    @test ws.z ≈ (df.y .- 0.5) ./ derivative
end
