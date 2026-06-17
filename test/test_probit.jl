using Regress
using Regress: fe
using Regress: probit
using CSV
using DataFrames


# Load a dataset
rwm_data = CSV.read(joinpath(@__DIR__, "data/rwm.data"), DataFrame, header=false, delim=' ', ignorerepeated=true)

rename!(rwm_data, [
    :id, :female, :year, :age, :hsat, :handdum, :handper, 
    :hhninc, :hhkids, :educ, :married, :haupts, :reals, 
    :fachhs, :abitur, :univ, :working, :bluec, :whitec, 
    :self, :beamt, :docvis, :hospvis, :public, :addon
])

# if id had done at least a visit in the year put 1 else 0
rwm_data[!, :visit_dummy] = ifelse.(rwm_data.docvis .> 0, 1, 0)
           

m = probit(
    rwm_data,
    @formula(visit_dummy ~ age + hhninc + hhkids + educ + married + fe(id) + fe(year));
    beta0 = nothing,
    max_iter=1000,
    tolerance=1e-6
)
println(m)

n = probit(
    rwm_data,
    @formula(visit_dummy ~ age + hhninc + hhkids + educ + married + fe(id));
    beta0 = nothing,
    max_iter=1000,
    tolerance=1e-6
)
println(n)

o = probit(
    rwm_data,
    @formula(visit_dummy ~ age + hhninc + hhkids + educ + married);
    beta0 = nothing,
    max_iter=1000,
    tolerance=1e-6
)
println(o)

p = probit(
    rwm_data,
    @formula(visit_dummy ~ educ);
    beta0 = nothing,
    max_iter=1000,
    tolerance=1e-6
)
println(p)