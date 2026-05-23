using Regress
using Regress: fe
using Regress: probit
using CSV
using DataFrames
using BenchmarkTools

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
           

@time m = probit(
    rwm_data,
    @formula(visit_dummy ~ age + hhninc + hhkids + educ + married + fe(id) + fe(year));
    beta0=[0,0,0,0,0],
    max_iter=1000,
    tolerance=1e-6
)


