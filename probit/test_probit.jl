include("fit_probit.jl")
using CSV
using DataFrames


# Load a dataset
rwm_data = CSV.read("/Users/edoardodicosimo/Downloads/rwm.data", DataFrame, header=false, delim=' ', ignorerepeated=true)

rename!(rwm_data, [
    :id, :female, :year, :age, :hsat, :handdum, :handper, 
    :hhninc, :hhkids, :educ, :married, :haupts, :reals, 
    :fachhs, :abitur, :univ, :working, :bluec, :whitec, 
    :self, :beamt, :docvis, :hospvis, :public, :addon
])

# if id had done at least a visit in the year put 1 else 0
rwm_data[!, :visit_dummy] = ifelse.(rwm_data.docvis .> 0, 1, 0)
           


hatBeta,i,df,d = fit_probit(
    rwm_data,
    @formula(visit_dummy ~ age + hhninc + hhkids + educ + married + fe(id)),
    [0,0,0,0,0],
    1000,
    0.000000001
)


