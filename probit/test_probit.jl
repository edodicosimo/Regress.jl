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


X_i = rwm_data[:,Cols("age", "hhninc", "hhkids", "educ", "married")] #
X_i = Matrix(X_i) #
y = Matrix(rwm_data[:,Cols("visit_dummy")]) #
β = [0,0,0,0,0]   #
eta = X_i*β #

v = log_likelihood_probit.(y,eta) 
gi = getindex.(v, 1)
hi = getindex.(v, 2)

z_i = eta .+ (gi./hi)
id = Matrix(rwm_data[:,Cols("id")])
df = DataFrame(
    z = vec(z_i),
    y = vec(y),
    h = vec(hi),
    x1 = X_i[:,1],
    x2 = X_i[:,2],
    x3 = X_i[:,3],
    x4 = X_i[:,4],
    x5 = X_i[:,5],
    id = vec(id)
)


m = Regress.ols(
    df,
    @formula(z ~ x1 + x2 + x3 + x4 + x5 + fe(id));
    weights = :h,
    save = :fe,
)

beta_new = coef(m)    #                   # coefficienti delle x, vettore
alpha_new = Regress.fe(m; keepkeys = true) # fixed effects stimati, con chiave c
ŷ = predict(m, df)                 


leftjoin(rwm_data, unique(alpha_new, :id), on=:id)

hatBeta = fit_probit(
    rwm_data,
    [0,0,0,0,0],
    100,
    0.01
)

