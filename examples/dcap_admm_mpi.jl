#=
Source:
  S. Ahmed and R. Garcia. "Dynamic Capacity Acquisition and Assignment under Uncertainty," Annals of Operations Research, vol.124, pp. 267-283, 2003

Input:
  nR: number of resources
  nN: number of tasks
  nT: number of time periods
  nS: number of scenarios

Sets:
  sR: resources
  sN: tasks
  sT: time periods
  sS: scenarios

Variables (1st Stage):
  x[i,t]: capacity acquired for resource i at period t
  u[i,t]: 1 if x[i,t] > 0, 0 otherwise

Variables (2nd Stage):
  y[i,j,t]: 1 if resource i is assigned to task j in period t, 0 otherwise

Parameters (general):
  a[i,t]: linear component of expansion cost for resource i at period t
  b[i,t]: fixed component of expansion cost for resource i at period t
  c[i,j,t,s]: cost of assigning resource i to task j in period t
  c0[j,t,s]: penalty incurred if task j in period t is not served

Parameters (scenario):
  d[j,t,s]: capacity required for to perform task j in period t in scenario s
=#
using DualDecomposition
using JuMP, Ipopt, GLPK
using Random
using ArgParse

const DD = DualDecomposition
const parallel = DD.parallel

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--subsolver"
          help = "solver for subproblem:\n
                  -glpk
                  -cplex"
          arg_type = String
          default = "glpk"
        "--timelim"
            help = "time limit"
            arg_type = Float64
            default = 3600.0
        "--alg"
          help = "algorithm mode:\n
                  -0: constant ρ\n
                  -1: residual balancing\n
                  -2: adaptive residual balancing\n
                  -3: relaxed ADMM\n
                  -4: adaptive relaxed ADMM"
          arg_type = Int
          default = 1
        "--nR"
            help = "number of resources"
            arg_type = Int
            default = 2
        "--nN"
            help = "number of tasks"
            arg_type = Int
            default = 3
        "--nT"
            help = "number of time periods"
            arg_type = Int
            default = 3
        "--nS"
            help = "number of scenarios"
            arg_type = Int
            default = 20
        "--rho"
            help = "initial penalty value"
            arg_type = Float64
            default = 1.0
        "--tol"
            help = "ADMM tolerance level"
            arg_type = Float64
            default = 1e-6
        "--tau"
            help = "Residual balancing multiplier"
            arg_type = Float64
            default = 2.0
        "--mu"
            help = "Residual balancing parameter"
            arg_type = Float64
            default = 1.0
        "--xi"
            help = "Residual balancing parameter"
            arg_type = Float64
            default = 10.0
        "--interval"
            help = "ADMM update interval"
            arg_type = Int
            default = 1
        "--age"
            help = "cut age"
            arg_type = Int
            default = 10
        "--dir"
            help = "output directory"
            arg_type = String
            default = "."
    end
    return parse_args(s)
end

parsed_args = parse_commandline()

subsolver = parsed_args["subsolver"]
if subsolver == "cplex"
  using CPLEX
end
timelim = parsed_args["timelim"]
alg = parsed_args["alg"]
nR = parsed_args["nR"]
nN = parsed_args["nN"]
nT = parsed_args["nT"]
nS = parsed_args["nS"]
rho = parsed_args["rho"]
tol = parsed_args["tol"]
tau = parsed_args["tau"]
mu = parsed_args["mu"]
xi = parsed_args["xi"]
uinterval = parsed_args["interval"]
age = parsed_args["age"]
dir = parsed_args["dir"]
seed::Int = 1

Random.seed!(seed)

sR = 1:nR
sN = 1:nN
sT = 1:nT
sS = 1:nS

## parameters
a = rand(nR, nT) * 5 .+ 5
b = rand(nR, nT) * 40 .+ 10
c = rand(nR, nN, nT, nS) * 5 .+ 5
c0 = rand(nN, nT, nS) * 500 .+ 500
d = rand(nN, nT, nS) .+ 0.5
Pr = ones(nS)/nS

# This creates a Lagrange dual problem for each scenario s.
function create_scenario_model(s::Int64)

    # construct JuMP.Model
    if subsolver == "cplex"
      model = Model(CPLEX.Optimizer)
      set_optimizer_attribute(model, "CPXPARAM_ScreenOutput", 0)
      set_optimizer_attribute(model, "CPXPARAM_MIP_Display", 0)
      set_optimizer_attribute(model, "CPX_PARAM_THREADS", 1)
    else
      model = Model(GLPK.Optimizer)
    end

    ## 1st stage
    @variable(model, x[i=sR,t=sT] >= 0)
    @variable(model, u[i=sR,t=sT], Bin)
    @variable(model, y[i=sR,j=sN,t=sT], Bin)
    @variable(model, z[j=sN,t=sT], Bin)
    @objective(model, Min, (
          sum(a[i,t]*x[i,t] + b[i,t]*u[i,t] for i in sR for t in sT)
        + sum(c[i,j,t,s]*y[i,j,t] for i in sR for j in sN for t in sT)
        + sum(c0[j,t,s]*z[j,t] for j in sN for t in sT)))
    @constraint(model, [i=sR,t=sT], x[i,t] - u[i,t] <= 0)
    @constraint(model, [i=sR,t=sT], -sum(x[i,tau] for tau in 1:t) + sum(d[j,t,s]*y[i,j,t] for j in sN) <= 0)
    @constraint(model, [j=sN,t=sT], sum(y[i,j,t] for i in sR) + z[j,t] == 1)

    return model
end

# Initialize MPI
parallel.init()

# Create DualDecomposition instance.
params = BM.Parameters()
BM.set_parameter(params, "print_output", false)
BM.set_parameter(params, "max_age", age)
algo = DD.AdmmLagrangeDual(BM.BasicMethod, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0), params)

# partition scenarios into processes
parallel.partition(nS)

# Add Lagrange dual problem for each scenario s.
models = Dict{Int,JuMP.Model}(s => create_scenario_model(s) for s in parallel.getpartition())
for s in parallel.getpartition()
    DD.add_block_model!(algo, s, models[s])
end

coupling_variables = Vector{DD.CouplingVariableRef}()
for s in parallel.getpartition()
    model = models[s]
    xref = model[:x]
    for i in sR, t in sT
        push!(coupling_variables, DD.CouplingVariableRef(s, (1,i,t), xref[i,t]))
    end
    uref = model[:u]
    for i in sR, t in sT
        push!(coupling_variables, DD.CouplingVariableRef(s, (2,i,t), uref[i,t]))
    end
end

# Set nonanticipativity variables as an array of symbols.
DD.set_coupling_variables!(algo, coupling_variables)

# Solve the problem with the solver; this solver is for the underlying bundle method.
LM = DD.AdmmMaster(alg=alg, ρ=rho, ϵ=tol, maxiter=100000, maxtime=timelim, update_interval = uinterval, τ=tau, μ=mu, ξ=xi)

DD.run!(algo, LM)

mkpath(dir)
DD.write_all(algo, dir=dir)
DD.write_all(LM, dir=dir)

if (parallel.is_root())
  @show DD.primal_objective_value(algo)
  @show DD.dual_objective_value(algo)
  @show DD.primal_solution(algo)
  @show DD.dual_solution(algo)
end

# Finalize MPI
parallel.finalize()

