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
        "--tol"
            help = "ADMM tolerance level"
            arg_type = Float64
            default = 1e-6
        "--dir"
            help = "output directory"
            arg_type = String
            default = "."
    end
    return parse_args(s)
end

parsed_args = parse_commandline()

nR = parsed_args["nR"]
nN = parsed_args["nN"]
nT = parsed_args["nT"]
nS = parsed_args["nS"]
tol = parsed_args["tol"]
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
    model = Model(GLPK.Optimizer)

    ## 1st stage
    @variable(model, x[i=sR,t=sT] >= 0)
    @variable(model, u[i=sR,t=sT], Bin)
    @variable(model, y[i=sR,j=sN,t=sT], Bin)
    @variable(model, z[j=sN,t=sT], Bin)
    @objective(model, Min, Pr[s]*(
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
algo = DD.LagrangeDual()

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
params = BM.Parameters()
BM.set_parameter(params, "ϵ_s", tol)
LM = DD.BundleMaster(BM.ProximalMethod, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0), params)

DD.run!(algo, LM)

mkpath(dir)
DD.write_all(algo, dir=dir)

if (parallel.is_root())
  @show DD.primal_objective_value(algo)
  @show DD.dual_objective_value(algo)
  @show DD.primal_solution(algo)
  @show DD.dual_solution(algo)
end

# Finalize MPI
parallel.finalize()

