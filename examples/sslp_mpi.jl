#=
Source:
  Ntaimo, L. and S. Sen, "The 'million-variable' march for stochastic combinatorial optimization," Journal of Optimization, 2005.

Input:
  nJ: number of servers
  nI: number of clients
  nS: number of scenarios

Sets:
  sI: clients
  sJ: servers
  sZ: zones

Variables (1st Stage):
  x[j]: 1 if a server is located at site j, 0 otherwise

Variables (2nd Stage):
  y[i,j]: 1 if client i is served by a server at location j, 0 otherwise
  y0[j]: any overflows that are not served due to limitations in server capacity

Parameters (general):
  c[j]: cost of locating a server at location j
  q[i,j]: revenue from client i being served by server at location j
  q0[j]: overflow penalty
  d[i,j]: client i resource demand from server at location j
  u: server capacity
  v: an upper bound on the total number of servers that can be located
  w[z]: minimum number of servers to be located in zone z
  Jz[z]: the subset of server locations that belong to zone z
  p[s]: probability of occurrence for scenario s

Parameters (scenario):
  h[i,s]: 1 if client i is present in scenario s, 0 otherwise
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
        "--nJ"
            help = "number of servers"
            arg_type = Int
            default = 10
        "--nI"
            help = "number of clients"
            arg_type = Int
            default = 50
        "--nS"
            help = "number of scenarios"
            arg_type = Int
            default = 50
        "--tol"
            help = "tolerance level"
            arg_type = Float64
            default = 1e-6
        "--age"
            help = "cut age"
            arg_type = Int
            default = 10
        "--proxu"
            help = "initial proximal penalty value"
            arg_type = Float64
            default = 1.e-2
        "--numcut"
            help = "number of cuts"
            arg_type = Int
            default = 1
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
nJ = parsed_args["nJ"]
nI = parsed_args["nI"]
nS = parsed_args["nS"]
tol = parsed_args["tol"]
age = parsed_args["age"]
proxu = parsed_args["proxu"]
numcut = parsed_args["numcut"]
dir = parsed_args["dir"]
seed::Int = 1
# function main_sslp(nJ::Int, nI::Int, nS::Int, seed::Int=1)

Random.seed!(seed)

sJ = 1:nJ
sI = 1:nI
sS = 1:nS

c = rand(40:80,nJ)
q = rand(0:25,nI,nJ,nS)
q0 = ones(nJ)*1000
d = q
u = 1.5*sum(d)/nJ
v = nJ
h = rand(0:1,nI,nS)
Pr = ones(nS)/nS

# This creates a Lagrange dual problem for each scenario s.
function create_scenario_model(s::Int64)
    if subsolver == "cplex"
      model = Model(CPLEX.Optimizer)
    else
      model = Model(GLPK.Optimizer)
    end

    @variable(model, x[j=sJ], Bin)
    @variable(model, y[i=sI,j=sJ], Bin)
    @variable(model, y0[j=sJ] >= 0)

    @objective(model, Min, (
            sum(c[j]*x[j] for j in sJ)
        - sum(q[i,j,s]*y[i,j] for i in sI for j in sJ)
        + sum(q0[j]*y0[j] for j in sJ)))

    @constraint(model, sum(x[j] for j in sJ) <= v)
    @constraint(model, [j=sJ], sum(d[i,j,s]*y[i,j] for i in sI) - y0[j] <= u*x[j])
    @constraint(model, [i=sI], sum(y[i,j] for j in sJ) == h[i,s])

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
    for i in sJ
        push!(coupling_variables, DD.CouplingVariableRef(s, i, xref[i]))
    end
end

# Set nonanticipativity variables as an array of symbols.
DD.set_coupling_variables!(algo, coupling_variables)

# Lagrange master method
params = BM.Parameters()
BM.set_parameter(params, "ϵ_s", tol)
BM.set_parameter(params, "max_age", age)
BM.set_parameter(params, "u", proxu)
BM.set_parameter(params, "ncuts_per_iter", numcut)
LM = DD.BundleMaster(BM.ProximalMethod, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0), params)

# Solve the problem with the solver; this solver is for the underlying bundle method.
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