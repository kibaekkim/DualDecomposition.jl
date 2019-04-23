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
if !isless(VERSION,v"0.7.0")
    using Random
	srand(s) = Random.seed!(s)
end
using JuDD
using JuMP, Ipopt
using CPLEX

function main_dcap(nR::Int, nN::Int, nT::Int, nS::Int, seed::Int=1; use_admm = false, admm_options...)

    srand(seed)

    global sR = 1:nR
    global sN = 1:nN
    global sT = 1:nT
    global sS = 1:nS

    ## parameters
    global a = rand(nR, nT) * 5 .+ 5
    global b = rand(nR, nT) * 40 .+ 10
    global c = rand(nR, nN, nT, nS) * 5 .+ 5
    global c0 = rand(nN, nT, nS) * 500 .+ 500
    global d = rand(nN, nT, nS) .+ 0.5
    Pr = ones(nS)/nS

    # Create JuDD instance.
	if use_admm
		algo = AdmmAlg(;rho = 50)
	else
	    algo = LagrangeDualAlg(nS)
	end

    # Add Lagrange dual problem for each scenario s.
    for s in 1:nS
        add_scenario_model(algo, s, Pr[s], create_scenario_model(s))
    end

    # Set nonanticipativity variables as an array of symbols.
    set_nonanticipativity_vars(algo, nonanticipativity_vars())

    # Solve the problem with the solver; this solver is for the underlying bundle method.
	if use_admm
    	JuDD.solve(algo, CplexSolver(CPX_PARAM_SCRIND=0))
	else
    	JuDD.solve(algo, IpoptSolver(print_level=0), master_alrogithm = :ProximalBundle)
	end
end

# This creates a Lagrange dual problem for each scenario s.
function create_scenario_model(s::Int64)

    # construct JuMP.Model
    model = Model(solver=CplexSolver(CPX_PARAM_SCRIND=0))

    ## 1st stage
    @variable(model, x[i=sR,t=sT] >= 0)
    @variable(model, u[i=sR,t=sT], Bin)
    @variable(model, y[i=sR,j=sN,t=sT], Bin)
    @variable(model, z[j=sN,t=sT], Bin)
    @objective(model, Min,
          sum(a[i,t]*x[i,t] + b[i,t]*u[i,t] for i in sR for t in sT)
        + sum(c[i,j,t,s]*y[i,j,t] for i in sR for j in sN for t in sT)
        + sum(c0[j,t,s]*z[j,t] for j in sN for t in sT))
    @constraint(model, [i=sR,t=sT], x[i,t] - u[i,t] <= 0)
    @constraint(model, [i=sR,t=sT], -sum(x[i,tau] for tau in 1:t) + sum(d[j,t,s]*y[i,j,t] for j in sN) <= 0)
    @constraint(model, [j=sN,t=sT], sum(y[i,j,t] for i in sR) + z[j,t] == 1)

    return model
end

# return the array of nonanticipativity variables
nonanticipativity_vars() = [:x,:u]

main_dcap(2,3,3,20; use_admm=true)
