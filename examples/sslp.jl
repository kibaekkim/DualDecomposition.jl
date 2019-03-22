#=
Source:
  Ntaimo, L. and S. Sen, "The 'million-variable' march for stochastic combinatorial optimization," Journal of Global Optimization, 2005.

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

using JuDD
using JuMP, CPLEX, Ipopt

function main_sslp(nJ::Int, nI::Int, nS::Int, seed::Int=1)

    # Create JuDD instance.
    JuDD.LagrangeDuals(nS)

    srand(seed)

    global sJ = 1:nJ
    global sI = 1:nI
    global sS = 1:nS

    global c = rand(40:80,nJ)
    global q = rand(0:25,nI,nJ,nS)
    global q0 = ones(nJ)*1000
    global d = q
    global u = 1.5*sum(d)/nJ
    global v = nJ
    global h = rand(0:1,nI,nS)
    Pr = ones(nS)/nS

    # Add Lagrange dual problem for each scenario s.
    for s in 1:nS
        JuDD.add_Lagrange_dual_model(s, Pr[s], create_scenario_model(s))
    end

    # Set nonanticipativity variables as an array of symbols.
    JuDD.set_nonanticipativity_vars(nonanticipativity_vars())

    # Solve the problem with the solver; this solver is for the underlying bundle method.
    JuDD.solve(IpoptSolver(print_level=0), master_alrogithm = :ProximalDualBundle)
end

# This creates a Lagrange dual problem for each scenario s.
function create_scenario_model(s::Int64)
    model = Model(solver=CplexSolver(CPX_PARAM_SCRIND=0))

    @variable(model, x[j=sJ], Bin)
    @variable(model, y[i=sI,j=sJ], Bin)
    @variable(model, y0[j=sJ] >= 0)

    @objective(model, Min,
          sum(c[j]*x[j] for j in sJ)
        - sum(q[i,j,s]*y[i,j] for i in sI for j in sJ)
        + sum(q0[j]*y0[j] for j in sJ))

    @constraint(model, sum(x[j] for j in sJ) <= v)
    @constraint(model, [j=sJ], sum(d[i,j,s]*y[i,j] for i in sI) - y0[j] <= u*x[j])
    @constraint(model, [i=sI], sum(y[i,j] for j in sJ) == h[i,s])

    return model
end

# return the array of nonanticipativity variables
nonanticipativity_vars() = [:x]

main_sslp(10,50,50)
# main_sslp(10,50,100)
# main_sslp(10,50,500)
# main_sslp(10,50,1000)
# main_sslp(10,50,2000)
