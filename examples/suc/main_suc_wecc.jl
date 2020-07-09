#=
Source:
  A. Papavasiliou and S. Oren. (2013) Multiarea Stochastic Unit Commitment for High Wind
  Penetration in a Transmission Constrained Network. Operations Research 61(3):578-592

Note:
  Contact Kibaek Kim (kimk@anl.gov) regarding the data files for the model.

Date:
  03-06-2019
=#

include("suc_wecc_data.jl")

using DualDecomposition
using JuMP, CPLEX, Ipopt

# Number of scenarios: up to 1000 for each season
const NS = 2
# Offset to the scenario index
const offset = 0
# Seasons (8 different demand profiles)
const seasons = ["SpringWD","SpringWE","SummerWD","SummerWE","FallWD","FallWE","WinterWD","WinterWE"]

# Get model data
uc = weccdata(NS, offset, seasons[1])

# This is the main function to solve the example by using dual decomposition.
function main_suc_wecc()
    # Create DualDecomposition instance.
    DualDecomposition.LagrangeDuals(NS)

    # Add Lagrange dual problem for each scenario s.
    for s in 1:NS
      DualDecomposition.add_Lagrange_dual_model(s, uc.π[s], create_scenario_model(s))
    end

    # Set nonanticipativity variables as an array of symbols.
    DualDecomposition.set_nonanticipativity_vars(nonanticipativity_vars())

    # Solve the problem with the solver; this solver is for the underlying bundle method.
    DualDecomposition.solve(IpoptSolver(print_level=0), master_alrogithm = :ProximalDualBundle)
end

# This creates a Lagrange dual problem for each scenario s.
function create_scenario_model(s::Int64)
    model = Model(solver=CplexSolver(CPX_PARAM_SCRIND=0))

    @variable(model, w[g=uc.Gs,t=uc.T0], Bin)
    @variable(model, 0 <= z[g=uc.Gs,t=uc.T] <= 1)
    @variable(model, u[g=uc.Gf,t=uc.T0], Bin)
    @variable(model, 0 <= v[g=uc.Gf,t=uc.T] <= 1)
    @variable(model, -120 <= θ[n=uc.N,t=uc.T] <= 120)
    @variable(model, -uc.TC[l] <= e[l=uc.L,t=uc.T] <= uc.TC[l])
    @variable(model, p[g=uc.G,t=uc.T0] >= 0)
    @variable(model, 0 <= loadshed[i=uc.LOAD,t=uc.T] <= uc.load[i,t]) # load shedding
    @variable(model, 0 <= ispill[i=uc.IMPORT,t=uc.T] <= uc.Igen[i,t]) # import spillage
    @variable(model, 0 <= rspill[i=uc.RE,t=uc.T] <= uc.Rgen[i,t])     # renewable spillage
    @variable(model, 0 <= wspill[i=uc.WIND,t=uc.T] <= uc.Wgen[i,t,s])   # wind spillage

    @objective(model, Min,
          sum(uc.K[g] * w[g,t] + uc.S[g] * z[g,t] for g in uc.Gs for t in uc.T)
        + sum((uc.K[g] * u[g,t] + uc.S[g] * v[g,t]) for g in uc.Gf for t in uc.T)
        + sum(uc.C[g] * p[g,t] for g in uc.G for t in uc.T)
        + sum(uc.Cl * loadshed[i,t] for i in uc.LOAD for t in uc.T)
        + sum(uc.Ci * ispill[i,t] for i in uc.IMPORT for t in uc.T)
        + sum(uc.Cr * rspill[i,t] for i in uc.RE for t in uc.T)
        + sum(uc.Cw * wspill[i,t] for i in uc.WIND for t in uc.T)
    )

    # Unit commitment for slow generators
    @constraint(model, [g=uc.Gs,t=Int(uc.UT[g]):length(uc.T)], sum(z[g,q] for q=Int(t-uc.UT[g]+1):t) <= w[g,t])
    @constraint(model, [g=uc.Gs,t=1:Int(length(uc.T)-uc.DT[g])], sum(z[g,q] for q=(t+1):Int(t+uc.DT[g])) <= w[g,t])
    @constraint(model, [g=uc.Gs,t=uc.T], z[g,t] >= w[g,t] - w[g,t-1])

    # Unit commitment for fast generators
    @constraint(model, [g=uc.Gf,t=Int(uc.UT[g]):length(uc.T)], sum(v[g,q] for q=Int(t-uc.UT[g]+1):t) <= u[g,t])
    @constraint(model, [g=uc.Gf,t=1:Int(length(uc.T)-uc.DT[g])], sum(v[g,q] for q=(t+1):Int(t+uc.DT[g])) <= u[g,t])
    @constraint(model, [g=uc.Gf,t=uc.T], v[g,t] >= u[g,t] - u[g,t-1])

    # Flow balance
    @constraint(model, [n=uc.N,t=uc.T],
        sum(e[l,t] for l in uc.L if uc.tbus[l] == n)
        + sum(p[g,t] for g in uc.G if uc.gen2bus[g] == n)
        + sum(loadshed[i,t] for i in uc.LOAD if uc.load2bus[i] == n)
        + sum(uc.Wgen[i,t,s] for i in uc.WIND if uc.wind2bus[i] == n)
        == uc.D[n,t]
        + sum(e[l,t] for l in uc.L if uc.fbus[l] == n)
        + sum(ispill[i,t] for i in uc.IMPORT if uc.import2bus[i] == n)
        + sum(rspill[i,t] for i in uc.RE if uc.re2bus[i] == n)
        + sum(wspill[i,t] for i in uc.WIND if uc.wind2bus[i] == n)
    )

    # Power flow equation
    @constraint(model, [l=uc.L,t=uc.T], e[l,t] == uc.B[l] * (θ[uc.fbus[l],t] - θ[uc.tbus[l],t]))

    # Max generation capacity
    @constraint(model, [g=uc.Gs,t=uc.T0], p[g,t] <= uc.Pmax[g] * w[g,t])
    @constraint(model, [g=uc.Gf,t=uc.T0], p[g,t] <= uc.Pmax[g] * u[g,t])

    # # Min generation capacity
    @constraint(model, [g=uc.Gs,t=uc.T0], p[g,t] >= uc.Pmin[g] * w[g,t])
    @constraint(model, [g=uc.Gf,t=uc.T0], p[g,t] >= uc.Pmin[g] * u[g,t])

    # Ramping capacity
    @constraint(model, [g=uc.G,t=uc.T], p[g,t] - p[g,t-1] <= uc.Rmax[g])
    @constraint(model, [g=uc.G,t=uc.T], p[g,t] - p[g,t-1] >= -uc.Rmin[g])

    return model
end

# return the array of nonanticipativity variables
nonanticipativity_vars() = [:w,:z]

main_suc_wecc()
