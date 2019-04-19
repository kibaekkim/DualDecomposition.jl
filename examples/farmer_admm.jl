using JuMP
using CPLEX
using JuDD

const NS = 3  # number of scenarios
const probability = ones(3) / 3

const CROPS = 1:3 # set of crops (wheat, corn and sugar beets, resp.)
const PURCH = 1:2 # set of crops to purchase (wheat and corn, resp.)
const SELL = 1:4  # set of crops to sell (wheat, corn, sugar beets under 6K and those over 6K)

const Cost = [150 230 260]    # cost of planting crops
const Budget = 500            # budget capacity
const Purchase = [238 210]    # purchase price
const Sell = [170 150 36 10]  # selling price
const Yield = [3.0 3.6 24.0; 2.5 3.0 20.0; 2.0 2.4 16.0]
const Minreq = [200 240 0]    # minimum crop requirement

# This is the main function to solve the example by using dual decomposition.
function main_farmer(;admm_options...)
    # Create ADMM instance.
    admm = ADMM.AdmmAlg(;admm_options...)

    # Add Lagrange dual problem for each scenario s.
    for s in 1:NS
        admm_addscenario(admm, s, probability[s], create_scenario_model(s))
    end

    # Set nonanticipativity variables as an array of symbols.
    admm_setnonantvars(admm, nonanticipativity_vars())

    # Solve the problem with the solver; this solver is for the underlying bundle method.
    admm_solve(admm, CplexSolver(CPX_PARAM_SCRIND=0))
end

# This creates a Lagrange dual problem for each scenario s.
function create_scenario_model(s::Int64)
    m = Model(solver=CplexSolver(CPX_PARAM_SCRIND=0))
    @variable(m, 0 <= x[i=CROPS] <= 500, Int)
    @variable(m, y[j=PURCH] >= 0)
    @variable(m, w[k=SELL] >= 0)

    @objective(m, Min,
          sum(Cost[i] * x[i] for i=CROPS)
        + sum(Purchase[j] * y[j] for j=PURCH) - sum(Sell[k] * w[k] for k=SELL))

    @constraint(m, sum(x[i] for i=CROPS) <= Budget)
    @constraint(m, [j=PURCH], Yield[s,j] * x[j] + y[j] - w[j] >= Minreq[j])
    @constraint(m, Yield[s,3] * x[3] - w[3] - w[4] >= Minreq[3])
    @constraint(m, w[3] <= 6000)
    return m
end

# return the array of nonanticipativity variables
nonanticipativity_vars() = [:x]

main_farmer()
