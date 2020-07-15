using DualDecomposition
using JuMP, Ipopt, GLPK

const DD = DualDecomposition

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

# This creates a Lagrange dual problem for each scenario s.
function create_scenario_model(s::Int64)
    m = Model(GLPK.Optimizer)
    @variable(m, 0 <= x[i=CROPS] <= 500, Int)
    @variable(m, y[j=PURCH] >= 0)
    @variable(m, w[k=SELL] >= 0)

    @objective(m, Min,
        probability[s] * sum(Cost[i] * x[i] for i=CROPS)
        + probability[s] * sum(Purchase[j] * y[j] for j=PURCH) 
        - probability[s] * sum(Sell[k] * w[k] for k=SELL))

    @constraint(m, sum(x[i] for i=CROPS) <= Budget)
    @constraint(m, [j=PURCH], Yield[s,j] * x[j] + y[j] - w[j] >= Minreq[j])
    @constraint(m, Yield[s,3] * x[3] - w[3] - w[4] >= Minreq[3])
    @constraint(m, w[3] <= 6000)
    return m
end

# Create DualDecomposition instance.
algo = DD.LagrangeDual()

# Add Lagrange dual problem for each scenario s.
models = Dict{Int,JuMP.Model}(s => create_scenario_model(s) for s in 1:NS)
for s in 1:NS
    DD.add_block_model!(algo, s, models[s])
end

coupling_variables = Vector{DD.CouplingVariableRef}()
for s in 1:NS
    model = models[s]
    xref = model[:x]
    for i in CROPS
        push!(coupling_variables, DD.CouplingVariableRef(s, i, xref[i]))
    end
end

# Set nonanticipativity variables as an array of symbols.
DD.set_coupling_variables!(algo, coupling_variables)

# Solve the problem with the solver; this solver is for the underlying bundle method.
DD.run!(algo, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
