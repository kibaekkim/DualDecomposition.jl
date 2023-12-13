using JuMP, GLPK, Ipopt
using DualDecomposition
using Random
using ArgParse

const DD = DualDecomposition
const parallel = DD.parallel

"""
a: interest rate
π: unit stock price
ρ: unit dividend price


K: number of stages
L: number of stock types
2^L scenarios in each stage
2^L^(K-1) scenarios in total
ρ = 0.05 * π
bank: interest rate 0.01
stock1: 1.03 or 0.97
stock2: 1.06 or 0.94
...

b_k: initial asset (if k=1) and income (else)
B_k: money in bank
x_{k,l}: number of stocks to buy/sell (integer)
y_{k,l}: total stocks 

deterministic model:

    max     B_K+∑_{l=1}^{L}π_{K,l}y_{K,l}

    s.t.    B_1+∑_{l=1}^{L}π_{1,l}x_{1,l} = b_1

            b_k+(1+a)B_{k-1}+∑_{l=1}^{L}ρ_{k,l}y_{k-1,l} = B_k+∑_{l=1}^{L}π_{k,l}x_{k,l}, ∀ k=2,…,K
    
            y_{1,l} = x_{1,l}, ∀ l=1,…,L
    
            y_{k-1,l}+x_{k,l} = y_{k,l}, ∀ k=2,…,K, l=1,…,L
    
            x_{k,l} ∈ ℤ , ∀ k=1,…,K, l=1,…,L
    
            y_{k,l} ≥ 0, ∀ k=1,…,K, l=1,…,L
    
            B_k ≥ 0, ∀ k=1,…,K.
"""

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--subsolver"
            help = "solver for subproblem:\n
                    -glpk
                    -cplex"
            arg_type = String
            default = "glpk"
        "--nK"
            help = "number of stages"
            arg_type = Int
            default = 3
        "--nL"
            help = "number of stock types"
            arg_type = Int
            default = 2
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
nK = parsed_args["nK"]
nL = parsed_args["nL"]
tol = parsed_args["tol"]
age = parsed_args["age"]
proxu = parsed_args["proxu"]
numcut = parsed_args["numcut"]
dir = parsed_args["dir"]
seed::Int = 1

Random.seed!(seed)

const K = nK
const L = nL
const a = 0.01
const b_init = 100  # initial capital
const b_in = 30   # income

# iteratively add nodes
# root node
function create_nodes()::DD.Tree
    ξ = Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}(:π => ones(L))
    tree = DD.Tree(ξ)

    #subproblem formulation
    function subproblem_builder(tree::DD.Tree, subtree::DD.SubTree, node::DD.SubTreeNode)
        mdl = subtree.model
        x = @variable(mdl, x[l=1:L], Int, base_name="n1_x")
        #x = @variable(mdl, x[l=1:L], base_name="n1_x")

        y = @variable(mdl, y[l=1:L] >= 0, base_name="n1_y")
        DD.set_output_variable!(node, :y, y)

        B = @variable(mdl, B >= 0, base_name="n1_B")
        DD.set_output_variable!(node, :B, B)

        π = DD.get_scenario(node)[:π]
        @constraints(mdl, 
            begin
                B + sum( π[l] * x[l] for l in 1:L) == b_init
                [l=1:L], y[l] - x[l] == 0 
            end
        )
        DD.set_stage_objective(node, 0.0)

        JuMP.unregister(mdl, :x)
        JuMP.unregister(mdl, :y)
        JuMP.unregister(mdl, :B)
    end

    DD.set_stage_builder!(tree, 1, subproblem_builder)

    create_nodes!(tree, 1)
    return tree
end

# child nodes
function create_nodes!(tree::DD.Tree, pt::Int)
    for scenario = 1:2^L
        prob = 1/2^L
        π = get_realization(DD.get_scenario(tree, pt)[:π], scenario)
        ξ = Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}(:π => π)
        id = DD.add_child!(tree, pt, ξ, prob)

        #subproblem formulation
        function subproblem_builder(tree::DD.Tree, subtree::DD.SubTree, node::DD.SubTreeNode)
            mdl = subtree.model
            id = DD.get_id(node)
            #x = @variable(mdl, x[l=1:L], Int, base_name = "n$(id)_x")
            x = @variable(mdl, x[l=1:L], base_name = "n$(id)_x")

            y = @variable(mdl, y[l=1:L] >= 0, base_name = "n$(id)_y")
            DD.set_output_variable!(node, :y, y)

            B = @variable(mdl, B >= 0, base_name = "n$(id)_B")
            DD.set_output_variable!(node, :B, B)

            y_ = @variable(mdl, y_[l=1:L] >= 0, base_name = "n$(id)_y_")
            DD.set_input_variable!(node, :y, y_)

            B_ = @variable(mdl, B_ >= 0, base_name = "n$(id)_B_")
            DD.set_input_variable!(node, :B, B_)

            π = DD.get_scenario(node)[:π]
            pt = DD.get_parent(node)
            ρ = DD.get_scenario(tree, pt)[:π] * 0.05
            @constraint(mdl, B + sum( π[l] * x[l] - ρ[l] * y_[l] for l in 1:L) - (1+a) * B_ == b_in)
            @constraint(mdl, [l=1:L], y[l] - x[l] - y_[l] == 0)


            #dummy bound for input variables to avoid subproblem unboundedness
            @constraint(mdl, [l=1:L], y_[l] <= 500)
            @constraint(mdl, B_ <= 500)
            if DD.get_stage(node) < K
                DD.set_stage_objective(node, 0.0)
            else
                DD.set_stage_objective(node, -(B + sum( π[l] * y[l] for l in 1:L )))
            end
            JuMP.unregister(mdl, :x)
            JuMP.unregister(mdl, :y)
            JuMP.unregister(mdl, :B)
            JuMP.unregister(mdl, :y_)
            JuMP.unregister(mdl, :B_)
        end

        DD.set_stage_builder!(tree, id, subproblem_builder)
        if DD.get_stage(tree, id) < K
            create_nodes!(tree, id)
        end
    end
end

# construct realization event
function get_realization(ξ::Array{Float64,1}, scenario::Int)::Array{Float64,1}
    ret = ones(L)
    multipliers = digits(scenario - 1, base=2, pad=L)*2 - ones(L)
    for l = 1:L
        ret[l] = ξ[l] * (1 + multipliers[l] * l * 0.03)
    end
    return ret
end

tree = create_nodes()
#node_cluster = DD.decomposition_not(tree)
node_cluster = DD.decomposition_scenario(tree)
#node_cluster = DD.decomposition_temporal(tree) #There is a DUAL_INFEASIBLE issue

# Number of block components
NS = length(node_cluster)

# Initialize MPI
parallel.init()

# Create DualDecomposition instance.
algo = DD.LagrangeDual()

# partition scenarios into processes
parallel.partition(NS)

coupling_variables = Vector{DD.CouplingVariableRef}()
models = Dict{Int,JuMP.Model}()

for block_id in parallel.getpartition()
    nodes = node_cluster[block_id]
    subtree = DD.create_subtree!(tree, block_id, coupling_variables, nodes)
    if subsolver == "cplex"
        set_optimizer(subtree.model, CPLEX.Optimizer)
        set_optimizer_attribute(subtree.model, "CPXPARAM_ScreenOutput", 0)
        set_optimizer_attribute(subtree.model, "CPXPARAM_MIP_Display", 0)
        set_optimizer_attribute(subtree.model, "CPX_PARAM_THREADS", 1)
    else
        set_optimizer(subtree.model, GLPK.Optimizer)
    end
    DD.add_block_model!(algo, block_id, subtree.model)
    models[block_id] = subtree.model
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
#LM = DD.BundleMaster(BM.TrustRegionMethod, GLPK.Optimizer)

# Solve the problem with the solver; this solver is for the underlying bundle method.
DD.run!(algo, LM)


# Write timing outputs to files
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