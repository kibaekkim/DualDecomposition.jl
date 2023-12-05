#=
Source:
S. Ahmed, A. King, and G. Parija. "A Multi-Stage Stochastic Integer Programming Approach for Capacity Expansion under Uncertainty". Journal of Global Optimization, vol. 26, pp. 3-24, 2003}.

Input:
  nI: number of resource types
  nT: number of time periods
  nS: number of scenarios per period

Sets:
  sI: resources
  sT: time periods
  sN: nodes in scenario tree

Variables:
  x[i,n]: capacity acquired for resource i at node n
  y[i,n]: 1 if resource i is capacity is acquired at node n, 0 otherwise

Parameters:
  α[i,n]: discounted variable investment cost for resource i at node n
  β[i,n]: discounted fixed investment cost for resource i at node n
  d[n]  : depand at node n

=#

using DualDecomposition
using JuMP, Ipopt, GLPK
using Random, Distributions
using ArgParse

const DD = DualDecomposition
const parallel = DD.parallel

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--alg"
            help = "algorithm mode:\n
                    -0: constant ρ\n
                    -1: residual balancing\n
                    -2: adaptive residual balancing\n
                    -3: relaxed ADMM\n
                    -4: adaptive relaxed ADMM"
            arg_type = Int
            default = 1
        "--nI"
            help = "number of resources"
            arg_type = Int
            default = 2
        "--nT"
            help = "number of time periods"
            arg_type = Int
            default = 3
        "--nS"
            help = "number of scenarios"
            arg_type = Int
            default = 2
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

alg = parsed_args["alg"]
nI = parsed_args["nI"]
nT = parsed_args["nT"]
nS = parsed_args["nS"]
rho = parsed_args["rho"]
tol = parsed_args["tol"]
tau = parsed_args["tau"]
uinterval = parsed_args["interval"]
age = parsed_args["age"]
dir = parsed_args["dir"]
seed::Int = 1

function create_root(nI::Int, nT::Int, nS::Int)::DD.Tree
    ξ = Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}(
        :α => ones(nI),
        :β => ones(nI)*10, 
        :d => 5.0)

    tree = DD.Tree(ξ)

    function subproblem_builder(tree::DD.Tree, subtree::DD.SubTree, node::DD.SubTreeNode)
        mdl = subtree.model
        x = @variable(mdl, x[i=1:nI] >= 0, base_name="n1_x")
        y = @variable(mdl, y[i=1:nI], Bin, base_name="n1_y")
        z = @variable(mdl, z[i=1:nI], base_name="n1_z")
        DD.set_output_variable!(node, :z, z)

        α = DD.get_scenario(node)[:α]
        β = DD.get_scenario(node)[:β]
        d = DD.get_scenario(node)[:d]
        M = d
        @constraints(mdl,
            begin
                [i=1:nI], x[i] <= M * y[i]
                [i=1:nI], z[i] == x[i]
                sum( z[i] for i in 1:nI ) >= d
            end
        )
        DD.set_stage_objective(node, sum( α[i] * x[i] for i in 1:nI) + sum( β[i] * y[i] for i in 1:nI) )

        JuMP.unregister(mdl, :x)
        JuMP.unregister(mdl, :y)
        JuMP.unregister(mdl, :z)
    end
    DD.set_stage_builder!(tree, 1, subproblem_builder)

    create_children!(tree, nI, nT, nS, 1)
    return tree
end

# child nodes
function create_children!(tree::DD.Tree, nI, nT::Int, nS::Int, pt::Int)
    for s = 1:nS
        prob = 1/nS
        α_, β_, d_ = create_params(tree, nI, pt)
        ξ = Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}(
        :α => α_,
        :β => β_,
        :d => d_)
        id = DD.add_child!(tree, pt, ξ, prob)

        function subproblem_builder(tree::DD.Tree, subtree::DD.SubTree, node::DD.SubTreeNode)
            mdl = subtree.model
            id = DD.get_id(node)
            x = @variable(mdl, x[i=1:nI] >= 0, base_name="n$(id)_x")
            y = @variable(mdl, y[i=1:nI], Bin, base_name="n$(id)_y")
            z = @variable(mdl, z[i=1:nI], base_name="n$(id)_z")
            DD.set_output_variable!(node, :z, z)

            z_ = @variable(mdl, z_[i=1:nI], base_name="n$(id)_z_") 
            DD.set_input_variable!(node, :z, z_)
    
            α = DD.get_scenario(node)[:α]
            β = DD.get_scenario(node)[:β]
            d = DD.get_scenario(node)[:d]
            M = ceil(d)
            @constraints(mdl,
                begin
                    [i=1:nI], x[i] <= M * y[i]
                    [i=1:nI], z[i] == z_[i] + x[i]
                    sum( z[i] for i in 1:nI ) >= d
                end
            )
            DD.set_stage_objective(node, sum( α[i] * x[i] for i in 1:nI) + sum( β[i] * y[i] for i in 1:nI) )
    
            JuMP.unregister(mdl, :x)
            JuMP.unregister(mdl, :y)
            JuMP.unregister(mdl, :z)
            JuMP.unregister(mdl, :z_)
        end

        DD.set_stage_builder!(tree, id, subproblem_builder)
        if DD.get_stage(tree, id) < nT
            create_children!(tree, nI, nT, nS, id)
        end
    end
end

function create_params(tree::DD.Tree, nI::Int, pt::Int)
    #α: μ = 0, σ = 0.25, 0.50, 0.75,...
    #β: μ =-1, σ = 0.25, 0.50, 0.75,...
    #d: μ = 1, σ = 1
    normal = Normal()
    α = DD.get_scenario(tree, pt)[:α]
    α_ = [ α[i] * exp((-(0.25*i)^2/2) + 0.25*i*rand(normal)) for i in 1:nI]
    β = DD.get_scenario(tree, pt)[:β]
    β_ = [ β[i] * exp((-1-(0.25*i)^2/2) + 0.25*i*rand(normal)) for i in 1:nI]
    d = DD.get_scenario(tree, pt)[:d]
    d_ = d * exp((1-1/2) + rand(normal)) 
    return α_, β_, d_
end


Random.seed!(seed)
tree = create_root(nI, nT, nS)
node_cluster = DD.decomposition_scenario(tree)
NS = length(node_cluster)


# Initialize MPI
parallel.init()

# Create DualDecomposition instance.
params = BM.Parameters()
BM.set_parameter(params, "print_output", false)
BM.set_parameter(params, "max_age", age)
algo = DD.AdmmLagrangeDual(BM.BasicMethod, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0), params)


# partition scenarios into processes
parallel.partition(nS)

coupling_variables = Vector{DD.CouplingVariableRef}()
models = Dict{Int,JuMP.Model}()

for block_id in parallel.getpartition()
    nodes = node_cluster[block_id]
    subtree = DD.create_subtree!(tree, block_id, coupling_variables, nodes)
    set_optimizer(subtree.model, GLPK.Optimizer)
    DD.add_block_model!(algo, block_id, subtree.model)
    models[block_id] = subtree.model
end

# Set nonanticipativity variables as an array of symbols.
DD.set_coupling_variables!(algo, coupling_variables)

# Solve the problem with the solver; this solver is for the underlying bundle method.
LM = DD.AdmmMaster(alg=alg, ρ=rho, ϵ=tol, maxiter=100000, update_interval = uinterval, τ=tau)

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

