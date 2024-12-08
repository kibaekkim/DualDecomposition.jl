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
using JuMP
using Random, Distributions
using ArgParse

const DD = DualDecomposition
const parallel = DD.parallel

settings = ArgParseSettings()
@add_arg_table settings begin
    "--nI"
        help = "number of resources"
        arg_type = Int
        default = 2
    "--nT"
        help = "number of time periods"
        arg_type = Int
        default = 3
    "--nS"
        help = "number of branches per stage"
        arg_type = Int
        default = 2
    "--decmode"
        help = "decomposition mode:\n
                -0 no decomposition\n
                -1 scenario decomposition\n
                -2 nodal decomposition
                "
        arg_type = Int
        default = 1
end

include("parser.jl")

nI = parsed_args["nI"]
nT = parsed_args["nT"]
nS = parsed_args["nS"]

decmode = parsed_args["decmode"]

seed::Int = 1

function create_root(nI::Int, nT::Int, nS::Int)::DD.Tree
    ξ = Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}(
        :α => ones(nI)*10,
        :β => ones(nI)*100, 
        :d => 50.0)

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
    α_ = [ ceil(α[i] * exp((-(0.25*i)^2/2) + 0.25*i*rand(normal))) for i in 1:nI]
    β = DD.get_scenario(tree, pt)[:β]
    β_ = [ ceil(β[i] * exp((-1-(0.25*i)^2/2) + 0.25*i*rand(normal))) for i in 1:nI]
    d = DD.get_scenario(tree, pt)[:d]
    d_ = ceil(d * exp((1-1/2) + rand(normal)))
    return α_, β_, d_
end


Random.seed!(seed)
tree = create_root(nI, nT, nS)
if decmode == 0
    node_cluster = DD.decomposition_not(tree)
elseif decmode == 1
    node_cluster = DD.decomposition_scenario(tree)
elseif decmode == 2
    node_cluster = DD.decomposition_temporal(tree)
end
NS = length(node_cluster)

function create_sub_model!(block_id::Int64, coupling_variables::Vector{DD.CouplingVariableRef})
    nodes = node_cluster[block_id]
    subtree = DD.create_subtree!(tree, block_id, coupling_variables, nodes)
    return subtree.model
end

include("core.jl")

