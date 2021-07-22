
"""
Scenario Tree
"""

function add_node!(graph::Plasmo.OptiGraph, ξ:: Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}},
        pt::Union{Plasmo.OptiNode,Nothing}, prob::Float64) :: Plasmo.OptiNode
    nd = Plasmo.add_node!(graph)
    nd.ext[:parent] = pt
    nd.ext[:child] = Array{Tuple{Plasmo.OptiNode, Float64},1}()
    nd.ext[:stage] = pt.ext[:stage] + 1
    nd.ext[:ξ] = ξ
    nd.ext[:p] = pt.ext[:p] * prob
    nd.ext[:in] = Dict{Symbol, Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}}}()
    nd.ext[:out] = Dict{Symbol, Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}}}()

    push!(pt.ext[:child], (nd, prob))
    return nd
end

function add_node!(graph::Plasmo.OptiGraph, ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}) :: Plasmo.OptiNode
    nd = Plasmo.add_node!(graph)
    nd.ext[:parent] = nothing
    nd.ext[:child] = Array{Tuple{Plasmo.OptiNode, Float64},1}()
    nd.ext[:stage] = 1
    nd.ext[:ξ] = ξ
    nd.ext[:p] = 1.0
    nd.ext[:in] = Dict{Symbol, Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}}}()
    nd.ext[:out] = Dict{Symbol, Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}}}()
    return nd
end

function set_input_variable!(nd::Plasmo.OptiNode, symb::Symbol, var::Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}})
    nd.ext[:in][symb] = var
end

function set_output_variable!(nd::Plasmo.OptiNode, symb::Symbol, var::Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}})
    nd.ext[:out][symb] = var
end

struct SubTree
    tree::Plasmo.OptiGraph
    parent::Union{Plasmo.OptiNode,Nothing}
end

function create_subtree(graph = Plasmo.OptiGraph, nodes::Vector{Plasmo.OptiNode})::SubTree
    subtree = Plasmo.OptiGraph()
    nodedict = Dict{Int64,Plasmo.OptiNode}()
    # add nodes to subtree and create dictionary
    for node in nodes
        Plasmo.add_node!(subtree, node)
        nodeidx = getindex(graph, node)
        nodedict[nodeidx] = node
    end
    # create edges and get parent of subtree
    subtree_parent = nothing
    for node in nodes
        if !isnothing(node.ext[:parent])
            parentidx = getindex(graph, node.ext[:parent])
            if haskey(nodedict, parentidx)
                parent = nodedict[parentidx]
                link_variables_directed!(subtree, node, parent)
            else 
                subtree_parent = node.ext[:parent]
            end
        end
    end
    return SubTree(subtree, subtree_parent)
end

function link_variables_directed!(graph::Plasmo.OptiGraph, child::Plasmo.OptiNode, parent::Plasmo.OptiNode)
    for (symb, var1) in child.ext[:in]
        var2 = parent.ext[:out][symb]
        @linkconstraint(graph, var1 .== var2)
    end
end

function link_variables_common!(graph::Plasmo.OptiGraph, node1::Plasmo.OptiNode, node2::Plasmo.OptiNode)
    for (symb, var1) in node1.ext[:out]
        var2 = node2.ext[:out][symb]
        @linkconstraint(graph, var1 .== var2)
    end
end

function check_leaf(node::Plasmo.OptiNode)::Bool
    if length(node.ext[:child]) == 0
        return true
    else
        return false
    end
end




struct TreeNode
    parent::Int                                 # index of parent node
    children::Vector{Int}                       # indices of child nodes
    stage::Int                                  # current stage
    ξ::Dict{Symbol,Any}                         # current scenario
    p::Float64                                  # probability of node
end

abstract type AbstractTree end

mutable struct Tree <: AbstractTree
    nodes::Dict{Int,TreeNode}     # list of nodes
    depth::Int                  # depth of tree
end

Tree(ξ::Dict{Symbol,Any}) = Tree(Dict{1 => TreeNode(0, Vector{Int}(), 1, ξ, 1.0 )}, 1)

function add_node!(tree::Tree, node::TreeNode)
    id = length(tree.nodes)
    tree.nodes[id] = node
    return id
end

function add_child!(tree::Tree, pt::Int, ξ::Dict{Symbol,Any}, prob::Float64)
    #   adds child node to tree.nodes[pt]
    1 <= pt <= length(tree.nodes) || throw(BoundsError(tree, pt))   # check if pt is valid
    stage = get_stage(tree, pt) + 1                                 # get new stage value
    p = get_probability(tree, pt) * prob                            # get new node probability
    node = TreeNode(pt, Vector{Int}(), stage, ξ, p )                # create node
    child_id = add_node!(tree, node)                                # add to tree and get node ID
    push!(tree.nodes[pt].children, child_id)                        # push child_id to parent node children
    if stage > tree.depth
        tree.depth = stage                                          # update length of tree to the maximum value
    end
end

get_children(tree, id) = tree.nodes[id].children
get_parent(tree,id) = tree.nodes[id].parent
get_stage(tree, id) = tree.nodes[id].stage
get_scenario(tree, id) = tree.nodes[id].ξ
get_probability(tree, id) = tree.nodes[id].p
get_node(tree, id) = tree.nodes[id]



function get_history(tree::AbstractTree, id::Int)::Array{Int}
    # gets a vector of tree node IDs up until current
    stage = get_stage(tree, id)
    hist = Array{Int}(undef, stage)

    current_id = id
    for k = stage:-1:1
        hist[k] = current_id
        current_id = get_parent(tree, current_id)
    end
    return hist
end

function get_future(tree::AbstractTree, root_id::Int)::Array{Int}
    #   output list of all leaf node IDs branching from root_id
    arr_leaves = Int[]

    function iterate_children(tree::AbstractTree, id::Int)
        children = get_children(tree, id)
        if length(children) == 0
            #buffer output
            push!(arr_leaves, id)
        else
            for child in children
                iterate_children(tree, child)
            end
        end
    end

    iterate_children(tree, root_id)
    return arr_leaves
end

function get_stage_id(tree::AbstractTree)::Array{Array{Int}}
    # gets a list of tree node IDs separated by stages
    K = tree.K
    nodelist = [ Int[] for _ in 1:K]

    for id in 1:length(tree.nodes)
        k = get_stage(tree, id)
        push!(nodelist[k], id)
    end
    return nodelist
end

function leaf2block(nodes::Array{Int})::Dict{Int,Int}
    leafdict = Dict{Int,Int}()
    for i in 1:length(nodes)
        id = nodes[i]
        leafdict[id] = i
    end
    return leafdict
end