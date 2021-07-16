
"""
Scenario Tree
"""

function add_node!(graph::Plasmo.OptiGraph, ξ::Any, pt::Union{Plasmo.OptiNode,Nothing}, prob::Float64) :: Plasmo.OptiNode
    nd = Plasmo.add_node!(graph)
    nd.ext[:parent] = pt
    nd.ext[:child] = Array{Tuple{Plasmo.OptiNode, Float64},1}()
    nd.ext[:stage] = pt.ext[:stage] + 1
    nd.ext[:ξ] = ξ
    nd.ext[:p] = pt.ext[:p] * prob
    nd.ext[:out] = Dict{JuMP.VariableRef,Array{JuMP.VariableRef,1}}()

    push!(pt.ext[:child], (nd, prob))
    return nd
end

function add_node!(graph::Plasmo.OptiGraph, ξ::Any) :: Plasmo.OptiNode
    nd = Plasmo.add_node!(graph)
    nd.ext[:parent] = nothing
    nd.ext[:child] = Array{Tuple{Plasmo.OptiNode, Float64},1}()
    nd.ext[:stage] = 1
    nd.ext[:ξ] = ξ
    nd.ext[:p] = 1.0
    nd.ext[:out] = Dict{JuMP.VariableRef,Array{JuMP.VariableRef,1}}()
    return nd
end

function set_state_variables!(nd::Plasmo.OptiNode, ndvar::JuMP.VariableRef)
    nd.ext[:out][ndvar] = Array{JuMP.VariableRef,1}()
end

function link_variables!(graph::Plasmo.OptiGraph, pt::Plasmo.OptiNode, ptvar::JuMP.VariableRef, ndvar::JuMP.VariableRef)
    #assuming that out-going variables are all the same, add out-going variable references
    push!(pt.ext[:out][ptvar], ndvar)
    @linkconstraint(graph, ptvar == ndvar)
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
    k::Int                                      # current stage
    ξ::Vector{Float64}                          # current scenario
end

abstract type AbstractTree end

mutable struct Tree <: AbstractTree
    nodes::Vector{TreeNode}     # list of nodes
    K::Int                      # length of tree
end

Tree(ξ::Vector{Float64}) = Tree([TreeNode(0, Vector{Int}(), 1, ξ )], 1)

function addchild!(tree::Tree, id::Int, ξ::Vector{Float64})
    #   adds child node to tree.nodes[id]
    1 <= id <= length(tree.nodes) || throw(BoundsError(tree, id))   # check if id is valid
    k = get_stage(tree, id) + 1                                     # get new stage value
    push!(tree.nodes, TreeNode(id, Vector{}(), k, ξ ))              # push to node list
    child_id = length(tree.nodes)                                   # get current node ID
    push!(tree.nodes[id].children, child_id)                        # push child_id to parent node children
    if k > tree.K
        tree.K = k  # update length of tree to the maximum value
    end
end

get_children(tree, id) = tree.nodes[id].children
get_parent(tree,id) = tree.nodes[id].parent
get_stage(tree, id) = tree.nodes[id].k
get_scenario(tree, id) = tree.nodes[id].ξ

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