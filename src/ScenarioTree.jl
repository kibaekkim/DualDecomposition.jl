
"""
Scenario Tree
"""

"""
function add_node!(graph::Plasmo.OptiGraph, ξ:: Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}},
        pt::Union{Plasmo.OptiNode,Nothing} = nothing, prob::Float64 = 1.0) :: Plasmo.OptiNode
    nd = Plasmo.add_node!(graph)
    nd.ext[:parent] = pt
    nd.ext[:child] = Array{Tuple{Plasmo.OptiNode, Float64},1}()
    nd.ext[:ξ] = ξ
    nd.ext[:in] = Dict{Symbol, Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}}}()
    nd.ext[:out] = Dict{Symbol, Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}}}()
    if isnothing(pt)
        nd.ext[:stage] = 1
        nd.ext[:p] = 1.0
    else
        nd.ext[:stage] = pt.ext[:stage] + 1
        nd.ext[:p] = pt.ext[:p] * prob
        push!(pt.ext[:child], (nd, prob))
    end
    return nd
end

function set_input_variable!(nd::Plasmo.OptiNode, symb::Symbol, var::Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}})
    nd.ext[:in][symb] = var
end

function set_output_variable!(nd::Plasmo.OptiNode, symb::Symbol, var::Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}})
    nd.ext[:out][symb] = var
end

mutable struct Subtree
    tree::Plasmo.OptiGraph
    parent::Union{Plasmo.OptiNode,Nothing}
    child::Union{Plasmo.OptiNode,Nothing}
end

function create_subtree(graph::Plasmo.OptiGraph, nodes::Vector{Plasmo.OptiNode})::Subtree
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
    subtree_child = nothing
    for node in nodes
        if !isnothing(node.ext[:parent])
            parentidx = getindex(graph, node.ext[:parent])
            if haskey(nodedict, parentidx)
                parent = nodedict[parentidx]
                link_variables!(subtree, node, parent)
            else 
                subtree_parent = node.ext[:parent]
                subtree_child = node
            end
        end
    end
    return Subtree(subtree, subtree_parent, subtree_child)
end

function link_variables!(graph::Plasmo.OptiGraph, child::Plasmo.OptiNode, parent::Plasmo.OptiNode)
    for (symb, var1) in child.ext[:in]
        var2 = parent.ext[:out][symb]
        @linkconstraint(graph, var1 .== var2)
    end
end

function couple_common_variables!(coupling_variables::Vector{CouplingVariableRef}, block_id::Int, node::Plasmo.OptiNode)
    label = node.label
    for (symb, var) in node.ext[:out]
        couple_variables!(coupling_variables, block_id, label, symb, var)
    end
end

function couple_incoming_variables!(coupling_variables::Vector{CouplingVariableRef}, block_id::Int, child::Plasmo.OptiNode, parent::Plasmo.OptiNode)
    label = parent.label
    for (symb, var) in child.ext[:in]
        couple_variables!(coupling_variables, block_id, label, symb, var)
    end
end
"""




abstract type AbstractTreeNode end

mutable struct TreeNode <: AbstractTreeNode
    id::Int                                                                         # label of node
    stage_builder::Union{Nothing,Function} # with input (tree::SubTree, node::SubTreeNode)
    parent::Int                                                                     # index of parent node
    children::Vector{Tuple{Int, Float64}}                                           # indices of child nodes
    stage::Int                                                                      # current stage
    ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}                        # current scenario
    p::Float64                                                                      # probability of node

    function TreeNode(id::Int, parent::Int, stage::Int, ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, p::Float64)
        tn = new()
        tn.id = id
        tn.stage_builder = nothing
        tn.parent = parent
        tn.children = Vector{Tuple{Int, Float64}}()
        tn.stage = stage
        tn.ξ = ξ
        tn.p = p
        return tn
    end
end

function TreeNode(ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}})
    return TreeNode(1, 0, 1, ξ, 1.0)
end

get_id(node::TreeNode) = node.id
get_parent(node::TreeNode) = node.parent
get_children(node::TreeNode) = node.children
get_stage(node::TreeNode) = node.stage
get_scenario(node::TreeNode) = node.ξ
get_probability(node::TreeNode) = node.p
function set_stage_builder!(node::TreeNode, func::Function)
    node.stage_builder = func
end

abstract type AbstractTree end

mutable struct Tree <: AbstractTree
    nodes::Dict{Int,TreeNode}     # list of nodes
    depth::Int                  # depth of tree
end

Tree(ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}) = Tree(Dict{1 => TreeNode(ξ)}, 1)

function add_node!(tree::Tree, node::TreeNode)
    @assert !haskey(tree.nodes, get_id(node))
    id = get_id(node)
    tree.nodes[id] = node
end

function add_child!(tree::Tree, pt::Int, ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, prob::Float64)::Float64
    #   adds child node to tree.nodes[pt]
    @assert haskey(tree.nodes, pt)                                  # check if pt is valid
    stage = get_stage(tree, pt) + 1                                 # get new stage value
    p = get_probability(tree, pt) * prob                            # get new node probability
    id = length(tree.nodes) + 1                                     # get node id
    add_node!(tree, TreeNode(id, pt, stage, ξ, p ))                 # create node and add to tree
    push!(get_children(tree, pt), (id, prob))                      # push id to parent node children
    if stage > tree.depth
        tree.depth = stage                                          # update length of tree to the maximum value
    end
    return id
end

get_children(tree, id) = get_children(tree.nodes[id])
get_parent(tree, id) = get_parent(tree.nodes[id])
get_stage(tree, id) = get_stage(tree.nodes[id])
get_scenario(tree, id) = get_scenario(tree.nodes[id])
get_probability(tree, id) = get_probability(tree.nodes[id])
function set_stage_builder!(tree, id, func::Function)
    set_stage_builder!(tree.nodes[id], func)
end

mutable struct SubTreeNode <: AbstractTreeNode
    treenode::TreeNode
    weight::Float64
    in::Dict{Symbol, Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}}}    # incoming variables
    out::Dict{Symbol, Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}}}   # outgoing variables
    obj::JuMP.AbstractJuMPScalar
    function SubTreeNode(treenode::TreeNode, weight::Float64)
        stn = new()
        stn.treenode = treenode
        stn.weight = weight
        stn.in = Dict{Symbol, Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}}}()
        stn.out = Dict{Symbol, Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}}}()
        stn.obj = 0
        return stn
    end
end

get_id(node::SubTreeNode) = get_id(node.treenode)
get_parent(node::SubTreeNode) = get_parent(node.treenode)
get_children(node::SubTreeNode) = get_children(node.treenode)
get_stage(node::SubTreeNode) = get_stage(node.treenode)
get_scenario(node::SubTreeNode) = get_scenario(node.treenode)
get_probability(node::SubTreeNode) = get_probability(node.treenode)

function set_input_variable!(nd::SubTreeNode, symb::Symbol, var::Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}})
    nd.in[symb] = var
end

function set_output_variable!(nd::SubTreeNode, symb::Symbol, var::Union{JuMP.VariableRef, <:AbstractArray{JuMP.VariableRef}})
    nd.out[symb] = var
end

function set_stage_objective(nd::SubTreeNode, obj::JuMP.AbstractJuMPScalar)
    nd.obj = obj
end

mutable struct SubTree <: AbstractTree
    block_id::Int
    nodes::Dict{Int,TreeNode}     # list of nodes
    model::JuMP.AbstractModel 
    parent::Int
    root::Int
    function SubTree(block_id::Int)
        return new(block_id, Dict{Int,TreeNode}(), JuMP.Model(), 0, 1)
    end
end

function create_subtree!(block_id::Int, sense::MOI.OptimizationSense, coupling_variables::Vector{CouplingVariableRef}, nodes::Vector{Tuple{TreeNode,Float64}})::SubTree
    subtree = SubTree(block_id)
    obj = 0
    # add nodes to subtree
    for (node, weight) in nodes
        subnode = SubTreeNode(node, weight)
        add_node!(subtree, subnode)
        obj += subnode.weight * subnode.obj
    end
    set_objective(subtree, sense, obj)
    # 
    for subnode in subtree.nodes
        id = get_id(subnode)
        couple_common_variables!(coupling_variables, block_id, subnode)
        parent = get_parent(subnode)
        if parent!=0 & haskey(subtree.nodes, parent)
            add_links!(subtree, id, parent)
        elseif parent!=0 # assuming 1st stage node is 1
            subtree.parent = parent
            subtree.root = id
            couple_incoming_variables!(coupling_variables, block_id, subnode)
        end
    end
    return subtree
end

function add_node!(tree::SubTree, node::SubTreeNode)
    @assert !haskey(tree.nodes, get_id(node))
    id = get_id(node)
    tree.nodes[id] = node
    node.treenode.stage_builder(tree, node) # make macro for changing variable names and constraint names to include node id
end

function add_links!(tree::SubTree, id::Int, pt::Int)
    node = tree.nodes[id]
    parent = tree.nodes[pt]
    for (symb, var1) in node.in
        var2 = parent.out[symb]
        @constraint(tree.model, var1 .== var2)
    end
end

function set_objective!(tree::SubTree, sense::MOI.OptimizationSense, obj::JuMP.AbstractJuMPScalar)
    JuMP.set_objective(tree.model, sense, obj)
end


function couple_common_variables!(coupling_variables::Vector{CouplingVariableRef}, block_id::Int, node::SubTreeNode)
    label = get_id(node)
    for (symb, var) in node.out
        couple_variables!(coupling_variables, block_id, label, symb, var)
    end
end

function couple_incoming_variables!(coupling_variables::Vector{CouplingVariableRef}, block_id::Int, child::SubTreeNode)
    label = get_parent(child)
    for (symb, var) in child.in
        couple_variables!(coupling_variables, block_id, label, symb, var)
    end
end

function couple_variables!(coupling_variables::Vector{CouplingVariableRef}, block_id::Int, label::String, symb::Symbol, 
        var::JuMP.VariableRef)
    push!(coupling_variables, CouplingVariableRef(block_id, [label, symb], var))
end

function couple_variables!(coupling_variables::Vector{CouplingVariableRef}, block_id::Int, label::String, symb::Symbol, 
        var::Array{JuMP.VariableRef})
    for (index, value) in pairs(var)
        push!(coupling_variables, CouplingVariableRef(block_id, [label, symb, Tuple(index)], value))
    end
end

function couple_variables!(coupling_variables::Vector{CouplingVariableRef}, block_id::Int, label::String, symb::Symbol, 
        var::JuMP.Containers.DenseAxisArray{JuMP.VariableRef})
    for (index, value) in pairs(var.data)
        push!(coupling_variables, CouplingVariableRef(block_id, [label, symb, keys(var)[index].I], value))
    end
end

function couple_variables!(coupling_variables::Vector{CouplingVariableRef}, block_id::Int, label::String, symb::Symbol, 
        var::JuMP.Containers.SparseAxisArray{JuMP.VariableRef})
    for (index, value) in var.data
        push!(coupling_variables, CouplingVariableRef(block_id, [label, symb, index], value))
    end
end



function check_leaf(node::Plasmo.OptiNode)::Bool
    if length(node.ext[:child]) == 0
        return true
    else
        return false
    end
end

"""
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
"""