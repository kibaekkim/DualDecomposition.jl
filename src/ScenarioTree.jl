
"""
    AbstractTreeNode

Abstract type of tree nodes
"""

abstract type AbstractTreeNode end

"""
    TreeNode

Tree node stores information of stage problem

    - `id`: index of node
    - `stage_builder`: adds variables and constraints to model with input (model::JuMP.Model, node::SubTreeNode)
    - `parent`: index of parent node
    - `children`: indices of child nodes
    - `stage`: current stage
    - `ξ`: current scenario
    - `p`: probability of node
"""

mutable struct TreeNode <: AbstractTreeNode
    id::Int
    stage_builder::Union{Nothing,Function}
    parent::Int
    children::Vector{Tuple{Int, Float64}}
    stage::Int
    ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}

    p::Float64

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

get_id(node::AbstractTreeNode) = node.id
get_parent(node::AbstractTreeNode) = node.parent
get_children(node::AbstractTreeNode) = node.children
get_stage(node::AbstractTreeNode) = node.stage
get_scenario(node::AbstractTreeNode) = node.ξ

get_probability(node::TreeNode) = node.p

function set_stage_builder!(node::AbstractTreeNode, func::Function)
    node.stage_builder = func
end

"""
    AbstractTree

Abstract type of tree
"""

abstract type AbstractTree end

"""
    Tree

Tree keeps information of tree nodes.

    - `nodes`: list of nodes
"""

mutable struct Tree{T<:AbstractTreeNode} <: AbstractTree
    nodes::Dict{Int,T}
end

Tree(ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}) = Tree(Dict{Int,TreeNode}(1 => TreeNode(ξ)))

get_NodeType(tree::Tree{T}) where {T} = T

"""
    add_node!

adds abstract tree node to abstract tree

# Arguments
    - `tree`: Abstract tree
    - `node`: Abstract tree node
"""

function add_node!(tree::AbstractTree, node::AbstractTreeNode)
    @assert !haskey(tree.nodes, get_id(node))
    id = get_id(node)
    tree.nodes[id] = node
end

"""
    add_child!

creates child nodee and adds to tree

# Arguments
    - `tree`: Tree
    - `pt`: Parent ID
    - 'ξ': Dictionary of random variable name as symbols and scenarios
    - `prob`: probability of transitioning from the parent to this node
"""

function add_child!(tree::Tree, pt::Int, ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, prob::Float64)::Int
    #   adds child node to tree.nodes[pt]
    @assert haskey(tree.nodes, pt)                                  # check if pt is valid
    stage = get_stage(tree, pt) + 1                                 # get new stage value
    p = get_probability(tree, pt) * prob                            # get new node probability
    id = length(tree.nodes) + 1                                     # get node id
    add_node!(tree, TreeNode(id, pt, stage, ξ, p ))                 # create node and add to tree
    push!(get_children(tree, pt), (id, prob))                      # push id to parent node children
    return id
end

get_children(tree::AbstractTree, id::Int) = get_children(tree.nodes[id])
get_parent(tree::AbstractTree, id::Int) = get_parent(tree.nodes[id])
get_stage(tree::AbstractTree, id::Int) = get_stage(tree.nodes[id])
get_scenario(tree::AbstractTree, id::Int) = get_scenario(tree.nodes[id])

get_probability(tree::Tree{TreeNode}, id::Int) = get_probability(tree.nodes[id])

function set_stage_builder!(tree::AbstractTree, id::Int, func::Function)
    set_stage_builder!(tree.nodes[id], func)
end

"""
    SubTreeNode

Contains TreeNode and other information that are used for dual decomposition.
    - `treenode`: an instance of TreeNode
    - `weight`: multiplied to the objective
    - `in`: dictionary of incoming variables from the previous stage
    - `out`: dictionary of outgoing variablees to the subsequent stage
    - `obj`: objective expression
"""

mutable struct SubTreeNode <: AbstractTreeNode
    treenode::AbstractTreeNode
    weight::Float64
    in::Dict{String, JuMP.VariableRef}    # incoming variables
    out::Dict{String, JuMP.VariableRef}   # outgoing variables
    control::Dict{String, JuMP.VariableRef}   # control variables
    obj::Union{Float64, JuMP.AbstractJuMPScalar}
    function SubTreeNode(treenode::AbstractTreeNode, weight::Float64)
        stn = new()
        stn.treenode = treenode
        stn.weight = weight
        stn.in = Dict{String, JuMP.VariableRef}()
        stn.out = Dict{String, JuMP.VariableRef}()
        stn.control = Dict{String, JuMP.VariableRef}()
        stn.obj = 0.0
        return stn
    end
end

get_id(node::SubTreeNode) = get_id(node.treenode)
get_parent(node::SubTreeNode) = get_parent(node.treenode)
get_children(node::SubTreeNode) = get_children(node.treenode)
get_stage(node::SubTreeNode) = get_stage(node.treenode)
get_scenario(node::SubTreeNode) = get_scenario(node.treenode)
get_probability(node::SubTreeNode) = get_probability(node.treenode)


"""
    SubTree

Used for creeating and keeeping subproblems

    - `block_id`: ID of the subtrees
    - `nodes`: dictionary of SubTreeNodes
    - `model`: JuMP model
    - `parent`: ID of parent if exists (default is 0)
    - `root`: ID of the root node of the subtree (default is 1)
"""

mutable struct SubTree <: AbstractTree
    block_id::Int
    nodes::Dict{Int,SubTreeNode}     # list of nodes
    model::JuMP.AbstractModel 
    parent::Int
    root::Int
    function SubTree(block_id::Int)
        return new(block_id, Dict{Int,SubTreeNode}(), JuMP.Model(), 0, 1)
    end
end

"""
    create_subtree!

creates subtree from vector of nodes and adds coupling variables

# Arguments
    - `block_id`: ID of subtree
    - 'coupling_variables': list of coupling variables to be modified
    - `nodes`: vector of nodes
"""

function create_subtree!(block_id::Int, coupling_variables::Vector{CouplingVariableRef}, nodes::Vector{Tuple{TreeNode,Float64}})::SubTree
    subtree = SubTree(block_id)
    # add nodes to subtree
    for (node, weight) in nodes
        subnode = SubTreeNode(node, weight)
        add_node!(subtree, subnode)
    end
    obj = 0
    for (id, subnode) in subtree.nodes
        subnode.treenode.stage_builder(subtree.model, subnode)
        unregister_all!(subtree.model)
        obj += subnode.weight * subnode.obj
    end
    JuMP.set_objective(subtree.model, MOI.MIN_SENSE, obj)

    # 
    for (id, subnode) in subtree.nodes
        couple_common_variables!(coupling_variables, block_id, subnode)
        parent = get_parent(subnode)
        if parent!=0 && haskey(subtree.nodes, parent)
            add_links!(subtree, id, parent)
        elseif parent!=0 # assuming 1st stage node is 1
            subtree.parent = parent
            subtree.root = id
            couple_incoming_variables!(coupling_variables, block_id, subnode)
        end
    end
    return subtree
end

"""
    unregister_all!

unregisters all keys in a model so that variables with same symbols can be used for different stages

"""

function unregister_all!(model::JuMP.Model)
    dict = JuMP.object_dictionary(model)
    for (key, var) in dict
        JuMP.unregister(model, key)
    end
end

"""
    add_links!

creates linking constraints for nodes within the subtree by connecting the incoming and outgoing variables

# Arguments
    - `tree`: SubTree
    - `id`: ID of node
    - 'pt': ID of parent node
"""

function add_links!(tree::SubTree, id::Int, pt::Int)
    node = tree.nodes[id]
    parent = tree.nodes[pt]
    for (ref, var1) in node.in
        var2 = parent.out[ref]
        @constraint(tree.model, var1 .== var2)
    end
end

"""
    couple_common_variables!

couple outgoing variables

# Arguments
    - `coupling_variables`: vector of coupling variables
    - `block_id`: ID of block
    - `node`: SubTreeNode
"""

function couple_common_variables!(coupling_variables::Vector{CouplingVariableRef}, block_id::Int, node::SubTreeNode)
    label = get_id(node)
    for (ref, var) in node.out
        couple_variables!(coupling_variables, block_id, label, ref, var)
    end
end

"""
    couple_incoming_variables!

couple incoming variables with the root node of the subtree

# Arguments
    - `coupling_variables`: vector of coupling variables
    - `block_id`: ID of block
    - `child`: SubTreeNode
"""

function couple_incoming_variables!(coupling_variables::Vector{CouplingVariableRef}, block_id::Int, child::SubTreeNode)
    label = get_parent(child)
    for (ref, var) in child.in
        couple_variables!(coupling_variables, block_id, label, ref, var)
    end
end

"""
    couple_variables!

adds variables to coupling_variables

# Arguments
    - `coupling_variables`: vector of coupling variables
    - `block_id`: ID of block
    - `label`: ID of node
    - `ref`: reference for variable 
    - `var`: variable or vector of variables
"""

function couple_variables!(coupling_variables::Vector{CouplingVariableRef}, block_id::Int, label::Int, ref::String, 
        var::JuMP.VariableRef)
    name = "n$(label)_" * ref
    push!(coupling_variables, CouplingVariableRef(block_id, name, var))
end

"""
    decomposition_not

outputs the entire tree

# Arguments
    - `tree`: Tree
"""


function decomposition_not(tree::Tree{TreeNode}):: Vector{Vector{Tuple{TreeNode,Float64}}}
    nodes = Vector{Tuple{TreeNode,Float64}}()
    for (id, node) in tree.nodes
        push!(nodes,(node, get_probability(node)))
    end
    return [nodes]
end

"""
    decomposition_scenario

outputs the scenario decomposition at each leaf nodes

# Arguments
    - `tree`: Tree
"""

function decomposition_scenario(tree::Tree{TreeNode}):: Vector{Vector{Tuple{TreeNode,Float64}}}
    node_cluster = Vector{Vector{Tuple{TreeNode,Float64}}}()
    for (id, node) in tree.nodes
        if check_leaf(node)
            nodes = Vector{Tuple{TreeNode,Float64}}()
            prob = get_probability(node)
            current = node
            while true
                push!(nodes,(current, prob))
                pt = get_parent(current)
                current = tree.nodes[pt]
                if check_root(current)
                    push!(nodes,(current, prob))
                    break
                end
            end
            push!(node_cluster, nodes)
        end
    end
    return node_cluster
end

"""
    decomposition_temporal

outputs the temporal decomposition at each nodes

# Arguments
    - `tree`: Tree
"""
#need fix

function decomposition_temporal(tree::Tree{TreeNode}):: Vector{Vector{Tuple{TreeNode,Float64}}}
    node_cluster = Vector{Vector{Tuple{TreeNode,Float64}}}()
    for (id, node) in tree.nodes
        push!(node_cluster,[(node, get_probability(node))])
    end
    return node_cluster
end

function check_leaf(node::AbstractTreeNode)::Bool
    if length(get_children(node)) == 0
        return true
    else
        return false
    end
end

check_leaf(node::SubTreeNode) = check_leaf(node.treenode)

function check_root(node::AbstractTreeNode)::Bool
    if get_parent(node) == 0
        return true
    else
        return false
    end
end

check_root(node::SubTreeNode) = check_root(node.treenode)