using JuMP, Ipopt, GLPK
using DualDecomposition
using Random

const DD = DualDecomposition

"""
    Scenario Tree
"""

struct TreeNode
    parent::Int                                 # index of parent node
    children::Vector{Int}                       # indices of child nodes
    k::Int                                      # current stage
    ξ::Vector{Float64}                          # current scenario
end
mutable struct Tree
    nodes::Vector{TreeNode}     # list of nodes
    K::Int                      # length of tree
end

Tree(ξ::Vector{Float64}) = 
    Tree([TreeNode(0, Vector{Int}(), 1, ξ )], 1)

function addchild!(tree::Tree, id::Int, ξ::Vector{Float64})
    #   adds child node to tree.nodes[id]
    1 <= id <= length(tree.nodes) || throw(BoundsError(tree, id))   # check if id is valid
    k = get_stage(tree, id) + 1                                     # get new stage value
    push!(tree.nodes, TreeNode(id, Vector{}(), k, ξ ))   # push to node list
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

function get_history(tree::Tree, id::Int)::Array{Int}
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

function get_future(tree::Tree, root_id::Int)::Array{Int}
    #   output list of all leaf node IDs branching from root_id
    arr_leaves = Int[]

    function iterate_children(tree::Tree, id::Int)
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

function get_stage_id(tree::Tree)::Array{Array{Int}}
    # gets a list of tree node IDs separated by stages
    K = tree.K
    nodelist = [ Int[] for _ in 1:K]

    for id in 1:length(tree.nodes)
        k = get_stage(tree, id)
        push!(nodelist[k], id)
    end
    return nodelist
end


"""
a: interest rate
π: unit stock price
ρ: unit dividend price


K = 3 number of stages
L = 2 number of investment vehicles
2^L scenarios in each stage
2^L^(K-1)=16 scenarios in total
ρ = 0.05 * π
bank: interest rate 0.01
asset1: 1.03 or 0.97
asset2: 1.06 or 0.94

"""

function create_tree(K::Int, L::Int)::Tree
    π = ones(L)                              
    tree = Tree(π)
    add_nodes!(K, L, tree, 1, 1)
    return tree
end

function add_nodes!(K::Int, L::Int, tree::Tree, id::Int, k::Int)
    if k < K-1
        ls = iterlist(L,tree.nodes[id].ξ)
        for π in ls
            addchild!(tree, id, π)
            childid = length(tree.nodes)
            add_nodes!(K, L, tree, childid, k+1)
        end
    elseif k == K-1
        ls = iterlist(L,tree.nodes[id].ξ)
        for π in ls
            addchild!(tree, id, π)
        end
    end
end

function iterlist(L::Int, π::Array{Float64})::Array{Array{Float64}}
    # generates all combinations of up and down scenarios
    ls = [Float64[] for _ in 1:2^L]
    ii = 1

    function foo(L::Int, l::Int, arr::Vector{Float64})
        up = (1.0 + 0.03 * l) * π[l]
        dn = (1.0 - 0.03 * l) * π[l]

        if l < L
            arr1 = copy(arr)
            arr1[l] = up
            foo(L, l+1, arr1)

            arr2 = copy(arr)
            arr2[l] = dn
            foo(L, l+1, arr2)
        else
            arr1 = copy(arr)
            arr1[l] = up
            ls[ii] = arr1
            ii+=1

            arr2 = copy(arr)
            arr2[l] = dn
            ls[ii] = arr2
            ii+=1
        end
    end

    foo(L, 1, Array{Float64}(undef, L))
    return ls
end

"""
    max     B_K+∑_{l=1}^{L}π_{K,l}y_{K,l}

    s.t.    B_1+∑_{l=1}^{L}π_{1,l}x_{1,l} = b_1

            b_k+(1+a)B_{k-1}+∑_{l=1}^{L}ρ_{k,l}y_{k-1,l} = B_k+∑_{l=1}^{L}π_{k,l}x_{k,l}, ∀ k=2,…,K
    
            y_{1,l} = x_{1,l}, ∀ l=1,…,L
    
            y_{k-1,l}+x_{k,l} = y_{k,l}, ∀ k=2,…,K, l=1,…,L
    
            x_{k,l} ∈ ℤ , ∀ k=1,…,K, l=1,…,L
    
            y_{k,l} ≥ 0, ∀ k=1,…,K, l=1,…,L
    
            B_k ≥ 0, ∀ k=1,…,K.
"""
const K = 3
const L = 2
const a = 0.01
const b1 = 100  # initial capital
const b2 = 30   # income

function create_scenario_model(K::Int, L::Int, tree::Tree, id::Int)
    hist = get_history(tree, id)
    m = Model(GLPK.Optimizer) 
    #@variable(m, x[1:K,1:L], integer=true)
    @variable(m, x[1:K,1:L])
    @variable(m, y[1:K,1:L]>=0)
    @variable(m, B[1:K]>=0)

    π = tree.nodes[1].ξ

    @constraint(m, B[1] + sum( π[l] * x[1,l] for l in 1:L) == b1)

    for l in 1:L
        @constraint(m, y[1,l]-x[1,l]==0)
    end

    for k = 2:K
        π = tree.nodes[hist[k]].ξ
        ρ = tree.nodes[hist[k-1]].ξ * 0.05

        @constraint(m, B[k] + sum( π[l] * x[k,l] - ρ[l] * y[k-1,l] for l in 1:L)
            - (1+a) * B[k-1] == b2)
        for l in 1:L
            @constraint(m, y[k,l]-x[k,l]-y[k-1,l]==0)
        end
    end
    for l in 1:L
        @constraint(m, x[K,l]==0)
    end
    π = tree.nodes[id].ξ
    @objective(m, Min, - (B[K] + sum( π[l] * y[K,l] for l in 1:L ))/(2^L)^(K-1) )
    return m
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

The main computation section

"""
function main_comp()
    # generate tree data structure
    tree = create_tree(K,L)

    # compute dual decomposition method
    LD = dual_decomp(L, tree)
end


function dual_decomp(L::Int, tree::Tree)
    # Create DualDecomposition instance.
    algo = DD.LagrangeDual(BM.TrustRegionMethod)
    #algo = DD.LagrangeDual()

    # Add Lagrange dual problem for each scenario s.
    nodelist = get_stage_id(tree)
    leafdict = leaf2block(nodelist[K])
    models = Dict{Int,JuMP.Model}(id => create_scenario_model(K,L,tree,id) for id in nodelist[K])
    for id in nodelist[K]
        DD.add_block_model!(algo, leafdict[id], models[id])
    end

    coupling_variables = Vector{DD.CouplingVariableRef}()
    for k in 1:K-1
        for root in nodelist[k]
            leaves = get_future(tree, root)
            for id in leaves
                model = models[id]
                yref = model[:y]
                for l in 1:L
                    push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [root, l], yref[k, l]))
                end
                Bref = model[:B]
                push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [root, L+1], Bref[k]))
            end
        end
    end
    # dummy coupling variables
    for id in nodelist[K]
        model = models[id]
        yref = model[:y]
        for l in 1:L
            push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [id, l], yref[K, l]))
        end
        Bref = model[:B]
        push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [id, L+1], Bref[K]))
    end

    # Set nonanticipativity variables as an array of symbols.
    DD.set_coupling_variables!(algo, coupling_variables)

    # Solve the problem with the solver; this solver is for the underlying bundle method.
    DD.run!(algo, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    return algo
end

