"""
DR_LagrangeDual

Lagrangian dual method for dual decomposition. This `mutable struct` constains:
- `block_model::BlockModel` object
- `var_to_index` mapping coupling variable to the index wrt the master problem
- `masiter::Int` sets the maximum number of iterations
- `tol::Float64` sets the relative tolerance for termination
- `tree` keeps Tree information
"""

mutable struct DR_LagrangeDual <: AbstractLagrangeDual
    block_model::DR_BlockModel
    var_to_index::Dict{Tuple{Int,Any},Int} # maps coupling variable to the index wrt the master problem
    heuristics::Vector{Type}
    subsolve_time::Vector{Dict{Int,Float64}}
    subcomm_time::Vector{Float64}
    subobj_value::Vector{Float64}
    master_time::Vector{Float64}

    tree::Union{Nothing,Tree{DR_TreeNode}}
    subtrees::Union{Nothing,Dict{Int,SubTree}}
    P_model::Union{Nothing, JuMP.Model}

    function DR_LagrangeDual()
        LD = new()
        LD.block_model = DR_BlockModel()
        LD.var_to_index = Dict()
        LD.heuristics = []
        LD.subsolve_time = []
        LD.subcomm_time = []
        LD.subobj_value = []
        LD.master_time = []

        LD.tree = nothing
        LD.subtrees = nothing
        LD.P_model = nothing
        
        return LD
    end
end

function DR_LagrangeDual(tree::Tree{DR_TreeNode})
    LD = DR_LagrangeDual()
    add_tree!(LD, tree)
    return LD
end

function get_solution!(LD::DR_LagrangeDual, method::BM.AbstractMethod)
    LD.block_model.dual_solution = copy(BM.get_solution(method))
    bundle = BM.get_model(method)
    model = BM.get_model(bundle)
    P = model[:P]
    for id in axes(P)[1]
        LD.block_model.P_solution[id] = JuMP.value(P[id])
    end
end

function add_constraints!(LD::DR_LagrangeDual, method::BundleMaster)
    node_to_couple = sort_couple_by_label(LD.tree, LD.block_model.variables_by_couple)

    model = BM.get_jump_model(method.inner)
    @variable(model, P[2:length(LD.tree.nodes)] >= 0)

    for (id, node) in LD.tree.nodes
        add_non_anticipativity!(LD, model, node, node_to_couple[id])
        add_ambiguity!(LD.tree, model, node, node.set)
    end
    for (id, node) in LD.tree.nodes
        add_ambiguity_link!(model, node, node.set)
    end


    #JuMP.print(model)
end

function sort_couple_by_label(tree::Tree, variables_by_couple::Dict{Any,Vector{CouplingVariableKey}})::Dict{Int,Vector{Any}}
    node_to_couple = Dict{Int, Vector{Any}}()
    for (id, nodes) in tree.nodes
        node_to_couple[id] = Vector{Any}()
    end
    for (couple_id, keys) in variables_by_couple
        loc = findfirst(x -> x=='_', couple_id)
        node_id = parse(Int, couple_id[2:loc-1])
        push!(node_to_couple[node_id], couple_id)
    end
    return node_to_couple
end

function add_non_anticipativity!(LD::DR_LagrangeDual, m::JuMP.Model, node::DR_TreeNode, couple_ids::Vector{Any})
    λ = m[:x]
    P = m[:P]

    if check_root(node)
        for couple_id in couple_ids
            vars = LD.block_model.variables_by_couple[couple_id]
            @constraint(m, sum(λ[index_of_λ(LD, v)] for v in vars) == get_cost(node, couple_id), base_name = couple_id)
        end
    else
        for couple_id in couple_ids
            vars = LD.block_model.variables_by_couple[couple_id]
            @constraint(m, sum(λ[index_of_λ(LD, v)] for v in vars) == get_cost(node, couple_id) * P[get_id(node)], base_name = couple_id)
        end
    end
end


function add_ambiguity!(tree::Tree{DR_TreeNode}, m::JuMP.Model, node::DR_TreeNode, set::AbstractAmbiguitySet) end
function add_ambiguity!(tree::Tree{DR_TreeNode}, m::JuMP.Model, node::DR_TreeNode, set::Nothing) end

function add_ambiguity!(tree::Tree{DR_TreeNode}, m::JuMP.Model, node::DR_TreeNode, set::WassersteinSet)
    P = m[:P]
    if check_root(node)
        @variable(m, w[id=get_children(node),s=1:set.N] >= 0, base_name = "n1_w")
        @constraint(m, sum( sum( w[id, s] * set.norm_func(tree.nodes[id].ξ, set.samples[s].ξ) for s in 1:set.N) for id in get_children(node)) <= set.ϵ)
        @constraint(m, [s=1:set.N], sum( w[id, s] for id in get_children(node)) == set.samples[s].p )
        @constraint(m, sum( P[child] for child in get_children(node)) == 1)
        JuMP.unregister(m, :w)
    elseif !check_leaf(node)
        node_id = get_id(node)
        @variable(m, w[id=get_children(node),s=1:set.N] >= 0, base_name = "n$(node_id)_w")
        @constraint(m, sum( sum( w[id, s] * set.norm_func(tree.nodes[id].ξ, set.samples[s].ξ) for s in 1:set.N) for id in get_children(node)) <= set.ϵ * P[node_id])
        @constraint(m, [s=1:set.N], sum( w[id, s] for id in get_children(node)) == set.samples[s].p * P[node_id])
        @constraint(m, sum( P[child] for child in get_children(node)) == P[node_id])
        JuMP.unregister(m, :w)
    end
end

function add_ambiguity_link!(m::JuMP.Model, node::DR_TreeNode, set::Nothing) end
function add_ambiguity_link!(m::JuMP.Model, node::DR_TreeNode, set::WassersteinSet)
    if !check_leaf(node)
        id = get_id(node)
        P = m[:P]
        w = Dict{Tuple{Int,Int},JuMP.VariableRef}( (child, s) => JuMP.variable_by_name(m, "n$(id)_w[$(child),$(s)]") for child=get_children(node), s in 1:set.N)
        @constraint(m, [child=get_children(node)], sum( w[child, s] for s in 1:set.N) == P[child])
    end
end

function initialize_bundle(tree::Tree{DR_TreeNode}, LD::DR_LagrangeDual, Optimizer)::Array{Float64,1}
    n = parallel.sum(num_coupling_variables(LD.block_model))
    bundle_init = Array{Float64,1}(undef, n)
    variable_keys = [v.key for v in LD.block_model.coupling_variables]
    all_variable_keys = parallel.allcollect(variable_keys)
    if parallel.is_root()
        P = get_feasible_P(tree, LD, Optimizer)
        #println(P)
        for key in all_variable_keys
            i = LD.var_to_index[(key.block_id,key.coupling_id)]
            N = length(LD.block_model.variables_by_couple[key.coupling_id])
            couple_id = key.coupling_id

            loc = findfirst(x -> x=='_', couple_id)
            node_id = parse(Int, couple_id[2:loc-1])
            
            if check_root(tree.nodes[node_id])
                bundle_init[i] =  get_cost(tree.nodes[node_id], couple_id) / N
            else
                bundle_init[i] =  get_cost(tree.nodes[node_id], couple_id) / N * P[node_id] #P[corresponding leaf node] ?
            end
            """
            for (id, node) in tree.nodes
                if check_leaf(node)
                    bundle_init[i] =  get_cost(tree.nodes[node_id], couple_id)* P[get_id(node)]
                end
            end
            """
        end
        parallel.bcast(bundle_init)
    else
        bundle_init = parallel.bcast(nothing)
    end
    return bundle_init
end

function get_feasible_P(tree::Tree{DR_TreeNode}, LD::DR_LagrangeDual, Optimizer)::Dict{Int,Float64}
    model = JuMP.Model(Optimizer)

    @variable(model, P[2:length(tree.nodes)] >= 0)
    for (id, node) in tree.nodes
        add_ambiguity!(tree, model, node, node.set)
    end
    for (id, node) in tree.nodes
        add_ambiguity_link!(model, node, node.set)
    end

    @objective(model, Min, 0)

    LD.P_model = model
    JuMP.optimize!(model)
    #JuMP.print(model)

    Pref = Dict()
    P = model[:P]

    for id in 2:length(tree.nodes)
        Pref[id] = JuMP.value(P[id])
    end
    return Pref
end