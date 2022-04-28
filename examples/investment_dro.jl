using JuMP, GLPK, Ipopt
using DualDecomposition
using Random

const DD = DualDecomposition
const parallel = DD.parallel

const rng = Random.MersenneTwister(1234)

"""
a: interest rate
π: unit stock price
ρ: unit dividend price


K: number of stages
L: number of stock types
2^L scenarios in each stage
2^L^(K-1)=16 scenarios in total
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
const K = 3
const L = 2
const a = 0.01
const b_init = 100  # initial capital
const b_in = 30   # income

"""
In each node, we have Np=10 samples from a log-normal distribution
"""
const Np = 10 # number of samples

function generate_sample(π::Array{Float64})::Array{DD.Sample}
    # generates random samples following a lognormal distribution
    ret = Array{DD.Sample}(undef, Np)
    for ii in 1:Np
        π_samp = Array{Float64}(undef, L)
        for l in 1:L
            sig = sqrt( log( 0.5+sqrt( ( 0.03*l )^2+0.25 ) ) )
            rnd = sig * randn(rng) .+ log(π[l])
            π_samp[l] = exp(rnd)
        end
        ret[ii] = DD.Sample(Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}(:π => π_samp), 1/Np)
    end
    return ret
end

# iteratively add nodes
# root node
function create_nodes()::DD.Tree
    ξ = Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}(:π => ones(L))
    ξ_samp = generate_sample(ξ[:π])
    set = DD.WassersteinSet(ξ_samp, 1.0, DD.norm_L1)
    tree = DD.Tree(ξ, set)

    #subproblem formulation
    function subproblem_builder(mdl::JuMP.Model, node::DD.SubTreeNode)
        @variable(mdl, x[l=1:L], DD.ControlInfo, subnode = node, ref_symbol = :x)

        @variable(mdl, y[l=1:L] >= 0, DD.OutStateInfo, subnode = node, ref_symbol = :y)

        @variable(mdl, B >= 0, DD.OutStateInfo, subnode = node, ref_symbol = :B)

        π = DD.get_scenario(node)[:π]
        @constraints(mdl, 
            begin
                B + sum( π[l] * x[l] for l in 1:L) == b_init
                [l=1:L], y[l] - x[l] == 0 
            end
        )
        DD.set_stage_objective(node, 0.0)
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
        if DD.get_stage(tree, pt) != K-1
            ξ_samp = generate_sample(ξ[:π])
            set = DD.WassersteinSet(ξ_samp, 1.0, DD.norm_L1)
            id = DD.add_child!(tree, pt, ξ, set)
        else 
            id = DD.add_child!(tree, pt, ξ, nothing)
        end
        

        #subproblem formulation
        function subproblem_builder(mdl::JuMP.Model, node::DD.SubTreeNode)
            @variable(mdl, x[l=1:L], DD.ControlInfo, subnode = node, ref_symbol = :x)

            @variable(mdl, y[l=1:L] >= 0, DD.OutStateInfo, subnode = node, ref_symbol = :y)

            @variable(mdl, B >= 0, DD.OutStateInfo, subnode = node, ref_symbol = :B)

            @variable(mdl, y_[l=1:L] >= 0, DD.InStateInfo, subnode = node, ref_symbol = :y)

            @variable(mdl, B_ >= 0, DD.InStateInfo, subnode = node, ref_symbol = :B)

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

det = DD.create_Wasserstein_deterministic!(tree)
set_optimizer(det.model, GLPK.Optimizer)
JuMP.optimize!(det.model)
println(JuMP.objective_value(det.model))


#node_cluster = DD.decomposition_not(tree)
#node_cluster = DD.decomposition_scenario(tree)
node_cluster = DD.decomposition_temporal(tree) #There is a DUAL_INFEASIBLE issue

# Number of block components
NS = length(node_cluster)

# Initialize MPI
parallel.init()

# Create DualDecomposition instance.
algo = DD.DR_LagrangeDual(tree)

# partition scenarios into processes
parallel.partition(NS)

coupling_variables = Vector{DD.CouplingVariableRef}()
models = Dict{Int,JuMP.Model}()

for block_id in parallel.getpartition()
    nodes = node_cluster[block_id]
    subtree = DD.create_subtree!(block_id, coupling_variables, nodes)
    set_optimizer(subtree.model, GLPK.Optimizer)
    DD.add_block_model!(algo, block_id, subtree.model)
    models[block_id] = subtree.model
end

# Set nonanticipativity variables as an array of symbols.
DD.set_coupling_variables!(algo, coupling_variables)

bundle_init = DD.initialize_bundle(tree, algo, GLPK.Optimizer)
#println(bundle_init)
# for (id, node) in tree.nodes
#    println(node.cost)
# end

# Lagrange master method
#LM = DD.BundleMaster(BM.ProximalMethod, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
LM = DD.BundleMaster(BM.TrustRegionMethod, GLPK.Optimizer)

# Solve the problem with the solver; this solver is for the underlying bundle method.
DD.run!(algo, LM, bundle_init)


# Write timing outputs to files
DD.write_all(algo)

# Finalize MPI
parallel.finalize()
