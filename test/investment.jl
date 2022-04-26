@testset "investment" begin

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
    K = 3
    L = 2
    a = 0.01
    b_init = 100  # initial capital
    b_in = 30   # income
    # iteratively add nodes
    # root node
    function create_nodes()::DD.Tree
        ξ = Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}(:π => ones(L))
        tree = DD.Tree(ξ)

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
            id = DD.add_child!(tree, pt, ξ, prob)

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
    #node_cluster = DD.decomposition_not(tree)
    node_cluster = DD.decomposition_scenario(tree)
    
    @testset "ProximalMethod" begin
        # Create DualDecomposition instance.
        algo = DD.LagrangeDual()

        coupling_variables = Vector{DD.CouplingVariableRef}()
        models = Dict{Int,JuMP.Model}()
        for (block_id, nodes) in enumerate(node_cluster)
            subtree = DD.create_subtree!(block_id, coupling_variables, nodes)
            set_optimizer(subtree.model, GLPK.Optimizer)
            DD.add_block_model!(algo, block_id, subtree.model)
            models[block_id] = subtree.model
        end

        # Set nonanticipativity variables as an array of symbols.
        DD.set_coupling_variables!(algo, coupling_variables)

        # Lagrange master method
        LM = DD.BundleMaster(BM.ProximalMethod, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

        # Solve the problem with the solver; this solver is for the underlying bundle method.
        DD.run!(algo, LM)

        @show DD.dual_objective_value(algo)
        @show DD.dual_solution(algo)
        @test isapprox(DD.dual_objective_value(algo), -171.75, rtol=1e-3)
    end

    @testset "TrustRegionMethod" begin
        # Create DualDecomposition instance.
        algo = DD.LagrangeDual()

        coupling_variables = Vector{DD.CouplingVariableRef}()
        models = Dict{Int,JuMP.Model}()
        for (block_id, nodes) in enumerate(node_cluster)
            subtree = DD.create_subtree!(block_id, coupling_variables, nodes)
            set_optimizer(subtree.model, GLPK.Optimizer)
            DD.add_block_model!(algo, block_id, subtree.model)
            models[block_id] = subtree.model
        end

        # Set nonanticipativity variables as an array of symbols.
        DD.set_coupling_variables!(algo, coupling_variables)

        # Lagrange master method
        LM = DD.BundleMaster(BM.TrustRegionMethod, GLPK.Optimizer)

        # Solve the problem with the solver; this solver is for the underlying bundle method.
        DD.run!(algo, LM)

        @show DD.dual_objective_value(algo)
        @show DD.dual_solution(algo)
        @test isapprox(DD.dual_objective_value(algo), -171.75, rtol=1e-3)
    end

    @testset "TemporalDecomposition" begin
        node_cluster = DD.decomposition_temporal(tree)
        # Create DualDecomposition instance.
        algo = DD.LagrangeDual()

        coupling_variables = Vector{DD.CouplingVariableRef}()
        models = Dict{Int,JuMP.Model}()
        for (block_id, nodes) in enumerate(node_cluster)
            subtree = DD.create_subtree!(block_id, coupling_variables, nodes)
            set_optimizer(subtree.model, GLPK.Optimizer)
            DD.add_block_model!(algo, block_id, subtree.model)
            models[block_id] = subtree.model
        end

        # Set nonanticipativity variables as an array of symbols.
        DD.set_coupling_variables!(algo, coupling_variables)

        # Lagrange master method
        LM = DD.BundleMaster(BM.ProximalMethod, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

        # Solve the problem with the solver; this solver is for the underlying bundle method.
        DD.run!(algo, LM)

        @show DD.dual_objective_value(algo)
        @show DD.dual_solution(algo)
        @test isapprox(DD.dual_objective_value(algo), -171.75, rtol=1e-3)
    end
end