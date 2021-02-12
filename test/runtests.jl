using Test
using DualDecomposition
using JuMP, Ipopt, GLPK
using MPI

const DD = DualDecomposition

@testset "farmer" begin
    NS = 3  # number of scenarios
    probability = ones(3) / 3
    CROPS = 1:3 # set of crops (wheat, corn and sugar beets, resp.)
    PURCH = 1:2 # set of crops to purchase (wheat and corn, resp.)
    SELL = 1:4  # set of crops to sell (wheat, corn, sugar beets under 6K and those over 6K)
    Cost = [150 230 260]    # cost of planting crops
    Budget = 500            # budget capacity
    Purchase = [238 210]    # purchase price
    Sell = [170 150 36 10]  # selling price
    Yield = [3.0 3.6 24.0; 2.5 3.0 20.0; 2.0 2.4 16.0]
    Minreq = [200 240 0]    # minimum crop requirement

    @testset "MILP" begin
        # This creates a Lagrange dual problem for each scenario s.
        function create_scenario_model(s::Int64)
            m = Model(GLPK.Optimizer)
            @variable(m, 0 <= x[i=CROPS] <= 500, Int)
            @variable(m, y[j=PURCH] >= 0)
            @variable(m, w[k=SELL] >= 0)
        
            @objective(m, Min,
                probability[s] * sum(Cost[i] * x[i] for i=CROPS)
                + probability[s] * sum(Purchase[j] * y[j] for j=PURCH) 
                - probability[s] * sum(Sell[k] * w[k] for k=SELL))
        
            @constraint(m, sum(x[i] for i=CROPS) <= Budget)
            @constraint(m, [j=PURCH], Yield[s,j] * x[j] + y[j] - w[j] >= Minreq[j])
            @constraint(m, Yield[s,3] * x[3] - w[3] - w[4] >= Minreq[3])
            @constraint(m, w[3] <= 6000)
            return m
        end

        models = Dict{Int,JuMP.Model}(s => create_scenario_model(s) for s in 1:NS)
        coupling_variables = Vector{DD.CouplingVariableRef}()
        for s in 1:NS
            model = models[s]
            xref = model[:x]
            for i in CROPS
                push!(coupling_variables, DD.CouplingVariableRef(s, i, xref[i]))
            end
        end

        @testset "Proximal Method" begin
            # Create DualDecomposition instance.
            algo = DD.LagrangeDual(BM.ProximalMethod)

            # Add Lagrange dual problem for each scenario s.
            for s in 1:NS
                DD.add_block_model!(algo, s, models[s])
            end

            # Set nonanticipativity variables as an array of symbols.
            DD.set_coupling_variables!(algo, coupling_variables)
            
            # Solve the problem with the solver; this solver is for the underlying bundle method.
            DD.run!(algo, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

            @show DD.dual_objective_value(algo)
            @show DD.dual_solution(algo)
            @test isapprox(DD.dual_objective_value(algo), -108390, rtol=1e-3)
        end

        @testset "Trust Region Method" begin
            # Create DualDecomposition instance.
            algo = DD.LagrangeDual(BM.TrustRegionMethod)

            # Add Lagrange dual problem for each scenario s.
            for s in 1:NS
                DD.add_block_model!(algo, s, models[s])
            end

            # Set nonanticipativity variables as an array of symbols.
            DD.set_coupling_variables!(algo, coupling_variables)
            
            # Solve the problem with the solver; this solver is for the underlying bundle method.
            DD.run!(algo, GLPK.Optimizer)

            @show DD.dual_objective_value(algo)
            @show DD.dual_solution(algo)
            @test isapprox(DD.dual_objective_value(algo), -108390, rtol=1e-3)
        end
    end

    @testset "QP" begin
        function deterministic_model_bound()
            m = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
            @variable(m, 0 <= x[i=CROPS] <= 500)
            @variable(m, y[s=1:NS,j=PURCH] >= 0)
            @variable(m, w[s=1:NS,k=SELL] >= 0)
        
            @objective(m, Min,
                sum(Cost[i] * x[i]^2 for i=CROPS)
                + sum(probability[s] * Purchase[j] * y[s,j] for s=1:NS, j=PURCH) 
                - sum(probability[s] * Sell[k] * w[s,k] for s=1:NS, k=SELL))
        
            @constraint(m, sum(x[i] for i=CROPS) <= Budget)
            @constraint(m, [s=1:NS,j=PURCH], Yield[s,j] * x[j] + y[s,j] - w[s,j] >= Minreq[j])
            @constraint(m, [s=1:NS], Yield[s,3] * x[3] - w[s,3] - w[s,4] >= Minreq[3])
            @constraint(m, [s=1:NS], w[s,3] <= 6000)
            optimize!(m)
            objval = Inf
            if termination_status(m) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
                objval = objective_value(m)
            end
            return objval
        end

        # This creates a Lagrange dual problem for each scenario s.
        function create_scenario_model(s::Int64)
            m = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
            @variable(m, 0 <= x[i=CROPS] <= 500)
            @variable(m, y[j=PURCH] >= 0)
            @variable(m, w[k=SELL] >= 0)
        
            @objective(m, Min,
                probability[s] * sum(Cost[i] * x[i]^2 for i=CROPS)
                + probability[s] * sum(Purchase[j] * y[j] for j=PURCH) 
                - probability[s] * sum(Sell[k] * w[k] for k=SELL))
        
            @constraint(m, sum(x[i] for i=CROPS) <= Budget)
            @constraint(m, [j=PURCH], Yield[s,j] * x[j] + y[j] - w[j] >= Minreq[j])
            @constraint(m, Yield[s,3] * x[3] - w[3] - w[4] >= Minreq[3])
            @constraint(m, w[3] <= 6000)
            return m
        end

        models = Dict{Int,JuMP.Model}(s => create_scenario_model(s) for s in 1:NS)
        coupling_variables = Vector{DD.CouplingVariableRef}()
        for s in 1:NS
            model = models[s]
            xref = model[:x]
            for i in CROPS
                push!(coupling_variables, DD.CouplingVariableRef(s, i, xref[i]))
            end
        end

        objval = deterministic_model_bound()
        @show objval
        @test objval < Inf

        @testset "Proximal Method" begin
            # Create DualDecomposition instance.
            algo = DD.LagrangeDual(BM.ProximalMethod)

            # Add Lagrange dual problem for each scenario s.
            for s in 1:NS
                DD.add_block_model!(algo, s, models[s])
            end

            # Set nonanticipativity variables as an array of symbols.
            DD.set_coupling_variables!(algo, coupling_variables)
            
            # Solve the problem with the solver; this solver is for the underlying bundle method.
            DD.run!(algo, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

            @show DD.dual_objective_value(algo)
            @show DD.dual_solution(algo)
            @test isapprox(DD.dual_objective_value(algo), objval, rtol=1e-3)
        end

        @testset "Trust Region Method" begin
            # Create DualDecomposition instance.
            algo = DD.LagrangeDual(BM.TrustRegionMethod)

            # Add Lagrange dual problem for each scenario s.
            for s in 1:NS
                DD.add_block_model!(algo, s, models[s])
            end

            # Set nonanticipativity variables as an array of symbols.
            DD.set_coupling_variables!(algo, coupling_variables)
            
            # Solve the problem with the solver; this solver is for the underlying bundle method.
            DD.run!(algo, GLPK.Optimizer)

            @show DD.dual_objective_value(algo)
            @show DD.dual_solution(algo)
            @test isapprox(DD.dual_objective_value(algo), objval, rtol=1e-3)
        end
    end
end

@testset "MPI tests" begin
    testdir = @__DIR__
    run(`$(Base.julia_cmd()) $(joinpath(testdir, "parallel.jl"))`)
    mpiexec(cmd ->run(`$cmd -np 2 $(Base.julia_cmd()) $(joinpath(testdir, "parallel.jl"))`))
    mpiexec(cmd ->run(`$cmd -np 3 $(Base.julia_cmd()) $(joinpath(testdir, "parallel.jl"))`))
    run(`$(Base.julia_cmd()) $(joinpath(testdir, "../examples/farmer_mpi.jl"))`)
    mpiexec(cmd ->run(`$cmd -np 2 $(Base.julia_cmd()) $(joinpath(testdir, "../examples/farmer_mpi.jl"))`))
    mpiexec(cmd ->run(`$cmd -np 3 $(Base.julia_cmd()) $(joinpath(testdir, "../examples/farmer_mpi.jl"))`))
end

@testset "investment" begin
    include("../examples/investment.jl")

    # generate tree data structure
    tree = create_tree(K,L)
    
    @testset "ProximalMethod" begin
        # Create DualDecomposition instance.
        algo = DD.LagrangeDual(BM.ProximalMethod)

        # compute dual decomposition method
        dual_decomp!(L, tree, algo, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

        @show DD.dual_objective_value(algo)
        @show DD.dual_solution(algo)
        @test isapprox(DD.dual_objective_value(algo), -171.75, rtol=1e-3)
    end

    @testset "TrustRegionMethod" begin
        # Create DualDecomposition instance.
        algo = DD.LagrangeDual(BM.TrustRegionMethod)

        # compute dual decomposition method
        dual_decomp!(L, tree, algo, GLPK.Optimizer)

        @show DD.dual_objective_value(algo)
        @show DD.dual_solution(algo)
        @test isapprox(DD.dual_objective_value(algo), -171.75, rtol=1e-3)
    end
end

# include("../examples/farmer.jl")
# include("../examples/dcap.jl")
# include("../examples/qdcap.jl") # Need CPLEX
# include("../examples/sslp.jl")