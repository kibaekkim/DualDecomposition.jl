@testset "Heuristics" begin
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

            @testset "All block heuristic with trust region method" begin
                # Create DualDecomposition instance.
                algo = DD.LagrangeDual()
    
                # Add Lagrange dual problem for each scenario s.
                for s in 1:NS
                    DD.add_block_model!(algo, s, models[s])
                end
    
                # Set nonanticipativity variables as an array of symbols.
                DD.set_coupling_variables!(algo, coupling_variables)
    
                # Lagrange master method
                LM = DD.BundleMaster(BM.TrustRegionMethod, GLPK.Optimizer)

                # add heuristic
                DD.add!(DD.AllBlockHeuristic, algo)
                
                # Solve the problem with the solver; this solver is for the underlying bundle method.
                DD.run!(algo, LM)
    
                @show DD.primal_objective_value(algo)
                @show DD.primal_solution(algo)
                @test isapprox(DD.primal_objective_value(algo), -107701.0, rtol=1e-3)
            end

            @testset "Rounding heuristic with trust region method" begin
                # Create DualDecomposition instance.
                algo = DD.LagrangeDual()
    
                # Add Lagrange dual problem for each scenario s.
                for s in 1:NS
                    DD.add_block_model!(algo, s, models[s])
                end
    
                # Set nonanticipativity variables as an array of symbols.
                DD.set_coupling_variables!(algo, coupling_variables)
    
                # Lagrange master method
                LM = DD.BundleMaster(BM.TrustRegionMethod, GLPK.Optimizer)

                # add heuristic
                DD.add!(DD.RoundingHeuristic, algo)
                
                # Solve the problem with the solver; this solver is for the underlying bundle method.
                DD.run!(algo, LM)
    
                @show DD.primal_objective_value(algo)
                @show DD.primal_solution(algo)
                @test isapprox(DD.primal_objective_value(algo), -107342., rtol=1e-3)
            end
        end
    end
end