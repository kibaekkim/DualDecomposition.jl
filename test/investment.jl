@testset "investment" begin
    include("../examples/investment.jl")

    # generate tree data structure
    tree = create_tree(K,L)
    
    @testset "ProximalMethod" begin
        # Create DualDecomposition instance.
        algo = DD.LagrangeDual()

        # Lagrange master method
        LM = DD.BundleMaster(BM.ProximalMethod, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

        # compute dual decomposition method
        dual_decomp!(L, tree, algo, LM)

        @show DD.dual_objective_value(algo)
        @show DD.dual_solution(algo)
        @test isapprox(DD.dual_objective_value(algo), -171.75, rtol=1e-3)
    end

    @testset "TrustRegionMethod" begin
        # Create DualDecomposition instance.
        algo = DD.LagrangeDual()
        
        # Lagrange master method
        LM = DD.BundleMaster(BM.TrustRegionMethod, GLPK.Optimizer)

        # compute dual decomposition method
        dual_decomp!(L, tree, algo, LM)

        @show DD.dual_objective_value(algo)
        @show DD.dual_solution(algo)
        @test isapprox(DD.dual_objective_value(algo), -171.75, rtol=1e-3)
    end
end