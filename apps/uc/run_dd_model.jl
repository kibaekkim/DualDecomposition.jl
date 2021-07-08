if length(ARGS) < 1
    @error "Need an argument for the number of scenarios"
else
    include("suc_dd_model.jl")

    num_scenarios = parse(Int,ARGS[1])

    # Initialize MPI
    Para.init()
    
    # Partition scenarios into processes
    Para.partition(num_scenarios)
    partitions = Para.getpartition()
    @show Para.myid(), partitions
    
    algo = create_decomposition_model(num_scenarios, Para.getpartition());
    algo.tol = 0.00001
    algo.maxiter = 300
    
    # Solve the problem with the solver; this solver is for the underlying bundle method.

    DD.run!(algo, 
        optimizer_with_attributes(
            CPLEX.Optimizer, 
	    "CPX_PARAM_LPMETHOD" => 4, # use barrier
	    "CPX_PARAM_BAREPCOMP" => 1e-6, # convergence tolerance
	    "CPX_PARAM_SOLUTIONTYPE" => 2, # no barrier crossover
	    "CPX_PARAM_DEPIND" => 3, # dependency checking
            "CPX_PARAM_SCRIND" => 0, 
            "CPX_PARAM_THREADS" => 1 
        )
    )
  
    # Finalize MPI
    Para.finalize()
end
