if length(ARGS) < 1
    @error "Need an argument for the number of scenarios"
else
    include("suc_dd_model.jl")

    num_scenarios = parse(Int,ARGS[1])

    # Initialize MPI
    Para.init()
    
    # Partition scenarios into processes
    Para.partition(num_scenarios)
    
    algo = create_decomposition_model(num_scenarios, Para.getpartition());
    algo.tol = 0.00001
    algo.maxiter = 1000
    
    # Solve the problem with the solver; this solver is for the underlying bundle method.
    DD.run!(algo, 
        optimizer_with_attributes(
            CPLEX.Optimizer, 
            "CPX_PARAM_SCRIND" => 0, 
            "CPX_PARAM_THREADS" => 1, 
            "CPX_PARAM_TILIM" => 60.0,
            "CPX_PARAM_EPGAP" => 0.0001
        )
    )
    
    # Finalize MPI
    Para.finalize()
end