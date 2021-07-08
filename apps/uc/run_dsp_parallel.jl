if length(ARGS) < 1
    @error "Need an argument for the number of scenarios"
else
    using MPI
    
    include("suc_dsp_model.jl")
    
    # Initialize MPI
    MPI.Init()
    
    # Hand over the MPI communicator to DSP
    DSPopt.parallelize(MPI.COMM_WORLD)
    
    # Generate StructJuMP model
    # m = create_dsp_model(parse(Int,ARGS[1]))
    m = create_dsp_model_fewer_couples(parse(Int,ARGS[1]))
    
    # Solve the model
    status = optimize!(m,
        is_stochastic = true,
        solve_type = DSPopt.Legacy,
        param = "params.txt"
    )
    
    # Print out solution
    if DSPopt.myrank() == 0 && status == MOI.OPTIMAL
        @show objective_value(m)
        @show dual_objective_value(m)
    end
    
    # Finalize MPI
    MPI.Finalize()
end
