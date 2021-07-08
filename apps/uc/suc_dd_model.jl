using DualDecomposition
using CPLEX
const DD = DualDecomposition
const Para = DD.parallel

include("suc_model.jl")

function create_decomposition_model(
    num_scenarios::Int = 1, 
    partitions = 1:num_scenarios,
    data_file::String = "./data/model_data.json")

    # Read model data from file
    data = load_suc_data(data_file)

    slow_start_gens = keys(data["slow_generators"])
    fast_start_gens = keys(data["fast_generators"])
    renewable_gens = keys(data["renewable_generators"])
    time_periods = 1:data["time_periods"]
    gen_startup_categories = Dict(g => 1:length(gen["startup"]) for (g,gen) in data["thermal_generators"])

    @info "Found $num_scenarios scenarios"
    @info "Found $(length(slow_start_gens)) slow-start generators"
    @info "Found $(length(fast_start_gens)) fast-start generators"
    @info "Found $(length(renewable_gens)) renewable generators"

    """
        create_scenario_subproblem

    Create scenario subproblem
    """
    function create_scenario_subproblem(s::Int64)
        m = create_suc_model(data, s:s, 1 / num_scenarios)
        set_optimizer(m, CPLEX.Optimizer)
        set_optimizer_attribute(m, "CPX_PARAM_THREADS", 1)
        set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
        set_optimizer_attribute(m, "CPX_PARAM_TILIM", 60.0)
        set_optimizer_attribute(m, "CPX_PARAM_EPGAP", 0.01)
        return m
    end

    # Create DualDecomposition instance.
    algo = DD.LagrangeDual(BM.TrustRegionMethod)

    # Add Lagrange dual problem for each scenario s.
    models = Dict{Int,JuMP.Model}(s => create_scenario_subproblem(s) for s in partitions)
    for s in partitions
        DD.add_block_model!(algo, s, models[s])
    end

    """
    It is sufficient to have `ug` as the only coupling variables,
        because `vg`, `wg`, and `delta_sg` will be implicitly determined for given `ug`.
    """
    coupling_variables = Vector{DD.CouplingVariableRef}()
    for s in partitions
        model = models[s]
        ug = model[:ug]
        for g in slow_start_gens, t in time_periods
            push!(coupling_variables, DD.CouplingVariableRef(s, (g,t), ug[g,t]))
        end
    end

    # Set nonanticipativity variables as an array of symbols.
    DD.set_coupling_variables!(algo, coupling_variables)

    return algo
end

#=
Example to use:

# Initialize MPI
Para.init()

# Partition scenarios into processes
Para.partition(num_scenarios)

algo = create_decomposition_model(2, Para.getpartition());

# Solve the problem with the solver; this solver is for the underlying bundle method.
DD.run!(algo, optimizer_with_attributes(CPLEX.Optimizer, "CPX_PARAM_SCRIND" => 0))

# Finalize MPI
Para.finalize()
=#
