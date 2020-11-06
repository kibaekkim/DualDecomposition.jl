using CPLEX

include("suc_model.jl")

function create_deterministic_model(num_scenarios_::Int = 1, data_file::String = "./data/model_data.json")

    # Read model data from file
    data = load_suc_data(data_file)

    # How many scenarios?
    (g,gen) = iterate(data["renewable_generators"])[1]
    num_scenarios = ifelse(num_scenarios_ < 0, length(gen), num_scenarios_)
    scenarios = 1:num_scenarios

    @info "Found $(length(scenarios)) scenarios"
    @info "Found $(length(slow_start_gens)) slow-start generators"
    @info "Found $(length(fast_start_gens)) fast-start generators"
    @info "Found $(length(renewable_gens)) renewable generators"

    return create_suc_model(data, scenarios)
end

# set_optimizer(m, CPLEX.Optimizer)
# set_optimizer_attribute(m, "CPX_PARAM_TILIM", 60.0)
# optimize!(m)