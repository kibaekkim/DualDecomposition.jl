using JSON
using CPLEX
using JuMP

function create_deterministic_model(num_scenarios_ = 1, data_file = "./data/model_data.json")

    # Read model data from file
    data = JSON.parsefile(data_file)

    # How many scenarios?
    (g,gen) = iterate(data["renewable_generators"])[1]
    num_scenarios = ifelse(num_scenarios_ < 0, length(gen), num_scenarios_)
    scenarios = 1:num_scenarios

    # Find slow- and fast-start generators
    slow_start_gens = String[]
    fast_start_gens = String[]
    for (g,gen) in data["thermal_generators"] 
        if gen["time_down_minimum"] <= 1 && gen["time_up_minimum"] <= 1
            push!(fast_start_gens, g)
        else
            push!(slow_start_gens, g)
        end
    end
    data["slow_generators"] = Dict(
        g => gen for (g, gen) in data["thermal_generators"] if in(g, slow_start_gens)
    )
    data["fast_generators"] = Dict(
        g => gen for (g, gen) in data["thermal_generators"] if in(g, fast_start_gens)
    )

    thermal_gens = keys(data["thermal_generators"])
    renewable_gens = keys(data["renewable_generators"])
    time_periods = 1:data["time_periods"]

    gen_startup_categories = Dict(g => 1:length(gen["startup"]) for (g,gen) in data["thermal_generators"])
    gen_pwl_points = Dict(g => 1:length(gen["piecewise_production"]) for (g,gen) in data["thermal_generators"])

    @info "Found $num_scenarios scenarios"
    @info "Found $(length(slow_start_gens)) slow-start generators"
    @info "Found $(length(fast_start_gens)) fast-start generators"
    @info "Found $(length(renewable_gens)) renewable generators"

    m = Model(CPLEX.Optimizer)
    set_optimizer_attribute(m, "CPX_PARAM_TILIM", 60.0)

    # First-stage variables

    @variable(m, ug[slow_start_gens,time_periods], binary=true) # Commitment status
    @variable(m, vg[slow_start_gens,time_periods], binary=true) # Startup status
    @variable(m, wg[slow_start_gens,time_periods], binary=true) # Shutdown status
    @variable(m, delta_sg[g in slow_start_gens,gen_startup_categories[g],time_periods], binary=true) # Startup in category ?

    # Second-stage variables
    @variable(m, ufg[fast_start_gens,time_periods,scenarios], binary=true) # Commitment status
    @variable(m, vfg[fast_start_gens,time_periods,scenarios], binary=true) # Startup status
    @variable(m, wfg[fast_start_gens,time_periods,scenarios], binary=true) # Shutdown status
    @variable(m, delta_sfg[g in fast_start_gens,gen_startup_categories[g],time_periods,scenarios], binary=true) # Startup in category ?

    @variable(m, cg[thermal_gens,time_periods,scenarios]) # Production cost
    @variable(m, pg[thermal_gens,time_periods,scenarios] >= 0) # Thermal generation
    @variable(m, pw[renewable_gens,time_periods,scenarios] >= 0) # Renewable generation
    @variable(m, rg[thermal_gens,time_periods,scenarios] >= 0) # Spinning reserve
    @variable(m, 0 <= lambda_lg[g in thermal_gens,gen_pwl_points[g],time_periods,scenarios] <= 1) # Fraction of power in piecewise generation point


    @objective(m, Min,
        # first-stage objective function terms
        sum(
            gen["piecewise_production"][1]["cost"]*ug[g,t] +
            sum(
                gen_startup["cost"]*delta_sg[g,i,t]
            for (i, gen_startup) in enumerate(gen["startup"]))
        for (g, gen) in data["slow_generators"], t in time_periods)
        # second-stage objective function terms
        + 1 / num_scenarios * (
            sum(
                sum(
                    gen["piecewise_production"][1]["cost"]*ufg[g,t,s] +
                    sum(
                        gen_startup["cost"]*delta_sfg[g,i,t,s]
                    for (i, gen_startup) in enumerate(gen["startup"]))
                for (g, gen) in data["fast_generators"], t in time_periods) +
                sum(cg[g,t,s] 
                for t in time_periods, (g, gen) in data["thermal_generators"])
            for s in scenarios)
        )
    );

    for (g, gen) in data["slow_generators"]

        if gen["unit_on_t0"] == 1
            @constraint(m, sum( (ug[g,t]-1) for t in 1:min(data["time_periods"], gen["time_up_minimum"] - gen["time_up_t0"]) ) == 0) # (4)
        else
            @constraint(m, sum( ug[g,t] for t in 1:min(data["time_periods"], gen["time_down_minimum"] - gen["time_down_t0"]) ) == 0) # (5)
        end

        @constraint(m, ug[g,1] - gen["unit_on_t0"] == vg[g,1] - wg[g,1]) # (6)

        @constraint(m, 0 ==
            sum(
                sum(
                    delta_sg[g,i,t]
                for t in max(1, gen["startup"][i+1]["lag"] - gen["time_down_t0"] + 1):min(gen["startup"][i+1]["lag"]-1, data["time_periods"]))
            for (i,startup) in enumerate(gen["startup"][1:end-1]))
        ) # (7)

        @constraint(m, [s in scenarios], pg[g,1,s] + rg[g,1,s] - gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["ramp_up_limit"]) # (8)
        @constraint(m, [s in scenarios], gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) - pg[g,1,s] <= gen["ramp_down_limit"]) # (9)
        @constraint(m, [s in scenarios], gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["unit_on_t0"]*(gen["power_output_maximum"] - gen["power_output_minimum"]) - max(0, gen["power_output_maximum"] - gen["ramp_shutdown_limit"])*wg[g,1]) # (10)
    end

    for (g, gen) in data["fast_generators"]

        if gen["unit_on_t0"] == 1
            @constraint(m, [s in scenarios], sum( (ufg[g,t,s]-1) for t in 1:min(data["time_periods"], gen["time_up_minimum"] - gen["time_up_t0"]) ) == 0) # (4)
        else
            @constraint(m, [s in scenarios], sum( ufg[g,t,s] for t in 1:min(data["time_periods"], gen["time_down_minimum"] - gen["time_down_t0"]) ) == 0) # (5)
        end

        @constraint(m, [s in scenarios], ufg[g,1,s] - gen["unit_on_t0"] == vfg[g,1,s] - wfg[g,1,s]) # (6)

        @constraint(m, [s in scenarios], 0 ==
            sum(
                sum(
                    delta_sfg[g,i,t,s]
                for t in max(1, gen["startup"][i+1]["lag"] - gen["time_down_t0"] + 1):min(gen["startup"][i+1]["lag"]-1, data["time_periods"]))
            for (i,startup) in enumerate(gen["startup"][1:end-1]))
        ) # (7)

        @constraint(m, [s in scenarios], pg[g,1,s] + rg[g,1,s] - gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["ramp_up_limit"]) # (8)
        @constraint(m, [s in scenarios], gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) - pg[g,1,s] <= gen["ramp_down_limit"]) # (9)
        @constraint(m, [s in scenarios], gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["unit_on_t0"]*(gen["power_output_maximum"] - gen["power_output_minimum"]) - max(0, gen["power_output_maximum"] - gen["ramp_shutdown_limit"])*wfg[g,1,s]) # (10)
    end

    for t in time_periods
        @constraint(m, [s in scenarios], 
            sum( pg[g,t,s] + gen["power_output_minimum"]*ug[g,t] for (g, gen) in data["slow_generators"] ) +
            sum( pg[g,t,s] + gen["power_output_minimum"]*ufg[g,t,s] for (g, gen) in data["fast_generators"] ) +
            sum( pw[g,t,s] for g in renewable_gens)
            == data["demand"][t]
        ) # (2)

        @constraint(m, [s in scenarios], sum(rg[g,t,s] for g in thermal_gens) >= data["reserves"][t]) # (3)

        for (g, gen) in data["slow_generators"]

            @constraint(m, ug[g,t] >= gen["must_run"]) # (11)

            if t > 1
                @constraint(m, ug[g,t] - ug[g,t-1] == vg[g,t] - wg[g,t]) # (12)
                @constraint(m, [s in scenarios], pg[g,t,s] + rg[g,t,s] - pg[g,t-1,s] <= gen["ramp_up_limit"]) # (19)
                @constraint(m, [s in scenarios], pg[g,t-1,s] - pg[g,t,s] <= gen["ramp_down_limit"]) # (20)
            end


            if t >= gen["time_up_minimum"] || t == data["time_periods"]
                @constraint(m, sum( vg[g,t2] for t2 in (t-min(gen["time_up_minimum"],data["time_periods"])+1):t) <= ug[g,t])  # (13)
            end

            if t >= gen["time_down_minimum"] || t == data["time_periods"]
                @constraint(m, sum( wg[g,t2] for t2 in (t-min(gen["time_down_minimum"],data["time_periods"])+1):t) <= 1 - ug[g,t])  # (14)
            end

            for (si,startup) in enumerate(gen["startup"][1:end-1])
                if t >= gen["startup"][si+1]["lag"]
                    time_range = startup["lag"]:(gen["startup"][si+1]["lag"]-1)
                    @constraint(m, delta_sg[g,si,t] <= sum(wg[g,t-i] for i in time_range)) # (15)
                end
            end

            @constraint(m, vg[g,t] == sum( delta_sg[g,i,t] for i in gen_startup_categories[g])) # (16)

            @constraint(m, [s in scenarios], pg[g,t,s] + rg[g,t,s] <= (gen["power_output_maximum"] - gen["power_output_minimum"])*ug[g,t] - max(0, (gen["power_output_maximum"] - gen["ramp_startup_limit"]))*vg[g,t]) # (17)

            if t < data["time_periods"]
                @constraint(m, [s in scenarios], pg[g,t,s] + rg[g,t,s] <= (gen["power_output_maximum"] - gen["power_output_minimum"])*ug[g,t] - max(0, (gen["power_output_maximum"] - gen["ramp_shutdown_limit"]))*wg[g,t+1]) # (18)
            end

            @constraint(m, [s in scenarios], pg[g,t,s] == sum((gen["piecewise_production"][l]["mw"] - gen["piecewise_production"][1]["mw"])*lambda_lg[g,l,t,s] for l in gen_pwl_points[g])) # (21)
            @constraint(m, [s in scenarios], cg[g,t,s] == sum((gen["piecewise_production"][l]["cost"] - gen["piecewise_production"][1]["cost"])*lambda_lg[g,l,t,s] for l in gen_pwl_points[g])) # (22)
            @constraint(m, [s in scenarios], ug[g,t] == sum(lambda_lg[g,l,t,s] for l in gen_pwl_points[g])) # (23)
        end

        for (g, gen) in data["fast_generators"]

            @constraint(m, [s in scenarios], ufg[g,t,s] >= gen["must_run"]) # (11)

            if t > 1
                @constraint(m, [s in scenarios], ufg[g,t,s] - ufg[g,t-1,s] == vfg[g,t,s] - wfg[g,t,s]) # (12)
                @constraint(m, [s in scenarios], pg[g,t,s] + rg[g,t,s] - pg[g,t-1,s] <= gen["ramp_up_limit"]) # (19)
                @constraint(m, [s in scenarios], pg[g,t-1,s] - pg[g,t,s] <= gen["ramp_down_limit"]) # (20)
            end


            if t >= gen["time_up_minimum"] || t == data["time_periods"]
                @constraint(m, [s in scenarios], sum( vfg[g,t2,s] for t2 in (t-min(gen["time_up_minimum"],data["time_periods"])+1):t) <= ufg[g,t,s])  # (13)
            end

            if t >= gen["time_down_minimum"] || t == data["time_periods"]
                @constraint(m, [s in scenarios], sum( wfg[g,t2,s] for t2 in (t-min(gen["time_down_minimum"],data["time_periods"])+1):t) <= 1 - ufg[g,t,s])  # (14)
            end

            for (si,startup) in enumerate(gen["startup"][1:end-1])
                if t >= gen["startup"][si+1]["lag"]
                    time_range = startup["lag"]:(gen["startup"][si+1]["lag"]-1)
                    @constraint(m, [s in scenarios], delta_sfg[g,si,t,s] <= sum(wfg[g,t-i,s] for i in time_range)) # (15)
                end
            end

            @constraint(m, [s in scenarios], vfg[g,t,s] == sum( delta_sfg[g,i,t,s] for i in gen_startup_categories[g])) # (16)

            @constraint(m, [s in scenarios], pg[g,t,s] + rg[g,t,s] <= (gen["power_output_maximum"] - gen["power_output_minimum"])*ufg[g,t,s] - max(0, (gen["power_output_maximum"] - gen["ramp_startup_limit"]))*vfg[g,t,s]) # (17)

            if t < data["time_periods"]
                @constraint(m, [s in scenarios], pg[g,t,s] + rg[g,t,s] <= (gen["power_output_maximum"] - gen["power_output_minimum"])*ufg[g,t,s] - max(0, (gen["power_output_maximum"] - gen["ramp_shutdown_limit"]))*wfg[g,t+1,s]) # (18)
            end

            @constraint(m, [s in scenarios], pg[g,t,s] == sum((gen["piecewise_production"][l]["mw"] - gen["piecewise_production"][1]["mw"])*lambda_lg[g,l,t,s] for l in gen_pwl_points[g])) # (21)
            @constraint(m, [s in scenarios], cg[g,t,s] == sum((gen["piecewise_production"][l]["cost"] - gen["piecewise_production"][1]["cost"])*lambda_lg[g,l,t,s] for l in gen_pwl_points[g])) # (22)
            @constraint(m, [s in scenarios], ufg[g,t,s] == sum(lambda_lg[g,l,t,s] for l in gen_pwl_points[g])) # (23)
        end

        for (rg, rgen) in data["renewable_generators"]
            @constraint(m, [s in scenarios], rgen["$s"]["power_output_minimum"][t] <= pw[rg,t,s] <= rgen["$s"]["power_output_maximum"][t]) # (24)
        end
    end

    return m
end

# optimize!(m)
