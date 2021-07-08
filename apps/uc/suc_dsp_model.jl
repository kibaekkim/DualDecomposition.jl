using DSPopt
using StructJuMP

include("suc_model.jl")

function create_dsp_model(num_scenarios::Int = 1, data_file::String = "./data/model_data.json")

    # Read model data from file
    data = load_suc_data(data_file)

    slow_start_gens = keys(data["slow_generators"])
    fast_start_gens = keys(data["fast_generators"])
    thermal_gens = keys(data["thermal_generators"])
    renewable_gens = keys(data["renewable_generators"])
    time_periods = 1:data["time_periods"]
    scenarios = 1:num_scenarios

    @info "Found $num_scenarios scenarios"
    @info "Found $(length(slow_start_gens)) slow-start generators"
    @info "Found $(length(fast_start_gens)) fast-start generators"
    @info "Found $(length(renewable_gens)) renewable generators"

    gen_startup_categories = Dict(g => 1:length(gen["startup"]) for (g,gen) in data["thermal_generators"])
    gen_pwl_points = Dict(g => 1:length(gen["piecewise_production"]) for (g,gen) in data["thermal_generators"])

    m = StructuredModel(num_scenarios = num_scenarios)

    # First-stage variables
    @variable(m, ug[slow_start_gens,time_periods], binary=true) # Commitment status
    @variable(m, vg[slow_start_gens,time_periods], binary=true) # Startup status
    @variable(m, wg[slow_start_gens,time_periods], binary=true) # Shutdown status
    @variable(m, delta_sg[g in slow_start_gens,gen_startup_categories[g],time_periods], binary=true) # Startup in category ?

    @objective(m, Min,
        # first-stage objective function terms
        sum(
            gen["piecewise_production"][1]["cost"]*ug[g,t] +
            sum(
                gen_startup["cost"]*delta_sg[g,i,t]
            for (i, gen_startup) in enumerate(gen["startup"]))
        for (g, gen) in data["slow_generators"], t in time_periods)
    )

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

        @constraint(m, gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["unit_on_t0"]*(gen["power_output_maximum"] - gen["power_output_minimum"]) - max(0, gen["power_output_maximum"] - gen["ramp_shutdown_limit"])*wg[g,1]) # (10)
    end

    for t in time_periods, (g, gen) in data["slow_generators"]

        @constraint(m, ug[g,t] >= gen["must_run"]) # (11)

        if t > 1
            @constraint(m, ug[g,t] - ug[g,t-1] == vg[g,t] - wg[g,t]) # (12)
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
    end

    for s in scenarios

        blk = StructuredModel(parent = m, id = s, prob = 1/num_scenarios)

        # Second-stage variables
        @variable(blk, ufg[fast_start_gens,time_periods], binary=true) # Commitment status
        @variable(blk, vfg[fast_start_gens,time_periods], binary=true) # Startup status
        @variable(blk, wfg[fast_start_gens,time_periods], binary=true) # Shutdown status
        @variable(blk, delta_sfg[g in fast_start_gens,gen_startup_categories[g],time_periods], binary=true) # Startup in category ?
        @variable(blk, cg[thermal_gens,time_periods]) # Production cost
        @variable(blk, pg[thermal_gens,time_periods] >= 0) # Thermal generation
        @variable(blk, 
            max(0, data["renewable_generators"][rg]["$s"]["power_output_minimum"][t]) <= 
            pw[rg=renewable_gens,t=time_periods] <=
            data["renewable_generators"][rg]["$s"]["power_output_maximum"][t]
        ) # Renewable generation
        @variable(blk, rg[thermal_gens,time_periods] >= 0) # Spinning reserve
        @variable(blk, 0 <= lambda_lg[g in thermal_gens,gen_pwl_points[g],time_periods] <= 1) # Fraction of power in piecewise generation point

        @objective(blk, Min,
            sum(
                gen["piecewise_production"][1]["cost"]*ufg[g,t] +
                sum(
                    gen_startup["cost"]*delta_sfg[g,i,t]
                for (i, gen_startup) in enumerate(gen["startup"]))
            for (g, gen) in data["fast_generators"], t in time_periods) +
            sum(cg[g,t] 
            for t in time_periods, (g, gen) in data["thermal_generators"])
        )

        for (g, gen) in data["slow_generators"]
            @constraint(blk, pg[g,1] + rg[g,1] - gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["ramp_up_limit"]) # (8)
            @constraint(blk, gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) - pg[g,1] <= gen["ramp_down_limit"]) # (9)
        end

        for (g, gen) in data["fast_generators"]
    
            if gen["unit_on_t0"] == 1
                @constraint(blk, sum( (ufg[g,t]-1) for t in 1:min(data["time_periods"], gen["time_up_minimum"] - gen["time_up_t0"]) ) == 0) # (4)
            else
                @constraint(blk, sum( ufg[g,t] for t in 1:min(data["time_periods"], gen["time_down_minimum"] - gen["time_down_t0"]) ) == 0) # (5)
            end
    
            @constraint(blk, ufg[g,1] - gen["unit_on_t0"] == vfg[g,1] - wfg[g,1]) # (6)
    
            @constraint(blk, 0 ==
                sum(
                    sum(
                        delta_sfg[g,i,t]
                    for t in max(1, gen["startup"][i+1]["lag"] - gen["time_down_t0"] + 1):min(gen["startup"][i+1]["lag"]-1, data["time_periods"]))
                for (i,startup) in enumerate(gen["startup"][1:end-1]))
            ) # (7)
    
            @constraint(blk, pg[g,1] + rg[g,1] - gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["ramp_up_limit"]) # (8)
            @constraint(blk, gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) - pg[g,1] <= gen["ramp_down_limit"]) # (9)
            @constraint(blk, gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["unit_on_t0"]*(gen["power_output_maximum"] - gen["power_output_minimum"]) - max(0, gen["power_output_maximum"] - gen["ramp_shutdown_limit"])*wfg[g,1]) # (10)
        end

        for t in time_periods
            @constraint(blk, 
                sum( pg[g,t] + gen["power_output_minimum"]*ug[g,t] for (g, gen) in data["slow_generators"] ) +
                sum( pg[g,t] + gen["power_output_minimum"]*ufg[g,t] for (g, gen) in data["fast_generators"] ) +
                sum( pw[g,t] for g in renewable_gens)
                == data["demand"][t]
            ) # (2)

            @constraint(blk, sum(rg[g,t] for g in thermal_gens) >= data["reserves"][t]) # (3)

            for (g, gen) in data["slow_generators"]

                if t > 1
                    @constraint(blk, pg[g,t] + rg[g,t] - pg[g,t-1] <= gen["ramp_up_limit"]) # (19)
                    @constraint(blk, pg[g,t-1] - pg[g,t] <= gen["ramp_down_limit"]) # (20)
                end

                @constraint(blk, pg[g,t] + rg[g,t] <= (gen["power_output_maximum"] - gen["power_output_minimum"])*ug[g,t] - max(0, (gen["power_output_maximum"] - gen["ramp_startup_limit"]))*vg[g,t]) # (17)

                if t < data["time_periods"]
                    @constraint(blk, pg[g,t] + rg[g,t] <= (gen["power_output_maximum"] - gen["power_output_minimum"])*ug[g,t] - max(0, (gen["power_output_maximum"] - gen["ramp_shutdown_limit"]))*wg[g,t+1]) # (18)
                end

                @constraint(blk, pg[g,t] == sum((gen["piecewise_production"][l]["mw"] - gen["piecewise_production"][1]["mw"])*lambda_lg[g,l,t] for l in gen_pwl_points[g])) # (21)
                @constraint(blk, cg[g,t] == sum((gen["piecewise_production"][l]["cost"] - gen["piecewise_production"][1]["cost"])*lambda_lg[g,l,t] for l in gen_pwl_points[g])) # (22)
                @constraint(blk, ug[g,t] == sum(lambda_lg[g,l,t] for l in gen_pwl_points[g])) # (23)
            end

            for (g, gen) in data["fast_generators"]

                @constraint(blk, ufg[g,t] >= gen["must_run"]) # (11)

                if t > 1
                    @constraint(blk, ufg[g,t] - ufg[g,t-1] == vfg[g,t] - wfg[g,t]) # (12)
                    @constraint(blk, pg[g,t] + rg[g,t] - pg[g,t-1] <= gen["ramp_up_limit"]) # (19)
                    @constraint(blk, pg[g,t-1] - pg[g,t] <= gen["ramp_down_limit"]) # (20)
                end

                if t >= gen["time_up_minimum"] || t == data["time_periods"]
                    @constraint(blk, sum( vfg[g,t2] for t2 in (t-min(gen["time_up_minimum"],data["time_periods"])+1):t) <= ufg[g,t])  # (13)
                end

                if t >= gen["time_down_minimum"] || t == data["time_periods"]
                    @constraint(blk, sum( wfg[g,t2] for t2 in (t-min(gen["time_down_minimum"],data["time_periods"])+1):t) <= 1 - ufg[g,t])  # (14)
                end

                for (si,startup) in enumerate(gen["startup"][1:end-1])
                    if t >= gen["startup"][si+1]["lag"]
                        time_range = startup["lag"]:(gen["startup"][si+1]["lag"]-1)
                        @constraint(blk, delta_sfg[g,si,t] <= sum(wfg[g,t-i] for i in time_range)) # (15)
                    end
                end

                @constraint(blk, vfg[g,t] == sum( delta_sfg[g,i,t] for i in gen_startup_categories[g])) # (16)

                @constraint(blk, pg[g,t] + rg[g,t] <= (gen["power_output_maximum"] - gen["power_output_minimum"])*ufg[g,t] - max(0, (gen["power_output_maximum"] - gen["ramp_startup_limit"]))*vfg[g,t]) # (17)

                if t < data["time_periods"]
                    @constraint(blk, pg[g,t] + rg[g,t] <= (gen["power_output_maximum"] - gen["power_output_minimum"])*ufg[g,t] - max(0, (gen["power_output_maximum"] - gen["ramp_shutdown_limit"]))*wfg[g,t+1]) # (18)
                end

                @constraint(blk, pg[g,t] == sum((gen["piecewise_production"][l]["mw"] - gen["piecewise_production"][1]["mw"])*lambda_lg[g,l,t] for l in gen_pwl_points[g])) # (21)
                @constraint(blk, cg[g,t] == sum((gen["piecewise_production"][l]["cost"] - gen["piecewise_production"][1]["cost"])*lambda_lg[g,l,t] for l in gen_pwl_points[g])) # (22)
                @constraint(blk, ufg[g,t] == sum(lambda_lg[g,l,t] for l in gen_pwl_points[g])) # (23)
            end
        end
    end

    return m
end

function create_dsp_model_fewer_couples(num_scenarios::Int = 1, data_file::String = "./data/model_data.json")

    # Read model data from file
    data = load_suc_data(data_file)

    slow_start_gens = keys(data["slow_generators"])
    fast_start_gens = keys(data["fast_generators"])
    thermal_gens = keys(data["thermal_generators"])
    renewable_gens = keys(data["renewable_generators"])
    time_periods = 1:data["time_periods"]
    scenarios = 1:num_scenarios

    @info "Found $num_scenarios scenarios"
    @info "Found $(length(slow_start_gens)) slow-start generators"
    @info "Found $(length(fast_start_gens)) fast-start generators"
    @info "Found $(length(renewable_gens)) renewable generators"

    gen_startup_categories = Dict(g => 1:length(gen["startup"]) for (g,gen) in data["thermal_generators"])
    gen_pwl_points = Dict(g => 1:length(gen["piecewise_production"]) for (g,gen) in data["thermal_generators"])

    m = StructuredModel(num_scenarios = num_scenarios)

    # First-stage variables
    @variable(m, ug[slow_start_gens,time_periods], binary=true) # Commitment status

    @objective(m, Min,
        # first-stage objective function terms
        sum(
            gen["piecewise_production"][1]["cost"]*ug[g,t] 
        for (g, gen) in data["slow_generators"], 
            t in time_periods)
    )

    for (g, gen) in data["slow_generators"]
        if gen["unit_on_t0"] == 1
            @constraint(m, sum( (ug[g,t]-1) for t in 1:min(data["time_periods"], gen["time_up_minimum"] - gen["time_up_t0"]) ) == 0) # (4)
        else
            @constraint(m, sum( ug[g,t] for t in 1:min(data["time_periods"], gen["time_down_minimum"] - gen["time_down_t0"]) ) == 0) # (5)
        end
    end

    for t in time_periods, (g, gen) in data["slow_generators"]

        @constraint(m, ug[g,t] >= gen["must_run"]) # (11)
    end

    for s in scenarios

        blk = StructuredModel(parent = m, id = s, prob = 1/num_scenarios)

        @variable(blk, vg[slow_start_gens,time_periods], binary=true) # Startup status
        @variable(blk, wg[slow_start_gens,time_periods], binary=true) # Shutdown status
        @variable(blk, delta_sg[g in slow_start_gens,gen_startup_categories[g],time_periods], binary=true) # Startup in category ?
        # Second-stage variables
        @variable(blk, ufg[fast_start_gens,time_periods], binary=true) # Commitment status
        @variable(blk, vfg[fast_start_gens,time_periods], binary=true) # Startup status
        @variable(blk, wfg[fast_start_gens,time_periods], binary=true) # Shutdown status
        @variable(blk, delta_sfg[g in fast_start_gens,gen_startup_categories[g],time_periods], binary=true) # Startup in category ?
        @variable(blk, cg[thermal_gens,time_periods]) # Production cost
        @variable(blk, pg[thermal_gens,time_periods] >= 0) # Thermal generation
        @variable(blk, 
            max(0, data["renewable_generators"][rg]["$s"]["power_output_minimum"][t]) <= 
            pw[rg=renewable_gens,t=time_periods] <=
            data["renewable_generators"][rg]["$s"]["power_output_maximum"][t]
        ) # Renewable generation
        @variable(blk, rg[thermal_gens,time_periods] >= 0) # Spinning reserve
        @variable(blk, 0 <= lambda_lg[g in thermal_gens,gen_pwl_points[g],time_periods] <= 1) # Fraction of power in piecewise generation point

        @objective(blk, Min,
            sum(
                gen_startup["cost"]*delta_sg[g,i,t]
            for (g, gen) in data["slow_generators"], 
                (i, gen_startup) in enumerate(gen["startup"]), 
                t in time_periods) +
            sum(
                gen["piecewise_production"][1]["cost"]*ufg[g,t] +
                sum(
                    gen_startup["cost"]*delta_sfg[g,i,t]
                for (i, gen_startup) in enumerate(gen["startup"]))
            for (g, gen) in data["fast_generators"], 
                t in time_periods) +
            sum(cg[g,t] 
            for t in time_periods, 
                (g, gen) in data["thermal_generators"])
        )

        for (g, gen) in data["slow_generators"]

            @constraint(blk, ug[g,1] - gen["unit_on_t0"] == vg[g,1] - wg[g,1]) # (6)

            @constraint(blk, 0 ==
                sum(
                    sum(
                        delta_sg[g,i,t]
                    for t in max(1, gen["startup"][i+1]["lag"] - gen["time_down_t0"] + 1):min(gen["startup"][i+1]["lag"]-1, data["time_periods"]))
                for (i,startup) in enumerate(gen["startup"][1:end-1]))
            ) # (7)

            @constraint(blk, pg[g,1] + rg[g,1] - gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["ramp_up_limit"]) # (8)
            @constraint(blk, gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) - pg[g,1] <= gen["ramp_down_limit"]) # (9)

            @constraint(blk, gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["unit_on_t0"]*(gen["power_output_maximum"] - gen["power_output_minimum"]) - max(0, gen["power_output_maximum"] - gen["ramp_shutdown_limit"])*wg[g,1]) # (10)
        end

        for (g, gen) in data["fast_generators"]
    
            if gen["unit_on_t0"] == 1
                @constraint(blk, sum( (ufg[g,t]-1) for t in 1:min(data["time_periods"], gen["time_up_minimum"] - gen["time_up_t0"]) ) == 0) # (4)
            else
                @constraint(blk, sum( ufg[g,t] for t in 1:min(data["time_periods"], gen["time_down_minimum"] - gen["time_down_t0"]) ) == 0) # (5)
            end
    
            @constraint(blk, ufg[g,1] - gen["unit_on_t0"] == vfg[g,1] - wfg[g,1]) # (6)
    
            @constraint(blk, 0 ==
                sum(
                    sum(
                        delta_sfg[g,i,t]
                    for t in max(1, gen["startup"][i+1]["lag"] - gen["time_down_t0"] + 1):min(gen["startup"][i+1]["lag"]-1, data["time_periods"]))
                for (i,startup) in enumerate(gen["startup"][1:end-1]))
            ) # (7)
    
            @constraint(blk, pg[g,1] + rg[g,1] - gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["ramp_up_limit"]) # (8)
            @constraint(blk, gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) - pg[g,1] <= gen["ramp_down_limit"]) # (9)
            @constraint(blk, gen["unit_on_t0"]*(gen["power_output_t0"] - gen["power_output_minimum"]) <= gen["unit_on_t0"]*(gen["power_output_maximum"] - gen["power_output_minimum"]) - max(0, gen["power_output_maximum"] - gen["ramp_shutdown_limit"])*wfg[g,1]) # (10)
        end

        for t in time_periods
            @constraint(blk, 
                sum( pg[g,t] + gen["power_output_minimum"]*ug[g,t] for (g, gen) in data["slow_generators"] ) +
                sum( pg[g,t] + gen["power_output_minimum"]*ufg[g,t] for (g, gen) in data["fast_generators"] ) +
                sum( pw[g,t] for g in renewable_gens)
                == data["demand"][t]
            ) # (2)

            @constraint(blk, sum(rg[g,t] for g in thermal_gens) >= data["reserves"][t]) # (3)

            for (g, gen) in data["slow_generators"]

                if t > 1
                    @constraint(blk, ug[g,t] - ug[g,t-1] == vg[g,t] - wg[g,t]) # (12)
                end
        
                if t >= gen["time_up_minimum"] || t == data["time_periods"]
                    @constraint(blk, sum( vg[g,t2] for t2 in (t-min(gen["time_up_minimum"],data["time_periods"])+1):t) <= ug[g,t])  # (13)
                end
        
                if t >= gen["time_down_minimum"] || t == data["time_periods"]
                    @constraint(blk, sum( wg[g,t2] for t2 in (t-min(gen["time_down_minimum"],data["time_periods"])+1):t) <= 1 - ug[g,t])  # (14)
                end
        
                for (si,startup) in enumerate(gen["startup"][1:end-1])
                    if t >= gen["startup"][si+1]["lag"]
                        time_range = startup["lag"]:(gen["startup"][si+1]["lag"]-1)
                        @constraint(blk, delta_sg[g,si,t] <= sum(wg[g,t-i] for i in time_range)) # (15)
                    end
                end
        
                @constraint(blk, vg[g,t] == sum( delta_sg[g,i,t] for i in gen_startup_categories[g])) # (16)

                if t > 1
                    @constraint(blk, pg[g,t] + rg[g,t] - pg[g,t-1] <= gen["ramp_up_limit"]) # (19)
                    @constraint(blk, pg[g,t-1] - pg[g,t] <= gen["ramp_down_limit"]) # (20)
                end

                @constraint(blk, pg[g,t] + rg[g,t] <= (gen["power_output_maximum"] - gen["power_output_minimum"])*ug[g,t] - max(0, (gen["power_output_maximum"] - gen["ramp_startup_limit"]))*vg[g,t]) # (17)

                if t < data["time_periods"]
                    @constraint(blk, pg[g,t] + rg[g,t] <= (gen["power_output_maximum"] - gen["power_output_minimum"])*ug[g,t] - max(0, (gen["power_output_maximum"] - gen["ramp_shutdown_limit"]))*wg[g,t+1]) # (18)
                end

                @constraint(blk, pg[g,t] == sum((gen["piecewise_production"][l]["mw"] - gen["piecewise_production"][1]["mw"])*lambda_lg[g,l,t] for l in gen_pwl_points[g])) # (21)
                @constraint(blk, cg[g,t] == sum((gen["piecewise_production"][l]["cost"] - gen["piecewise_production"][1]["cost"])*lambda_lg[g,l,t] for l in gen_pwl_points[g])) # (22)
                @constraint(blk, ug[g,t] == sum(lambda_lg[g,l,t] for l in gen_pwl_points[g])) # (23)
            end

            for (g, gen) in data["fast_generators"]

                @constraint(blk, ufg[g,t] >= gen["must_run"]) # (11)

                if t > 1
                    @constraint(blk, ufg[g,t] - ufg[g,t-1] == vfg[g,t] - wfg[g,t]) # (12)
                    @constraint(blk, pg[g,t] + rg[g,t] - pg[g,t-1] <= gen["ramp_up_limit"]) # (19)
                    @constraint(blk, pg[g,t-1] - pg[g,t] <= gen["ramp_down_limit"]) # (20)
                end

                if t >= gen["time_up_minimum"] || t == data["time_periods"]
                    @constraint(blk, sum( vfg[g,t2] for t2 in (t-min(gen["time_up_minimum"],data["time_periods"])+1):t) <= ufg[g,t])  # (13)
                end

                if t >= gen["time_down_minimum"] || t == data["time_periods"]
                    @constraint(blk, sum( wfg[g,t2] for t2 in (t-min(gen["time_down_minimum"],data["time_periods"])+1):t) <= 1 - ufg[g,t])  # (14)
                end

                for (si,startup) in enumerate(gen["startup"][1:end-1])
                    if t >= gen["startup"][si+1]["lag"]
                        time_range = startup["lag"]:(gen["startup"][si+1]["lag"]-1)
                        @constraint(blk, delta_sfg[g,si,t] <= sum(wfg[g,t-i] for i in time_range)) # (15)
                    end
                end

                @constraint(blk, vfg[g,t] == sum( delta_sfg[g,i,t] for i in gen_startup_categories[g])) # (16)

                @constraint(blk, pg[g,t] + rg[g,t] <= (gen["power_output_maximum"] - gen["power_output_minimum"])*ufg[g,t] - max(0, (gen["power_output_maximum"] - gen["ramp_startup_limit"]))*vfg[g,t]) # (17)

                if t < data["time_periods"]
                    @constraint(blk, pg[g,t] + rg[g,t] <= (gen["power_output_maximum"] - gen["power_output_minimum"])*ufg[g,t] - max(0, (gen["power_output_maximum"] - gen["ramp_shutdown_limit"]))*wfg[g,t+1]) # (18)
                end

                @constraint(blk, pg[g,t] == sum((gen["piecewise_production"][l]["mw"] - gen["piecewise_production"][1]["mw"])*lambda_lg[g,l,t] for l in gen_pwl_points[g])) # (21)
                @constraint(blk, cg[g,t] == sum((gen["piecewise_production"][l]["cost"] - gen["piecewise_production"][1]["cost"])*lambda_lg[g,l,t] for l in gen_pwl_points[g])) # (22)
                @constraint(blk, ufg[g,t] == sum(lambda_lg[g,l,t] for l in gen_pwl_points[g])) # (23)
            end
        end
    end

    return m
end

#=
m = create_dsp_model(2)
status = optimize!(m,
    is_stochastic = true,
    solve_type = DSPopt.Legacy,
    param = "params.txt"
)
=#
