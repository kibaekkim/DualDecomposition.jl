"""
This is to generate a dataset independent to the original sources.
"""

include("read_rts_data.jl")

function create_model_data(scen_data::Dict, data_dir = "./data/pglib-uc/rts_gmlc")
    all_data_files = readdir(data_dir)

    raw_data = Dict()
    data = Dict()
    for (i,data_file) in enumerate(all_data_files)
        raw_data[i] = JSON.parsefile("$data_dir/$data_file")
    end
    data["thermal_generators"] = raw_data[1]["thermal_generators"]
    data["time_periods"] = 24
    data["reserves"] = raw_data[1]["reserves"]
    data["demand"] = raw_data[1]["demand"]

    data["renewable_generators"] = Dict{String,Dict}()
    for (gentype,gen) in scen_data
        if in(gentype, ["WIND", "PV"])
            for (k,g) in gen
                data["renewable_generators"][k] = Dict{Int,Dict}(
                    s => Dict(
                        "name" => k,
                        "power_output_maximum" => val,
                        "power_output_minimum" => zeros(length(val))
                    )
                    for (s,val) in g
                )
            end
        elseif gentype != "source"
            for (k,g) in gen
                data["renewable_generators"][k] = Dict{Int,Dict}(
                    s => Dict(
                        "name" => k,
                        "power_output_maximum" => val,
                        "power_output_minimum" => val
                    )
                    for (s,val) in g
                )
            end
        end
    end

    return data
end

function write_model_data_file(model_data::Dict, filename = "model_data.json")
    open(filename, "w") do fp
        write(fp, JSON.json(model_data))
    end
    return
end

#= 
"Generate JSON file"
write_model_data_file(
    create_model_data(
        create_scenario_data(
            read_RTS_Data()
        )
    ),
    "./data/model_data.json"
)
=#
