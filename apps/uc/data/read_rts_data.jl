using DataFrames
using CSV
using JSON

function read_RTS_Data(rts_dir = "../../../RTS-GMLC/RTS_Data")
    data_dir = "$rts_dir/SourceData"
    ts_dir = "$rts_dir/timeseries_data_files"

    rts_data = Dict()

    for k in ["gen", "branch", "dc_branch", "bus", "reserves", "storage", "simulation_objects", "timeseries_pointers"]
        rts_data[k] = CSV.read("$data_dir/$k.csv")
    end

    rts_data["timeseries_data"] = Dict()
    rts_data["timeseries_data"]["DAY_AHEAD"] = Dict(
        "CSP" => CSV.read("$ts_dir/CSP/DAY_AHEAD_Natural_Inflow.csv"),
        "Hydro" => CSV.read("$ts_dir/Hydro/DAY_AHEAD_Hydro.csv"),
        "Load" => CSV.read("$ts_dir/Load/DAY_AHEAD_regional_Load.csv"),
        "PV" => CSV.read("$ts_dir/PV/DAY_AHEAD_pv.csv"),
        "RTPV" => CSV.read("$ts_dir/RTPV/DAY_AHEAD_rtpv.csv"),
        "WIND" => CSV.read("$ts_dir/WIND/DAY_AHEAD_wind.csv"),
        "Reserves" => Dict()
    )
    rts_data["timeseries_data"]["REAL_TIME"] = Dict(
        "CSP" => CSV.read("$ts_dir/CSP/REAL_TIME_Natural_Inflow.csv"),
        "Hydro" => CSV.read("$ts_dir/Hydro/REAL_TIME_Hydro.csv"),
        "Load" => CSV.read("$ts_dir/Load/REAL_TIME_regional_Load.csv"),
        "PV" => CSV.read("$ts_dir/PV/REAL_TIME_pv.csv"),
        "RTPV" => CSV.read("$ts_dir/RTPV/REAL_TIME_rtpv.csv"),
        "WIND" => CSV.read("$ts_dir/WIND/REAL_TIME_wind.csv"),
        "Reserves" => Dict()
    )
    for resv_type in ["Flex_Down", "Flex_Up", "Reg_Down", "Reg_Up", "Spin_Up_R1", "Spin_Up_R2", "Spin_Up_R3"]
        rts_data["timeseries_data"]["DAY_AHEAD"]["Reserves"][resv_type] = CSV.read("$ts_dir/Reserves/DAY_AHEAD_regional_$resv_type.csv")
    end
    for resv_type in ["Reg_Down", "Reg_Up", "Spin_Up_R1", "Spin_Up_R2", "Spin_Up_R3"]
        rts_data["timeseries_data"]["REAL_TIME"]["Reserves"][resv_type] = CSV.read("$ts_dir/Reserves/REAL_TIME_regional_$resv_type.csv")
    end
    return rts_data
end

"""
Create a dictionary for scenario data from RTS_Data
"""
function create_scenario_data(rts_data::Dict)
    scen_data = Dict(
        "CSP" => Dict(), 
        "Hydro" => Dict(), 
        "PV" => Dict(), 
        "RTPV" => Dict(), 
        "WIND" => Dict(),
        "source" => "https://github.com/GridMod/RTS-GMLC")
    for (k,d) in rts_data["timeseries_data"]["DAY_AHEAD"]
        if in(k, ["CSP", "Hydro", "PV", "RTPV", "WIND"])
            col_names = names(d)
            for j in 5:size(d,2)
                d_mat = reshape(d[:,j], 24, :)
                scen_data[k][col_names[j]] = Dict(s => d_mat[:,s] for s in 1:size(d_mat,2))
            end
        end
    end
    return scen_data
end

function write_scenario_data_file(scen_data::Dict, filename = "scen_data.json")
    open(filename, "w") do fp
        write(fp, JSON.json(scen_data))
    end
    return
end

#=
scratch

rts_data["gen"]
rts_data["timeseries_data"]["DAY_AHEAD"]["PV"][5]
size(rts_data["timeseries_data"]["DAY_AHEAD"]["PV"])

pv = [
    sum(rts_data["timeseries_data"]["DAY_AHEAD"]["PV"][i,6:end]) 
    for i in 1:size(rts_data["timeseries_data"]["DAY_AHEAD"]["PV"],1)
    ]
reshape(pv, 24, :)
length(pv) / 24

aggregate(rts_data["timeseries_data"]["DAY_AHEAD"]["PV"], [6:29], sum)

data["renewable_generators"][1]["320_PV_1"]["power_output_minimum"]
data["renewable_generators"][1]["320_PV_1"]["power_output_maximum"]
=#