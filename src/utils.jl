"""
    write_file!

This function writes a vector to a file.

# Arguments
- `v`: vector to write
- `name`: file name
- `dir`: optional argument to give path to the file
"""
function write_file!(v::Vector{T}, name::String, dir = ".") where T <: Number
    if parallel.is_root()
        open("$dir/$name", "w") do io
            for i in eachindex(v)
                println(io, v[i])
            end
        end
    end
end

function write_file!(v::Vector{Dict{Int,T}}, name::String, dir = ".") where T <: Number
    open("$dir/$(name)_$(parallel.myid()).txt", "w") do io
        ids = keys(v[1])
        j = 1
        for id in ids
            if j > 1
                print(io, ",")
            end
            print(io, id)
            j += 1
        end
        print(io, "\n")

        for i in eachindex(v)
            j = 1
            for id in ids
                if j > 1
                    print(io, ",")
                end
                print(io, v[i][id])
                j += 1
            end
            print(io, "\n")
        end
    end
end

mutable struct DataHelper
    dir::String
    BnBNode::Int
    iter::Int

    start_time::Float64
    data_stream::Union{Nothing,IOStream}

    function DataHelper(dir = ".")
        dh = new()
        dh.dir = dir
        dh.BnBNode = 0
        dh.iter = 0
        dh.start_time =time()

        if parallel.is_root()
            dh.data_stream = open("$(dir)/data_stream.csv", "a")
        else
            dh.data_stream = nothing
        end
        return dh
    end
end

function write_data!(dual_bound::Union{Nothing,Float64}, primal_bound::Union{Nothing,Float64},  dh::DataHelper)
    if parallel.is_root()
        io = dh.data_stream
        dh.iter += 1
        println(io, "Iter $(dh.iter), $(time() - dh.start_time), $(dual_bound), $(primal_bound)")
    end
end

function close_all(dh::DataHelper)
    
    if parallel.is_root()
        close(dh.data_stream)
    end
end