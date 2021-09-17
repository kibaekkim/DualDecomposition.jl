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


    subobj_values::Union{Nothing,IOStream}
    primal_value::Union{Nothing,IOStream}

    lagrange_value::Union{Nothing,IOStream}
    sub_solution::Dict{Int,IOStream}

    function DataHelper(dir = ".")
        dh = new()
        dh.dir = dir
        dh.BnBNode = 0
        dh.iter = 0

        dh.sub_solution = Dict{Int, IOStream}()
        for block_id in parallel.getpartition()
            dh.sub_solution[block_id] = open("$(dir)/sub_solution_$(block_id).csv", "a")
        end

        if parallel.is_root()
            dh.subobj_values = open("$(dir)/subobj_values.csv", "a")
            dh.primal_value = open("$(dir)/primal_value.csv", "a")
            dh.lagrange_value = open("$(dir)/lagrange_value.csv", "a")
        else
            dh.dual_value = nothing
            dh.primal_value = nothing
        end
        return dh
    end
end

function write_line!(v::Any, dh::DataHelper, io::IOStream)
    if parallel.is_root()
        println(io, "Iter $(dh.iter), $(v)")
    end
end

function write_line!(v::Dict{<:Any,<:Any}, keys::Vector{<:Any}, dh::DataHelper, io::IOStream)
    print(io, "Iter $(dh.iter)")
    for key in keys
        print(io, ", ")
        print(io, v[key])
    end
    print(io, "\n")
end

function write_line!(v::Vector{<:Any}, dh::DataHelper, io::IOStream)
    print(io, "Iter $(dh.iter)")
    for var in v
        print(io, ", ")
        print(io, var)
    end
    print(io, "\n")
end

function write_line!(v::SparseVector, keys::Vector{Int}, dh::DataHelper, io::IOStream)
    print(io, "Iter $(dh.iter)")
    for key in keys
        try
            print(io, ", $(v[key])")
        catch
            print(io, ", ")
        end
    end
    print(io, "\n")
end

function close_all(dh::DataHelper)
    
    if parallel.is_root()
        close(dh.subobj_values)
        close(dh.primal_value)
        close(dh.lagrange_value)
    end
    for block_id in parallel.getpartition()
        close(dh.sub_solution[block_id])
    end
end