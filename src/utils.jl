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
