module parallel
using MPI
using SparseArrays

function init()
    if !MPI.Initialized()
        MPI.Init()
    end
end

function finalize()
    if MPI.Initialized()
        MPI.Finalize()
    end
end

function myid()
    if MPI.Initialized()
        return MPI.Comm_rank(MPI.COMM_WORLD)
    else
        return 0
    end
end

function nprocs()
    if MPI.Initialized()
        return MPI.Comm_size(MPI.COMM_WORLD)
    else
        return 1
    end
end

function getpartition()
    partitionlist
end

function partition(part)
    global mylist = Array{Array{Int64,1},1}(undef, nprocs())
    for i in 1:nprocs()
        mylist[i] = Array{Int64,1}(undef, 0)
    end
    for i in 1:part
        push!(mylist[mod(i-1,nprocs())+1], i)
    end
    global partitionlist = mylist[myid()+1]
end

function sum(x::Number)
    if nprocs() == 1
        return x
    else
        return MPI.Reduce(x, +, 0, MPI.COMM_WORLD)
    end
end

function deserialize!(serialized, counts::Vector{Cint}, x::Vector{T}) where {T}
    @assert length(serialized) == Base.sum(counts)
    sind = 1
    eind = 0
    for i in 1:length(counts)
        eind += counts[i]
        xs = MPI.deserialize(serialized[sind:eind])
        for j in xs
            push!(x, j)
        end
        sind += counts[i]
    end
end

function allcollect(x::Vector{T}) where {T}
    if nprocs() > 1
        x_serialized = MPI.serialize(x)
        counts::Vector{Cint} = MPI.Allgatherv([length(x_serialized)], ones(Cint, nprocs()), MPI.COMM_WORLD)
        collect_serialized = MPI.Allgatherv(x_serialized, counts, MPI.COMM_WORLD)
        x = Vector{T}()
        deserialize!(collect_serialized, counts, x)
    end
    return x
end

function collect(x::Vector{T}) where {T}
    if nprocs() > 1
        x_serialized = MPI.serialize(x)
        counts::Vector{Cint} = MPI.Allgatherv([length(x_serialized)], ones(Cint, nprocs()), MPI.COMM_WORLD)
        collect_serialized = MPI.Gatherv(x_serialized, counts, 0, MPI.COMM_WORLD)
        if myid() == 0
            x = Vector{T}()
            deserialize!(collect_serialized, counts, x)
        end
    end
    return x
end

function combine_dict(x::Dict{Int,Float64})
    if nprocs() > 1
        ks = Vector{Int}()
        vs = Vector{Float64}()
        for (k,v) in x
            push!(ks,k)
            push!(vs,v)
        end
        counts::Vector{Cint} = MPI.Allgatherv([length(ks)], ones(Cint, nprocs()), MPI.COMM_WORLD)
        ks_collected = MPI.Gatherv(ks, counts, 0, MPI.COMM_WORLD)
        vs_collected = MPI.Gatherv(vs, counts, 0, MPI.COMM_WORLD)
        if myid() == 0
            for i in 1:length(ks_collected)
                x[ks_collected[i]] = vs_collected[i]
            end
        end
    end
    return x
end

function combine_dict(x::Dict{Int,SparseVector{Float64}})
    if nprocs() > 1
        ks = Vector{Int}()
        vs = Vector{SparseVector{Float64}}()
        for (k,v) in x
            push!(ks,k)
            push!(vs,v)
        end
        counts::Vector{Cint} = MPI.Allgatherv([length(ks)], ones(Cint, nprocs()), MPI.COMM_WORLD)
        ks_collected = MPI.Gatherv(ks, counts, 0, MPI.COMM_WORLD)
        vs_collected = collect(vs)
        if myid() == 0
            @assert length(vs_collected) == Base.sum(counts)
            for i in 1:length(ks_collected)
                x[ks_collected[i]] = vs_collected[i]
            end
        end
    end
    return x
end

function bcast(buf)
    if nprocs() > 1
        MPI.bcast(buf, 0, MPI.COMM_WORLD)
    end
end

end
