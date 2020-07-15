module parallel
using MPI

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

function collect(x::Vector{T}) where {T}
    if nprocs() == 1
        return x
    else
        counts = MPI.Allgatherv(length(x), 1, MPI.COMM_WORLD)
        return MPI.Allgatherv(x, counts, MPI.COMM_WORLD)
    end
end

function reduce(x::Array{T,1}) where {T<:Number}
    counts = Cint[size(mylist[i],1) for i in 1:size(mylist,1)]
    res=MPI.Allgatherv(x, counts, MPI.COMM_WORLD)
end

function reduce(x::Array{Float64,2})
    counts = Cint[size(mylist[i],1)*size(x,2) for i in 1:size(mylist,1)]
    countssum::Int64=sum(counts)/size(x,2)
    buf=Array{Float64,1}(undef, size(x,1)*size(x,2))
    for i in 1:size(x,1)
        for j in 1:size(x,2)
            buf[(i-1)*size(x,2)+j] = x[i,j]
        end
    end
    res=MPI.Allgatherv(buf, counts, MPI.COMM_WORLD)
    ret=Array{Float64,2}(undef, countssum, size(x,2))
    for i in 1:countssum
        for j in 1:size(x,2)
            ret[i,j]=res[(i-1)*size(x,2)+j]
        end
    end
    ret
end

function bcast(buf)
    if nprocs() > 1
        MPI.bcast(buf, 0, MPI.COMM_WORLD)
    end
end

end
