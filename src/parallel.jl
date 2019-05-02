module parallel
using Compat
using MPI

function init()
    MPI.Init()
end

function finalize()
    MPI.Finalize()
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
        push!(mylist[(i-1)÷(part÷nprocs())+1], i)
    end
    global partitionlist = mylist[myid()+1]
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
end
