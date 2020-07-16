using Test
using SparseArrays
using DualDecomposition

const DD = DualDecomposition
const parallel = DD.parallel

parallel.init()

@testset "combine_dict with $(parallel.nprocs()) procs" begin
    x = Dict{Int,Float64}()
    xlen = 3
    for i = 1:xlen
        x[xlen*parallel.myid()+i] = xlen*parallel.myid()+i*0.2
    end

    x_combined = parallel.combine_dict(x)
    if parallel.is_root()
        for j = 1:parallel.nprocs(), i = 1:xlen
            @test x_combined[xlen*(j-1)+i] == xlen*(j-1)+i*0.2
        end
    end

    y = Dict{Int,SparseVector{Float64}}()
    ylen = 3
    for i = 1:ylen
        y[ylen*parallel.myid()+i] = [ylen*parallel.myid()+i*0.2+j for j=1:3]
    end
    y_combined = parallel.combine_dict(y)
    if parallel.is_root()
        for id = 1:parallel.nprocs(), i = 1:ylen
            @test y_combined[ylen*(id-1)+i] == sparsevec([ylen*(id-1)+i*0.2+j for j=1:3])
            @test typeof(y_combined[ylen*(id-1)+i]) == SparseVector{Float64,Int64}
        end
    end
end

parallel.finalize()