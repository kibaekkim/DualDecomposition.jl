@testset "MPI tests" begin
    testdir = @__DIR__
    run(`$(Base.julia_cmd()) $(joinpath(testdir, "parallel.jl"))`)
    mpiexec(cmd ->run(`$cmd -np 2 $(Base.julia_cmd()) $(joinpath(testdir, "parallel.jl"))`))
    mpiexec(cmd ->run(`$cmd -np 3 $(Base.julia_cmd()) $(joinpath(testdir, "parallel.jl"))`))
    run(`$(Base.julia_cmd()) $(joinpath(testdir, "../examples/farmer_mpi.jl"))`)
    mpiexec(cmd ->run(`$cmd -np 2 $(Base.julia_cmd()) $(joinpath(testdir, "../examples/farmer_mpi.jl"))`))
    mpiexec(cmd ->run(`$cmd -np 3 $(Base.julia_cmd()) $(joinpath(testdir, "../examples/farmer_mpi.jl"))`))
end