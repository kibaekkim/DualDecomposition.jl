
using ScenTrees
using JLD2


if isfile("$GRAPHDIR/tree.jld2")
    LoadTree = load_object("$GRAPHDIR/tree.jld2")
else
    using CSV
    using DataFrames
    using Distributions
    nIterations = 100000

    dLoad1 = CSV.read(joinpath(@__DIR__,"data/UC_data/PJMestimated2016.txt"), DataFrame)
    dLoad1 = dLoad1[sortperm(dLoad1[:,1]),3]
    dLoad1 = permutedims(reshape(dLoad1,(nStage*nPeriod,8736÷(nStage*nPeriod))), [2,1])

    arr = ones(Int, nPeriod)
    arr[1] = nBranch
    branching = Array{Int}(undef,0)
    for i=1:nStage
        append!(branching,arr)
    end
    branching[1] = 1


    LoadTree = tree_approximation!(ScenTrees.Tree(branching,1),kernel_scenarios(dLoad1, Logistic; Markovian = false), nIterations, 2, 2);
    save_object("$GRAPHDIR/tree.jld2", LoadTree)
    # tree_plot(LoadTree)
end
