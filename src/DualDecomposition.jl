"""
    DualDecomposition

This module implements a block model for dual Decomposition and Lagrangian 
dual method for solving the block model. Applications of the dual Decomposition
include stochastic programming, temporal decomposition, and network decomposition.
"""
module DualDecomposition

using JuMP
using BundleMethod
using SparseArrays
const BM = BundleMethod

export BM

abstract type AbstractMethod end

"""
    run!

Empty function for `AbstractMethod`
"""
function run! end

include("BlockModel.jl")
include("LagrangeDual.jl")

end  # module DualDecomposition
