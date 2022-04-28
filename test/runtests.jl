using Test
using DualDecomposition
using JuMP, Ipopt, GLPK
using MPI
using LinearAlgebra

const DD = DualDecomposition

include("farmer.jl")
include("mpi.jl")
include("investment.jl")
include("heuristics.jl")
include("investment_dro.jl")
