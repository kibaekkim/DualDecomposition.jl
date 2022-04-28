using Test
using DualDecomposition
using JuMP, Ipopt, GLPK
using MPI
using LinearAlgebra
using Random

const DD = DualDecomposition

include("investment_dro.jl")
include("farmer.jl")
include("mpi.jl")
include("investment.jl")
include("heuristics.jl")
