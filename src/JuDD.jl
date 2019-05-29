#=
Dual Decomposition of Stochastic Programming in Julia

Assume: two-stage stochastic programming
=#

module JuDD

using Compat
using JuMP, CPLEX
using BundleMethod
const BM = BundleMethod

abstract type AbstractAlg end

include("LagrangeDual.jl")
include("ADMM.jl")

export
    LagrangeDualAlg,
    AdmmAlg,
    add_scenario_model,
    add_scenario_models,
    set_nonanticipativity_vars
    # solve

function print_warning()
    Compat.@warn("This is an abstract function.")
end

function add_scenario_models(LD::AbstractAlg, ns::Integer, p::Vector{Float64},
        create_scenario::Function)
    print_warning()
end

function get_scenario_model(alg::AbstractAlg, s::Integer)
    print_warning()
end

function set_nonanticipativity_vars(alg::AbstractAlg, names::Vector{Symbol})
    print_warning()
end

function solve(alg::AbstractAlg, solver)
    print_warning()
end

end  # modJuDD
