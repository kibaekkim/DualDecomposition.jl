
"""
DR_BlockModel

Block model struture contrains a set of `JuMP.Model` objects, each of which
represents a sub-block model with the information of how to couple these block
models. 
"""

mutable struct DR_BlockModel <: AbstractBlockModel
    model::Dict{Int,JuMP.Model} # Dictionary of block models
    coupling_variables::Vector{CouplingVariableRef} # array of variables that couple block models
    variables_by_couple::Dict{Any,Vector{CouplingVariableKey}} # maps `couple_id` to `CouplingVariableKey`

    dual_bound::Float64
    dual_solution::Vector{Float64}

    P_solution::Dict{Int,Float64}

    # TODO: These may be available with heuristics.
    primal_bound::Float64
    primal_solution::Dict{Any, Float64} #coupling_id : value 
    combined_weights::Dict{Int, Float64} # block_id : value 
    record::Dict{Any, Any}

    function DR_BlockModel()
        return new(
            Dict(), 
            [],
            Dict(),
            0.0,
            [],
            Dict(),
            +Inf,
            Dict(),
            Dict(),
            Dict())
    end
end