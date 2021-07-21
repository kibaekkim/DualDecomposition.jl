"""
    CouplingVariableKey

Key to map oupling variables. `block_id` identifies the problem block containing 
the variable, and `coupling_id` identifies a set of variables whose values are equal.
"""
struct CouplingVariableKey
    block_id::Int
    coupling_id
end

"""
    CouplingVariableRef

Coupling variable reference with `key::CouplingVariableKey`.
"""
struct CouplingVariableRef
    key::CouplingVariableKey
    ref::JuMP.VariableRef
    function CouplingVariableRef(key::CouplingVariableKey, ref::JuMP.VariableRef)
        return new(key, ref)
    end
end

CouplingVariableRef(block_id::Int, coupling_id, ref::JuMP.VariableRef) = CouplingVariableRef(CouplingVariableKey(block_id, coupling_id), ref)

"""
    BlockModel

Block model struture contrains a set of `JuMP.Model` objects, each of which
represents a sub-block model with the information of how to couple these block
models. 
"""
abstract type AbstractBlockModel end

mutable struct BlockModel <: AbstractBlockModel
    model::Dict{Int,JuMP.Model} # Dictionary of block models
    coupling_variables::Vector{CouplingVariableRef} # array of variables that couple block models
    variables_by_couple::Dict{Any,Vector{CouplingVariableKey}} # maps `couple_id` to `CouplingVariableKey`

    dual_bound::Float64
    dual_solution::Vector{Float64}

    # TODO: These may be available with heuristics.
    primal_bound::Float64
    primal_solution::Dict{Int, Float64} #coupling_id : value 
    combined_weights::Dict{Int, Float64} # block_id : value 
    record::Dict{Any, Any}

    function BlockModel()
        return new(
            Dict(), 
            [],
            Dict(),
            0.0,
            [],
            +Inf,
            Dict(),
            Dict(),
            Dict())
    end
end

"""
    add_block_model!

Add block model `model` to `block_model::AbstractBlockModel` with `block_id`.
"""
function add_block_model!(block_model::AbstractBlockModel, block_id::Integer, model::JuMP.Model)
    block_model.model[block_id] = model
end

"""
    num_blocks

Number of blocks in `block_model::AbstractBlockModel`
"""
num_blocks(block_model::AbstractBlockModel) = length(block_model.model)

"""
    block_model

This returns a dictionary of `JuMP.Model` objects.
"""
block_model(block_model::AbstractBlockModel) = block_model.model

"""
    block_model

This returns a `JuMP.Model` object for a given `block_id`.
"""
block_model(block_model::AbstractBlockModel, block_id::Integer) = block_model.model[block_id]

"""
    has_block_model

This returns true if `block_model::AbstractBlockModel` has key `block_id::Integer`; false otherwise.
"""
has_block_model(block_model::AbstractBlockModel, block_id::Integer) = haskey(block_model.model, block_id)

"""
    num_coupling_variables

This returns the number of coupling variables in `block_model::AbstractBlockModel`.
"""
num_coupling_variables(block_model::AbstractBlockModel) = length(block_model.coupling_variables)

"""
    coupling_variables

This returns the array of coupling variables in `block_model::AbstractBlockModel`.
"""
coupling_variables(block_model::AbstractBlockModel) = block_model.coupling_variables

"""
    set_coupling_variables!

This sets coupling variables `variables` to `block_model::AbstractBlockModel`.
"""
function set_coupling_variables!(block_model::AbstractBlockModel, variables::Vector{CouplingVariableRef})
    block_model.coupling_variables = variables
end

"""
    set_variables_by_couple!

This sets `BlockModel.variables_by_couple`.
"""
function set_variables_by_couple!(block_model::AbstractBlockModel, variables::Vector{CouplingVariableKey})
    block_model.variables_by_couple = Dict(v.coupling_id => [] for v in variables)
    for v in variables
        push!(block_model.variables_by_couple[v.coupling_id], v)
    end
end

"""
    set_block_weights!
    set the weights of the blocks that will be used by the primal heuristics
"""
function set_block_weights!(block_model::AbstractBlockModel, weights::Dict{Int, Float64})
    block_model.combined_weights = weights
end

"""
    dual_objective_value

This returns the dual objective value obtained from a method.
"""
dual_objective_value(block_model::AbstractBlockModel) = block_model.dual_bound

"""
    dual_solution

This returns a vector of dual solution obtained from a method.
"""
dual_solution(block_model::AbstractBlockModel) = block_model.dual_solution

"""
primal_objective_value

This returns the best primal objective value obtained from a method.
"""
primal_objective_value(block_model::AbstractBlockModel) = block_model.primal_bound

"""
primal_solution

This returns a vector of the best primal solution obtained from a method.
"""
primal_solution(block_model::AbstractBlockModel) = block_model.primal_solution
