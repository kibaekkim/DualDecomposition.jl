"""
JuMP variable extension

# Usage: node::SubTreeNode
    julia> @variable(model, x[l=1:L] >= 0, OutStateInfo, subnode = node, ref_symbol = :x)
    julia> @variable(model, x_[l=1:L], InStateInfo, subnode = node, ref_symbol = :x)
"""
abstract type MultistageInfo end

struct InStateInfo <: MultistageInfo
    info::JuMP.VariableInfo
    subnode::SubTreeNode
    ref_symbol::Symbol
end

struct OutStateInfo <: MultistageInfo
    info::JuMP.VariableInfo
    subnode::SubTreeNode
    ref_symbol::Symbol
end

struct ControlInfo <: MultistageInfo
    info::JuMP.VariableInfo
    subnode::SubTreeNode
    ref_symbol::Symbol
end

function JuMP.build_variable(
    _error::Function,
    info::JuMP.VariableInfo,
    infotype::Type{<:MultistageInfo};
    subnode = nothing,
    ref_symbol = nothing,
    kwargs...
)
    if isnothing(subnode)
        _error("Enter current subtree node")
    end
    if isnothing(ref_symbol)
        _error("Enter reference symbol")
    end
    return infotype(info, subnode, ref_symbol)
end

function JuMP.add_variable(
    model::JuMP.Model,
    info::MultistageInfo,
    name::String,
)
    node_id = get_id(info.subnode)
    var = JuMP.add_variable(
        model,
        JuMP.ScalarVariable(info.info),
        "n$(node_id)_" * name
    )
    ref = ""
    try
        element = name[findfirst(x -> x=='[', name):end]
        ref = "$(info.ref_symbol)" * element
    catch
        ref = "$(info.ref_symbol)"
    end
    if info isa InStateInfo
        set_input_variable!(info.subnode, ref, var)
    elseif info isa OutStateInfo
        set_output_variable!(info.subnode, ref, var)
    elseif info isa ControlInfo
        set_control_variable!(info.subnode, ref, var)
    end
    return var
end

function set_input_variable!(nd::SubTreeNode, ref::String, var::JuMP.VariableRef)
    nd.in[ref] = var
end

function set_output_variable!(nd::SubTreeNode, ref::String, var::JuMP.VariableRef)
    nd.out[ref] = var
end

function set_control_variable!(nd::SubTreeNode, ref::String, var::JuMP.VariableRef)
    nd.control[ref] = var
end

function set_stage_objective(nd::SubTreeNode, obj::Union{Float64, JuMP.AbstractJuMPScalar})
    nd.obj = obj
end
