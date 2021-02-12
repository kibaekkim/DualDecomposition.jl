mutable struct BundleMaster <: AbstractLagrangeMaster
    constructor
    optimizer
    inner

    function BundleMaster(constructor, optimizer)
        bm = new()
        bm.constructor = constructor
        bm.optimizer = optimizer
        bm.inner = nothing
        return bm
    end
end

function load!(method::BundleMaster, num_coupling_variables::Int, num_blocks::Int, eval_function::Function, init_sol::Vector{Float64})
    method.inner = method.constructor(num_coupling_variables, num_blocks, eval_function, init_sol)

    # Set optimizer to bundle method
    model = BM.get_jump_model(method.inner)
    JuMP.set_optimizer(model, method.optimizer)

    # This builds the bunlde model.
    BM.build_bundle_model!(method.inner)
end

function add_constraints!(LD::AbstractLagrangeDual, method::BundleMaster)
    model = BM.get_jump_model(method.inner)
    λ = model[:x]
    for (id, vars) in LD.block_model.variables_by_couple
        @constraint(model, sum(λ[index_of_λ(LD, v)] for v in vars) == 0)
    end
end

function run!(method::BundleMaster)
    BM.run!(method.inner)
end

function get_objective!(LD::AbstractLagrangeDual, method::BundleMaster)
    LD.block_model.dual_bound = -BM.get_objective_value(method.inner)
end

function get_solution!(LD::AbstractLagrangeDual, method::BundleMaster)
    LD.block_model.dual_solution = copy(BM.get_solution(method.inner))
end

get_times(method::BundleMaster)::Vector{Float64} = method.inner.model.time
