
abstract type AbstractLagrangeDual <: AbstractMethod end

"""
    LagrangeDual

Lagrangian dual method for dual decomposition. This `mutable struct` constains:
    - `block_model::BlockModel` object
    - `var_to_index` mapping coupling variable to the index wrt the master problem
    - `masiter::Int` sets the maximum number of iterations
    - `tol::Float64` sets the relative tolerance for termination
"""

mutable struct LagrangeDual{T<:BM.AbstractMethod} <: AbstractLagrangeDual
    block_model::BlockModel
    var_to_index::Dict{Tuple{Int,Any},Int} # maps coupling variable to the index wrt the master problem
    bundle_method
    maxiter::Int # maximum number of iterations
    tol::Float64 # convergence tolerance

    function LagrangeDual(T = BM.ProximalMethod, 
            maxiter::Int = 1000, tol::Float64 = 1e-6)
        LD = new{T}()
        LD.block_model = BlockModel()
        LD.var_to_index = Dict()
        LD.bundle_method = T
        LD.maxiter = maxiter
        LD.tol = tol
        
        return LD
    end
end

"""
Wrappers of the functions defined for `BlockModel`
"""

add_block_model!(LD::AbstractLagrangeDual, block_id::Integer, model::JuMP.Model) = add_block_model!(LD.block_model, block_id, model)
num_blocks(LD::AbstractLagrangeDual) = num_blocks(LD.block_model)
block_model(LD::AbstractLagrangeDual, block_id::Integer) = block_model(LD.block_model, block_id)
block_model(LD::AbstractLagrangeDual) = block_model(LD.block_model)
has_block_model(LD::AbstractLagrangeDual, block_id::Integer) = has_block_model(LD.block_model, block_id)
num_coupling_variables(LD::AbstractLagrangeDual) = num_coupling_variables(LD.block_model)
coupling_variables(LD::AbstractLagrangeDual) = coupling_variables(LD.block_model)

function set_coupling_variables!(LD::AbstractLagrangeDual, variables::Vector{CouplingVariableRef})
    set_coupling_variables!(LD.block_model, variables)
    variable_keys = [v.key for v in variables]
    # collect all coupling variables
    all_variable_keys = parallel.allcollect(variable_keys)
    set_variables_by_couple!(LD.block_model, all_variable_keys)
    LD.var_to_index = Dict((v.block_id,v.coupling_id) => i for (i,v) in enumerate(all_variable_keys))
end

dual_objective_value(LD::AbstractLagrangeDual) = dual_objective_value(LD.block_model)
dual_solution(LD::AbstractLagrangeDual) = dual_solution(LD.block_model)

"""
    run!(LD::AbstractLagrangeDual, optimizer)

This runs the Lagrangian dual method for solving the block model. `optimizer`
specifies the optimization solver used for `BundleMethod` package.
"""
function run!(LD::AbstractLagrangeDual, optimizer, bundle_init::Union{Nothing,Array{Float64}} = nothing)

    # We assume that the block models are distributed.
    num_all_blocks = parallel.sum(num_blocks(LD))
    num_all_coupling_variables = parallel.sum(num_coupling_variables(LD))

    # initialize bundle_init if it is nothing
    if isnothing(bundle_init)
        bundle_init = zeros(num_all_coupling_variables)
    end
    @assert length(bundle_init) == num_all_coupling_variables

    # check the validity of LagrangeDual
    if num_all_blocks <= 0 || num_all_coupling_variables == 0
        println("Invalid LagrangeDual structure.")
        return
    end

    function solveLagrangeDual(λ::Array{Float64,1})
        @assert length(λ) == num_all_coupling_variables

        # broadcast λ
        if parallel.is_root()
            parallel.bcast(λ)
        end

        # output
        objvals = Dict{Int,Float64}()
        subgrads = Dict{Int,SparseVector{Float64}}()

        # Adjust block objective function
        for var in coupling_variables(LD)
            adjust_objective_function!(LD, var, λ[index_of_λ(LD, var)])
        end

        for (id,m) in block_model(LD)
            # Initialize subgradients
            subgrads[id] = sparsevec(Dict{Int,Float64}(), length(λ))

            # Solver the Lagrange dual
            solve_sub_block!(m)

            @show id, JuMP.termination_status(m)
            @assert JuMP.termination_status(m) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]

            # We may want consider other statuses.
            if JuMP.termination_status(m) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
                objvals[id] = -JuMP.objective_value(m)
            end
        end

        # Get subgradients
        for var in coupling_variables(LD)
            # @assert has_block_model(LD, var.key.block_id)
            subgrads[var.key.block_id][index_of_λ(LD, var)] = -JuMP.value(var.ref)
        end

        # Reset objective coefficients
        for var in coupling_variables(LD)
            reset_objective_function!(LD, var, λ[index_of_λ(LD, var)])
        end

        # TODO: we may be able to add heuristic steps here.

        # Collect objvals, subgrads
        objvals_combined = parallel.combine_dict(objvals)
        objvals_vec = Vector{Float64}(undef, length(objvals_combined))
        if parallel.is_root()
            for (k,v) in objvals_combined
                objvals_vec[k] = v
            end
        end
        subgrads_combined = parallel.combine_dict(subgrads)

        return objvals_vec, subgrads_combined
    end

    if parallel.is_root()
        # Create bundle method instance
        bundle = LD.bundle_method(num_all_coupling_variables, num_all_blocks, solveLagrangeDual, bundle_init)
        BM.get_model(bundle).user_data = LD
    
        # Set optimizer to the JuMP model
        model = BM.get_jump_model(bundle)
        JuMP.set_optimizer(model, optimizer)
    
        # parameters for BundleMethod
        # bundle.M_g = max(500, dv.nvars + nmodels + 1)
        bundle.maxiter = LD.maxiter
        BM.set_bundle_tolerance!(bundle, LD.tol)
    
        # This builds the bunlde model.
        BM.build_bundle_model!(bundle)
    
        # Add bounding constraints to the Lagrangian master
        add_constraints!(LD, bundle)

        # This runs the bundle method.
        BM.run!(bundle)

        # get dual objective value
        get_objective!(LD, bundle)
    
        # get dual solution
        get_solution!(LD, bundle)

        # broadcast we are done.
        parallel.bcast(Float64[])
    else
        λ = parallel.bcast(nothing)
        while length(λ) > 0
            solveLagrangeDual(λ)
            λ = parallel.bcast(nothing)
        end
    end
end

"""
This adds the bounding constraints to the Lagrangian master problem.
"""
function add_constraints!(LD::AbstractLagrangeDual, method::BM.AbstractMethod)
    model = BM.get_jump_model(method)
    λ = model[:x]
    for (id, vars) in LD.block_model.variables_by_couple
        @constraint(model, sum(λ[index_of_λ(LD, v)] for v in vars) == 0)
    end
end

"""
This adjusts the objective function of each Lagrangian subproblem.
"""
function adjust_objective_function!(LD::AbstractLagrangeDual, var::CouplingVariableRef, λ::Float64)
    @assert has_block_model(LD, var.key.block_id)
    affobj = objective_function(LD, var.key.block_id)
    @assert typeof(affobj) == AffExpr
    coef = haskey(affobj.terms, var.ref) ? affobj.terms[var.ref] + λ : λ
    JuMP.set_objective_coefficient(block_model(LD, var.key.block_id), var.ref, coef)
end

"""
This resets the objective function of each Lagrangian subproblem.
"""
function reset_objective_function!(LD::AbstractLagrangeDual, var::CouplingVariableRef, λ::Float64)
    @assert has_block_model(LD, var.key.block_id)
    affobj = objective_function(LD, var.key.block_id)
    @assert typeof(affobj) == AffExpr
    coef = haskey(affobj.terms, var.ref) ? affobj.terms[var.ref] - λ : -λ
    JuMP.set_objective_coefficient(block_model(LD, var.key.block_id), var.ref, coef)
end

"""
This wraps the steps to optimize a block problem.
"""
function solve_sub_block!(model::JuMP.Model)
    JuMP.optimize!(model)
    reoptimize!(model)
end

"""
This re-optimizes block models if not solved to local optimality
"""
function reoptimize!(model::JuMP.Model)
    solve_itr = 0
    while !(JuMP.termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]) && solve_itr < 10
        JuMP.set_start_value.(all_variables(model), rand())
        JuMP.optimize!(model)
        solve_itr += 1
    end
end

"""
Wrappers of other functions
"""
objective_function(LD::AbstractLagrangeDual, block_id::Integer) = JuMP.objective_function(block_model(LD, block_id), QuadExpr).aff

index_of_λ(LD::AbstractLagrangeDual, var::CouplingVariableKey) = LD.var_to_index[var.block_id,var.coupling_id]
index_of_λ(LD::AbstractLagrangeDual, var::CouplingVariableRef) = index_of_λ(LD, var.key)


"""
These get dual objective and solutions
"""
function get_objective!(LD::AbstractLagrangeDual, method::BM.AbstractMethod)
    LD.block_model.dual_bound = -BM.get_objective_value(method)
end

function get_solution!(LD::AbstractLagrangeDual, method::BM.AbstractMethod)
    LD.block_model.dual_solution = copy(BM.get_solution(method))
end
