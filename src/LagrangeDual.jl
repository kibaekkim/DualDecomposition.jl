
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

    user_constraints
    user_args

    function LagrangeDual(T = BM.ProximalMethod, 
            maxiter::Int = 1000, tol::Float64 = 1e-6,
            user_constraints = nothing; kwargs...)
        LD = new{T}()
        LD.block_model = BlockModel()
        LD.var_to_index = Dict()
        LD.bundle_method = T
        LD.maxiter = maxiter
        LD.tol = tol

        LD.user_constraints = user_constraints
        LD.user_args = kwargs

        parallel.init()
        finalizer(finalize!, LD)
        
        return LD
    end
end

finalize!(LD::AbstractLagrangeDual) = parallel.finalize()

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
function run!(LD::AbstractLagrangeDual, optimizer)

    # We assume that the block models are distributed.
    num_all_blocks = parallel.sum(num_blocks(LD))
    num_all_coupling_variables = parallel.sum(num_coupling_variables(LD))

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
            JuMP.optimize!(m)
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
        bundle = LD.bundle_method(num_all_coupling_variables, num_all_blocks, solveLagrangeDual)
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
        LD.block_model.dual_bound = -BM.get_objective_value(bundle)
    
        # get dual solution
        LD.block_model.dual_solution = copy(BM.get_solution(bundle))

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
Wrappers of other functions
"""
objective_function(LD::AbstractLagrangeDual, block_id::Integer) = JuMP.objective_function(block_model(LD, block_id), QuadExpr).aff

index_of_λ(LD::AbstractLagrangeDual, var::CouplingVariableKey) = LD.var_to_index[var.block_id,var.coupling_id]
index_of_λ(LD::AbstractLagrangeDual, var::CouplingVariableRef) = index_of_λ(LD, var.key)
