"""
    LagrangeDual

Lagrangian dual method for dual decomposition. This `mutable struct` constains:
    - `block_model::BlockModel` object
    - `vref_to_index` mapping `JuMP.VariableRef` to index used in this method
    - `masiter::Int` sets the maximum number of iterations
    - `tol::Float64` sets the relative tolerance for termination
"""
mutable struct LagrangeDual{T<:BM.AbstractMethod} <: AbstractMethod
    block_model::BlockModel
    vref_to_index::Dict{JuMP.VariableRef,Int} # maps `JuMP.VariableRef` to index in decomposition method
    bundle_method
    maxiter::Int # maximum number of iterations
    tol::Float64 # convergence tolerance

    function LagrangeDual(T = BM.ProximalMethod, 
            maxiter::Int = 1000, tol::Float64 = 1e-6)
        LD = new{T}()
        LD.block_model = BlockModel()
        LD.vref_to_index = Dict()
        LD.bundle_method = T
        LD.maxiter = maxiter
        LD.tol = tol

        parallel.init()
        finalizer(finalize!, LD)
        
        return LD
    end
end

finalize!(LD::LagrangeDual) = parallel.finalize()

"""
Wrappers of the functions defined for `BlockModel`
"""

add_block_model!(LD::LagrangeDual, block_id::Integer, model::JuMP.Model) = add_block_model!(LD.block_model, block_id, model)
num_blocks(LD::LagrangeDual) = num_blocks(LD.block_model)
block_model(LD::LagrangeDual, block_id::Integer) = block_model(LD.block_model, block_id)
block_model(LD::LagrangeDual) = block_model(LD.block_model)
has_block_model(LD::LagrangeDual, block_id::Integer) = has_block_model(LD.block_model, block_id)
num_coupling_variables(LD::LagrangeDual) = num_coupling_variables(LD.block_model)
coupling_variables(LD::LagrangeDual) = coupling_variables(LD.block_model)

function set_coupling_variables!(LD::LagrangeDual, variables::Vector{CouplingVariableRef})
    # collect all coupling variables
    all_variables = parallel.allcollect(variables)
    set_coupling_variables!(LD.block_model, all_variables)
    LD.vref_to_index = Dict(v.ref => i for (i,v) in enumerate(all_variables))
end

dual_objective_value(LD::LagrangeDual) = dual_objective_value(LD.block_model)
dual_solution(LD::LagrangeDual) = dual_solution(LD.block_model)

"""
    run!(LD::LagrangeDual, optimizer)

This runs the Lagrangian dual method for solving the block model. `optimizer`
specifies the optimization solver used for `BundleMethod` package.
"""
function run!(LD::LagrangeDual, optimizer)
    # check the validity of LagrangeDual
    if num_blocks(LD) <= 0 || num_coupling_variables(LD) == 0
        println("Invalid LagrangeDual structure.")
        return
    end

    function solveLagrangeDual(λ::Array{Float64,1})
        @assert length(λ) == num_coupling_variables(LD)

        # broadcast λ
        if parallel.myid() == 0
            parallel.bcast(λ)
        end

        # output
        # objvals = Vector{Float64}(undef, num_blocks(LD))
        objvals = Dict{Int,Float64}()
        subgrads = Dict{Int,SparseVector{Float64}}()

        # Adjust block objective function
        for var in coupling_variables(LD)
            adjust_objective_function!(LD, var, λ[index_of_λ(LD, var.ref)])
        end

        for (id,m) in block_model(LD)
            @show parallel.myid(), id
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
            if has_block_model(LD, var.block_id)
                @show parallel.myid(), var.block_id
                subgrads[var.block_id][index_of_λ(LD, var.ref)] = -JuMP.value(var.ref)
            end
        end

        # Reset objective coefficients
        for var in coupling_variables(LD)
            reset_objective_function!(LD, var, λ[index_of_λ(LD, var.ref)])
        end

        # TODO: we may be able to add heuristic steps here.

        # Collect objvals, subgrads
        objvals_combined = parallel.combine_dict(objvals)
        objvals_vec = Vector{Float64}(undef, length(objvals_combined))
        if parallel.myid() == 0
            for (k,v) in objvals_combined
                objvals_vec[k] = v
            end
        end
        subgrads_combined = parallel.combine_dict(subgrads)

        return objvals_vec, subgrads_combined
    end

    # We assume that the block models are distributed.
    num_all_blocks = parallel.sum(num_blocks(LD))

    if parallel.myid() == 0
        # Create bundle method instance
        bundle = LD.bundle_method(num_coupling_variables(LD), num_all_blocks, solveLagrangeDual)
        BM.get_model(bundle).user_data = LD
    
        # Set optimizer to the JuMP model
        model = BM.get_jump_model(bundle)
        JuMP.set_optimizer(model, optimizer)
    
        # parameters for BundleMethod
        # bundle.M_g = max(500, dv.nvars + nmodels + 1)
        bundle.maxiter = LD.maxiter
        set_bundle_tolerance!(LD, bundle)
    
        # This builds the bunlde model.
        BM.build_bundle_model!(bundle)
    
        # Add bounding constraints to the Lagrangian master
        add_constraints!(LD, bundle)

        # This runs the bundle method.
        BM.run!(bundle)

        # broadcast we are done.
        parallel.bcast(Float64[])
    else
        λ = parallel.bcast(nothing)
        while length(λ) > 0
            solveLagrangeDual(λ)
            λ = parallel.bcast(nothing)
        end
    end

    # get dual objective value
    LD.block_model.dual_bound = -get_objective_value(bundle)

    # get dual solution
    LD.block_model.dual_solution = copy(get_solution(bundle))
end

"""
    Wrappers for BundleMethod functions
"""

get_objective_value(method::BM.ProximalMethod) = BM.getobjectivevalue(method)
get_objective_value(method::BM.TrustRegionMethod) = BM.get_objective_value(method)
get_solution(method::BM.ProximalMethod) = BM.getsolution(method)
get_solution(method::BM.TrustRegionMethod) = BM.get_solution(method)

function set_bundle_tolerance!(LD::LagrangeDual, method::BM.ProximalMethod)
    method.ϵ_s = LD.tol
end

function set_bundle_tolerance!(LD::LagrangeDual, method::BM.TrustRegionMethod)
    method.ϵ = LD.tol
end

"""
This adds the bounding constraints to the Lagrangian master problem.
"""
function add_constraints!(LD::LagrangeDual, method::BM.AbstractMethod)
    model = BM.get_jump_model(method)
    λ = model[:x]
    for (id, vars) in LD.block_model.variables_by_couple
        @constraint(model, sum(λ[index_of_λ(LD, v)] for v in vars) == 0)
    end
end

"""
This adjusts the objective function of each Lagrangian subproblem.
"""
function adjust_objective_function!(LD::LagrangeDual, var::CouplingVariableRef, λ::Float64)
    if has_block_model(LD, var.block_id)
        affobj = objective_function(LD, var.block_id)
        @assert typeof(affobj) == AffExpr
        coef = haskey(affobj.terms, var.ref) ? affobj.terms[var.ref] + λ : λ
        JuMP.set_objective_coefficient(block_model(LD, var.block_id), var.ref, coef)
    end
end


"""
This resets the objective function of each Lagrangian subproblem.
"""
function reset_objective_function!(LD::LagrangeDual, var::CouplingVariableRef, λ::Float64)
    if has_block_model(LD, var.block_id)
        affobj = objective_function(LD, var.block_id)
        @assert typeof(affobj) == AffExpr
        coef = haskey(affobj.terms, var.ref) ? affobj.terms[var.ref] - λ : -λ
        JuMP.set_objective_coefficient(block_model(LD, var.block_id), var.ref, coef)
    end
end

"""
Wrappers of other functions
"""
objective_function(LD::LagrangeDual, block_id::Integer) = JuMP.objective_function(block_model(LD, block_id), QuadExpr).aff

index_of_λ(LD::LagrangeDual, vref::JuMP.VariableRef)::Int = LD.vref_to_index[vref]
