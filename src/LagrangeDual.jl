"""
    LagrangeDual

Lagrangian dual method for dual decomposition. This `mutable struct` constains:
    - `block_model::BlockModel` object
    - `var_to_index` mapping coupling variable to the index wrt the master problem
    - `masiter::Int` sets the maximum number of iterations
    - `tol::Float64` sets the relative tolerance for termination
"""

mutable struct LagrangeDual <: AbstractLagrangeDual
    block_model::BlockModel
    var_to_index::Dict{Tuple{Int,Any},Int} # maps coupling variable to the index wrt the master problem
    subsolve_time::Vector{Dict{Int,Float64}}
    subcomm_time::Vector{Float64}
    subobj_value::Vector{Float64}
    master_time::Vector{Float64}

    function LagrangeDual()
        LD = new()
        LD.block_model = BlockModel()
        LD.var_to_index = Dict()
        LD.subsolve_time = []
        LD.subcomm_time = []
        LD.subobj_value = []
        LD.master_time = []
        
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
    run!

This runs the Lagrangian dual method for solving the block model.
"""
function run!(LD::AbstractLagrangeDual, LM::AbstractLagrangeMaster, initial_λ = nothing)

    # We assume that the block models are distributed.
    num_all_blocks = parallel.sum(num_blocks(LD))
    num_all_coupling_variables = parallel.sum(num_coupling_variables(LD))

    # initialize λ if it is nothing
    if isnothing(initial_λ)
        initial_λ = zeros(num_all_coupling_variables)
    end
    @assert length(initial_λ) == num_all_coupling_variables

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
        subsolve_time = Dict{Int,Float64}()

        # Adjust block objective function
        for var in coupling_variables(LD)
            adjust_objective_function!(LD, var, λ[index_of_λ(LD, var)])
        end

        for (id,m) in block_model(LD)
            # Initialize subgradients
            subgrads[id] = sparsevec(Dict{Int,Float64}(), length(λ))

            # Solver the Lagrange dual
            stime = time()
            solve_sub_block!(m)
            subsolve_time[id] = time() - stime

            @assert JuMP.termination_status(m) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]

            # We may want consider other statuses.
            if JuMP.termination_status(m) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
                objvals[id] = -JuMP.objective_value(m)
            end
        end
        push!(LD.subsolve_time, subsolve_time)

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

        parallel.barrier()
        comm_time = time()

        # Collect objvals, subgrads
        objvals_combined = parallel.combine_dict(objvals)
        objvals_vec = Vector{Float64}(undef, length(objvals_combined))
        if parallel.is_root()
            for (k,v) in objvals_combined
                objvals_vec[k] = v
            end
            push!(LD.subobj_value, sum(objvals_vec))
        end
        subgrads_combined = parallel.combine_dict(subgrads)

        if parallel.is_root()
            push!(LD.subcomm_time, time() - comm_time)
            # @printf("Subproblem sommunication time: %6.1f sec.\n", time() - comm_time)
        end

        return objvals_vec, subgrads_combined
    end

    if parallel.is_root()
        load!(LM, num_all_coupling_variables, num_all_blocks, solveLagrangeDual, initial_λ)
    
        # Add bounding constraints to the Lagrangian master
        add_constraints!(LD, LM)

        # This runs the bundle method.
        run!(LM)

        # Copy master solution time
        LD.master_time = get_times(LM)

        # get dual objective value
        LD.block_model.dual_bound = get_objective(LM)
    
        # get dual solution
        LD.block_model.dual_solution = get_solution(LM)

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

function write_times(LD::AbstractLagrangeDual; dir = ".")
    write_file!(LD.subsolve_time, "subsolve_time.txt", dir)
    write_file!(LD.master_time, "subcomm_time.txt", dir)
    write_file!(LD.master_time, "master_time.txt", dir)
end

function write_all(LD::AbstractLagrangeDual; dir = ".")
    write_times(LD, dir = dir)
    write_file!(LD.subobj_value, "subobj_value.txt", dir)
end
