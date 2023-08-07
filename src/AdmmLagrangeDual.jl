using LinearAlgebra
using SparseArrays
"""
    AdmmLagrangeDual

ADMM Lagrangian dual method for dual decomposition. This `mutable struct` constains:
    - `block_model::BlockModel` object
    - `var_to_index` mapping coupling variable to the index wrt the master problem
"""

mutable struct AdmmLagrangeDual <: AbstractLagrangeDual
    constructor::DataType
    optimizer::Union{Nothing,DataType,MOI.OptimizerWithAttributes}
    params::BM.Parameters
    obj_limit::Float64

    block_model::BlockModel
    var_to_index::Dict{Tuple{Int,Any},Int} # maps coupling variable to the index wrt the master problem
    block_to_vars::Dict{Int,Vector{CouplingVariableRef}} # maps (local) block_id to list of vars
    coupling_id_keys::Array{Any}
    coupling_id_dict::Dict{Any,Int}

    bundlemethods::Dict{Int,BM.BasicMethod}
    # heuristics::Vector{Type}
    subsolve_time::Vector{Dict{Int,Float64}}
    subcomm_time::Vector{Float64}
    subobj_value::Vector{Float64}
    master_time::Vector{Float64}

    function AdmmLagrangeDual(constructor, optimizer, params = BM.Parameters())
        LD = new()
        LD.constructor = constructor
        LD.optimizer = optimizer
        LD.params = params
        LD.obj_limit = Inf
        
        LD.block_model = BlockModel()
        LD.var_to_index = Dict()
        LD.block_to_vars = Dict()
        LD.coupling_id_keys = []
        LD.coupling_id_dict = Dict()

        LD.bundlemethods = Dict()
        # LD.heuristics = []
        LD.subsolve_time = []
        LD.subcomm_time = []
        LD.subobj_value = []
        LD.master_time = []
        
        return LD
    end
end

function set_coupling_variables!(LD::AdmmLagrangeDual, variables::Vector{CouplingVariableRef})
    set_coupling_variables!(LD.block_model, variables)
    variable_keys = [v.key for v in variables]
    # collect all coupling variables
    all_variable_keys = parallel.allcollect(variable_keys)
    set_variables_by_couple!(LD.block_model, all_variable_keys)
    LD.var_to_index = Dict((v.block_id,v.coupling_id) => i for (i,v) in enumerate(all_variable_keys))
    
    for (id,m) in block_model(LD)
        LD.block_to_vars[id] = []
    end
    for var in variables
        push!(LD.block_to_vars[var.key.block_id], var)
    end
    LD.coupling_id_keys = collect(keys(LD.block_model.variables_by_couple))
    for (i,c) in enumerate(LD.coupling_id_keys)
        LD.coupling_id_dict[c] = i
    end
end

"""
    run!

This runs the Lagrangian dual method for solving the block model.
"""
function run!(LD::AdmmLagrangeDual, LM::AdmmMaster)

    # We assume that the block models are distributed.
    num_all_blocks = parallel.sum(num_blocks(LD))
    num_all_coupling_variables = parallel.sum(num_coupling_variables(LD))

    # check the validity of LagrangeDual
    if num_all_blocks <= 0 || num_all_coupling_variables == 0
        println("Invalid LagrangeDual structure.")
        return
    end


    for (id,block) in block_model(LD)
        function solve_subproblem(u::Array{Float64,1})
            @assert length(u) == length(LD.block_to_vars[id])
            for (i, var) in enumerate(LD.block_to_vars[id])
                adjust_objective_function!(LD, var, u[i])
            end
    
            # Solver the Lagrange dual
            m = block_model(LD,id)
            solve_sub_block!(m)
    
            status = JuMP.termination_status(m)
            # @assert status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
    
            # We may want consider other statuses.
            if status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
                objval = -JuMP.objective_value(m)
            else
                @error "Unexpected solution status: $(status)"
            end
    
            # Initialize subgradients
            subgrad = Dict{Int,SparseVector{Float64}}()
            subgrad[1] = sparsevec(Dict{Int,Float64}(), length(u))
            for (i, var) in enumerate(LD.block_to_vars[id])
                subgrad[1][i] = -JuMP.value(var.ref)
            end
            for (i, var) in enumerate(LD.block_to_vars[id])
                reset_objective_function!(LD, var, u[i])
            end
    
            return [objval], subgrad
        end
        num_coupling_variables = length(LD.block_to_vars[id])
        num_blocks = 1
        LD.bundlemethods[id] = LD.constructor(num_coupling_variables, num_blocks, solve_subproblem; init = zeros(num_coupling_variables), params = LD.params)
        # Set optimizer to bundle method
        JuMP.set_optimizer(BM.get_jump_model(LD.bundlemethods[id]), LD.optimizer)

        # This builds the bunlde model.
        BM.build_bundle_model!(LD.bundlemethods[id])

        # set objective limit
        BM.set_obj_limit(LD.bundlemethods[id], -LD.obj_limit)
    end

    function solveAdmmLagrangeDual(ρ::Float64, v:: Array{Float64,1}, λ::Array{Float64,1})
        @assert length(v) == num_all_coupling_variables
        @assert length(λ) == length(LD.coupling_id_keys)

        # broadcast λ
        if parallel.is_root()
            parallel.bcast((ρ,v,λ))
        end

        # output
        objvals = Dict{Int,Float64}()
        us = Dict{Int,SparseVector{Float64}}()
        subsolve_time = Dict{Int,Float64}()

        for (id, bm) in LD.bundlemethods
            n = length(LD.block_to_vars[id])
            P = ρ * sparse(Matrix(1.0I, n, n))
            q = zeros(n)
            for (i, var) in enumerate(LD.block_to_vars[id])
                idx = index_of_λ(LD, var)
                ci  = LD.coupling_id_dict[var.key.coupling_id]
                q[i] = λ[ci] - ρ * v[idx]
            end
            q = sparse(q)
            BM.update_objective!(bm, P, q)

            stime = time()
            BM.run!(bm)
            subsolve_time[id] = time() - stime

            objvals[id] = sum(bm.θ)
            newu = bm.y
            us[id] = sparsevec(Dict{Int,Float64}(), num_all_coupling_variables)
            for (i, var) in enumerate(LD.block_to_vars[id])
                idx = index_of_λ(LD, var)
                us[id][idx] = newu[i]
            end
        end
        push!(LD.subsolve_time, subsolve_time)


        parallel.barrier()
        comm_time = time()

        # Collect objvals, us
        objvals_combined = parallel.combine_dict(objvals)
        objvals_vec = Vector{Float64}(undef, length(objvals_combined))
        if parallel.is_root()
            for (k,v) in objvals_combined
                objvals_vec[k] = v
            end
            push!(LD.subobj_value, sum(objvals_vec))
        end

        us_combined = parallel.combine_dict(us)

        if parallel.is_root()
            push!(LD.subcomm_time, time() - comm_time)
            # @printf("Subproblem sommunication time: %6.1f sec.\n", time() - comm_time)
        end

        return objvals_vec, us_combined
    end

    if parallel.is_root()
        load!(LM, num_all_coupling_variables, num_all_blocks, solveAdmmLagrangeDual, zeros(num_all_coupling_variables))
    
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
        parallel.bcast((nothing, Float64[], Float64[]))
    else
        (ρ,v,λ) = parallel.bcast(nothing)
        while length(λ) > 0
            solveAdmmLagrangeDual(ρ, v, λ)
            (ρ,v,λ) = parallel.bcast(nothing)
        end
    end
end


function write_times(LD::AdmmLagrangeDual; dir = ".")
    write_file!(LD.subsolve_time, "subsolve_time", dir)
    write_file!(LD.subcomm_time, "subcomm_time.txt", dir)
    write_file!(LD.master_time, "master_time.txt", dir)
end

function write_all(LD::AdmmLagrangeDual; dir = ".")
    write_times(LD, dir = dir)
    write_file!(LD.subobj_value, "subobj_value.txt", dir)
end