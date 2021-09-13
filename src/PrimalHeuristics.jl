abstract type AbstractPrimalHeuristic end
abstract type RoundingHeuristic <: AbstractPrimalHeuristic end
abstract type AllBlockHeuristic <: AbstractPrimalHeuristic end

function add!(htype::Type{T}, LD::AbstractLagrangeDual) where T <: AbstractPrimalHeuristic
    push!(LD.heuristics, htype)
end

"""
    run!

This run a rounding heuristic.
"""
function run!(::Type{RoundingHeuristic}, LD::AbstractLagrangeDual, val, ub, lb)
    #set weights of each model if none exists
    num_all_blocks = parallel.sum(num_blocks(LD))
    all_block_ids = Set([ks[1] for (ks, i) in LD.var_to_index])
    if num_all_blocks != LD.block_model.combined_weights.count
        LD.block_model.combined_weights = Dict(block_id => 1/num_all_blocks for block_id in all_block_ids)
    end
    
    # get the mean value of the coupling variables at the root process and broadcast to other processes
    new_primal_solution = Dict() #maps coupling_id to value 
    if parallel.is_root()    
        for (ks, i) in LD.var_to_index
            coupling_id = ks[2]
            block_id = ks[1]
            new_primal_solution[coupling_id] = haskey(new_primal_solution, coupling_id) ? new_primal_solution[coupling_id] + LD.block_model.combined_weights[block_id] * val[block_id][i] : LD.block_model.combined_weights[block_id] * val[block_id][i]
        end         
        parallel.bcast(new_primal_solution)
    else
        new_primal_solution = parallel.bcast(nothing)
    end 

    #fix variables with new primal solution and enforce integrality through rounding
    for variables in LD.block_model.coupling_variables
        if JuMP.is_integer(variables.ref) || JuMP.is_binary(variables.ref)
            JuMP.fix(variables.ref, round(new_primal_solution[variables.key.coupling_id]), force=true)
            new_primal_solution[variables.key.coupling_id] = round(new_primal_solution[variables.key.coupling_id])
        else
            JuMP.fix(variables.ref, new_primal_solution[variables.key.coupling_id], force=true)
            new_primal_solution[variables.key.coupling_id] = new_primal_solution[variables.key.coupling_id]
        end
    end  

    #obtain primal bound by solving the subproblems in parallel
    cur_primal_bound = 0.0
    for (id,m) in block_model(LD)
        JuMP.optimize!(m)
        if ! (JuMP.termination_status(m)  in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
            cur_primal_bound = + Inf 
            break 
        else
            cur_primal_bound += JuMP.objective_value(m)
        end 
    end   
    cur_primal_bound_sum = parallel.sum(cur_primal_bound)

    #update primal bound and solution
    if cur_primal_bound_sum < LD.block_model.primal_bound
        LD.block_model.primal_bound = cur_primal_bound_sum
        LD.block_model.primal_solution = copy(new_primal_solution)
        if parallel.is_root()
            println("  Found new best primal bound: $(LD.block_model.primal_bound)")
        end 
    end 

    #unfix variables and recover their original bounds
    for var in coupling_variables(LD)
        JuMP.unfix(var.ref)
        JuMP.set_lower_bound(var.ref, lb[var.key.block_id][index_of_λ(LD, var)])
        JuMP.set_upper_bound(var.ref, ub[var.key.block_id][index_of_λ(LD, var)])
    end  
end

"""
    run!

This run a heuristic that simply fixes the coupling variable values.
"""
function run!(::Type{AllBlockHeuristic}, LD::AbstractLagrangeDual, val, ub, lb)
    all_block_ids = Set([ks[1] for (ks, i) in LD.var_to_index])
    all_coupling_ids = Set([ks[2] for (ks, i) in LD.var_to_index])
    
    # broadcast the val
    if parallel.is_root()
        parallel.bcast(val)
    else
        val = parallel.bcast(nothing)
    end 

    #iterate over all blocks. Fix the coupling variables to the optimal solution of each block.
    for block_id in all_block_ids
        new_primal_solution = Dict() #maps coupling_id to value 
        for coupling_id in all_coupling_ids
            new_primal_solution[coupling_id] = val[block_id][LD.var_to_index[block_id, coupling_id]]
        end       

        #fix variables with new primal solution and enforce integrality
        for variables in LD.block_model.coupling_variables
            JuMP.fix(variables.ref, new_primal_solution[variables.key.coupling_id], force=true)
        end 

        #obtain primal bound by solving the subproblems in parallel
        cur_primal_bound = 0.0
        for (id,m) in block_model(LD)
            JuMP.optimize!(m)
            if ! (JuMP.termination_status(m)  in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
                cur_primal_bound = + Inf 
                break 
            else
                cur_primal_bound += JuMP.objective_value(m)
            end 
        end   

        #update bounds and solution
        cur_primal_bound_sum = parallel.sum(cur_primal_bound)
        if cur_primal_bound_sum < LD.block_model.primal_bound
            LD.block_model.primal_bound = cur_primal_bound_sum
            LD.block_model.primal_solution = copy(new_primal_solution)

            if parallel.is_root()
                println("  Found new best primal bound: $(LD.block_model.primal_bound)")
            end 
        end 
    end 

    #unfix variables
    for var in coupling_variables(LD)
        JuMP.unfix(var.ref)
        JuMP.set_lower_bound(var.ref, lb[var.key.block_id][index_of_λ(LD, var)])
        JuMP.set_upper_bound(var.ref, ub[var.key.block_id][index_of_λ(LD, var)])
    end   
end
