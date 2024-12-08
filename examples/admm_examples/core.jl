using DualDecomposition
using Ipopt, GLPK, HiGHS

# Initialize MPI
parallel.init()

# Create DualDecomposition instance.
if masteralg == "bm"
    algo = DD.LagrangeDual()
elseif masteralg == "am"
    params = BM.Parameters()
    if !bundlelog
        BM.set_parameter(params, "print_output", false)
    end
    BM.set_parameter(params, "max_age", age)
    BM.set_parameter(params, "maxiter", maxsubiter)
    algo = DD.AdmmLagrangeDual(BM.BasicMethod, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0), params)
end

# partition scenarios into processes
parallel.partition(NS)

# Add Lagrange dual problem for each scenario s.
coupling_variables = Vector{DD.CouplingVariableRef}()
models = Dict{Int,JuMP.Model}()

for block_id in parallel.getpartition()
    model = create_sub_model!(block_id, coupling_variables)

    set_time_limit_sec(model, miptime)
    MOI.set(model, MOI.RelativeGapTolerance(), mipgap)
    if mipsolver == "cplex"
        set_optimizer(model, CPLEX.Optimizer)
        set_optimizer_attribute(model, "CPXPARAM_ScreenOutput", 0)
        set_optimizer_attribute(model, "CPX_PARAM_THREADS", 1)
        if miplog
            CPXsetlogfilename(backend(model).optimizer.model.env, "$dir/cpxlog_$block_id.txt", "a") #not safe
        else
            set_optimizer_attribute(model, "CPXPARAM_MIP_Display", 0)
        end
    elseif mipsolver == "glpk"
        set_optimizer(model, GLPK.Optimizer)
    elseif mipsolver == "highs"
        set_optimizer(model, HiGHS.Optimizer)
        if !miplog
            JuMP.set_silent(model)
        end
    end

    DD.add_block_model!(algo, block_id, model)
    models[block_id] = model
end

# Set nonanticipativity variables as an array of symbols.
DD.set_coupling_variables!(algo, coupling_variables)

# Lagrange master method
if masteralg == "bm"
    params = BM.Parameters()
    BM.set_parameter(params, "time_limit", timelim)
    BM.set_parameter(params, "ϵ_s", tol)
    BM.set_parameter(params, "max_age", age)
    BM.set_parameter(params, "maxiter", maxsubiter)
    BM.set_parameter(params, "u", proxu)
    BM.set_parameter(params, "ncuts_per_iter", numcut)
    LM = DD.BundleMaster(BM.ProximalMethod, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0), params)
elseif masteralg == "am"
    LM = DD.AdmmMaster(alg=amalg, ρ=rho, ϵ=tol, maxiter=100000, maxtime=timelim, update_interval = uinterval, τ=tau, μ=mu, ξ=xi)
end

DD.run!(algo, LM)

mkpath(dir)
DD.write_all(algo, dir=dir)
if masteralg == "am"
    DD.write_all(LM, dir=dir)
end

if (parallel.is_root())
  	@show DD.primal_objective_value(algo)
  	@show DD.dual_objective_value(algo)
  	@show DD.primal_solution(algo)
  	@show DD.dual_solution(algo)
end

# Finalize MPI
parallel.finalize()

