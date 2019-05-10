include("parallel.jl")
mutable struct LagrangeDualAlg <: AbstractAlg
    num_scenarios::Int64            # total number of scenarios
    probability::Dict{Int64,Float64}    # probabilities
    model::Dict{Int64,JuMP.Model}        # Dictionary of dual subproblems
    nonanticipativity_vars::Array{Symbol,1}
    num_nonant_vars::Int64
    nonant_indices::Array{Int64,1}
    master_algorithms::Dict{Symbol,Type}
    bestLB::Float64

    # Subproblem solution values
    colVals::Dict{Int64,Array{Float64,1}}

    # parameters
    maxiter::Integer    # maximum number of iterations
    maxtime::Float64    # time limit
    tol::Float64         # convergence tolerance

    function LagrangeDualAlg(n::Int64;
            maxiter=Int(1e+10), maxtime=1.e+10, tol=1.e-4, has_mpi_comm=false)
        algo = Dict(
            :ProximalBundle => BM.ProximalMethod,
            :ProximalDualBundle => BM.ProximalDualMethod
        )
        if !has_mpi_comm
            parallel.init()
            finalizer(LagrangeDualAlg->parallel.finalize(), LagrangeDualAlg)
        end
        parallel.partition(n)
        global LD = new(n, Dict(), Dict(), [], 0, [], algo, -Inf, Dict(),
            maxiter, maxtime, tol)
        return LD
    end
end

function add_scenario_models(LD::LagrangeDualAlg, ns::Integer, p::Vector{Float64},
                             create_scenario::Function)
    for s in parallel.getpartition()
        LD.probability[s] = p[s]
        LD.model[s] = create_scenario(s)
    end
end

function get_scenario_model(LD::LagrangeDualAlg, s::Integer)
    return LD.model[s]
end

function set_nonanticipativity_vars(LD::LagrangeDualAlg, names::Vector{Symbol})
    LD.nonanticipativity_vars = names
end

function solve(LD::LagrangeDualAlg, solver; master_alrogithm = :ProximalBundle)
    # check the validity of LagrangeDualAlg
    if LD.num_scenarios <= 0 || length(LD.nonanticipativity_vars) == 0
        println("Invalid LagrangeDual structure.")
        return
    end

    # Get some model to retrieve model information
    if length(LD.model) > 0
        some_model = collect(values(LD.model))[1]

        for v in LD.nonanticipativity_vars
            vi = getindex(some_model, v)

            # Get the dimension of nonanticipativity variables
            LD.num_nonant_vars += length(vi)

            # Get the indices for nonanticipativity variables
            for i in vi.innerArray
                push!(LD.nonant_indices, i.col)
            end
        end
    end

    # Number of variables in the bundle method
    nvars = LD.num_nonant_vars * LD.num_scenarios

    # Create bundle method instance
    bundle = BM.Model{LD.master_algorithms[master_alrogithm]}(nvars, LD.num_scenarios, solveLagrangeDual, true)

    # set the underlying solver Ipopt or PipsNlp
    bundle.solver = solver

    # parameters for BundleMethod
    # bundle.M_g = max(500, dv.nvars + nmodels + 1)
    bundle.maxiter = LD.maxiter
    bundle.maxtime = LD.maxtime
    bundle.ext.ϵ_s = LD.tol

    # Scale the objective coefficients by probability
    for (s,m) in LD.model
        affobj = m.obj.aff
        affobj.coeffs *= LD.probability[s]
    end

    # solve!
    BM.run(bundle)

    # Store the best known subproblem solutions
    for s in parallel.getpartition()
        LD.model[s].colVal = copy(LD.colVals[s])
    end
end

function solveLagrangeDual(λ::Array{Float64,1})
    # output
    sindices = Int64[]
    objvals = Float64[]
    subgrads = Array{Float64,2}(undef, 0, length(λ))

    for s in parallel.getpartition()
        # get the current model
        m = LD.model[s]

        # @show s
        # initialize results
        objval = 0.0
        subgrad = zeros(length(λ))

        # Get the affine part of objective function
        affobj = m.obj.aff

        # Change objective coefficients
        start_index = (s - 1) * LD.num_nonant_vars + 1
        for j in LD.nonant_indices
            var = Variable(m, j)
            if var in affobj.vars
                objind = findfirst(x->x==var, affobj.vars)
                affobj.coeffs[objind] += λ[start_index]
            else
                push!(affobj.vars, var)
                push!(affobj.coeffs, λ[start_index])
            end
            start_index += 1
        end

        # Solver the Lagrange dual
        status = JuMP.solve(m)

        if status == :Optimal
            objval = getobjectivevalue(m)
            for j in 1:LD.num_nonant_vars
                subgrad[(s - 1) * LD.num_nonant_vars + j] = getvalue(Variable(m, LD.nonant_indices[j]))
            end
        end

        # Add objective value and subgradient
        push!(sindices, s)
        push!(objvals, objval)
        subgrads = vcat(subgrads, subgrad')

        # Reset objective coefficients
        start_index = (s - 1) * LD.num_nonant_vars + 1
        for j in LD.nonant_indices
            var = Variable(m, j)
            objind = findfirst(x->x==var, affobj.vars)
            affobj.coeffs[objind] -= λ[start_index]
            start_index += 1
        end
    end
    objvals2=parallel.reduce(objvals)

    # Do we have a better solution found?
    if sum(objvals2) >= 1.0001 * LD.bestLB
        LD.bestLB = sum(objvals2)
        for s in parallel.getpartition()
            LD.colVals[s] = copy(LD.model[s].colVal)
        end
    end
    return -objvals2, -subgrads
end
