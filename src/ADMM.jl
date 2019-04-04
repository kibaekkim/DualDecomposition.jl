#=
ADMM method applied to the dual decomposition

Consider SMIP of the form
  min \sum_{j=1}^N p_j (c^T x_j + d_j^T y_j)
  subject to
  x_j = z for all j
  (x_j,y_j) \in G_j for all j.

The augmented Lagrange dual (ALD) of SMIP is given by
  min \sum_{j=1}^N [ p_j (c^T x_j + d_j^T y_j)
    - \lambda_j^T (z - x_j)
    + 0.5 \rho \|z - x_j\|^2 ]
  subject to (x_j,y_j) \in G_j for all j.

ADMM to solve the ALD is as follows:
REPEAT the following steps.
1. For given \lambda_j^k and z^k, solve the ALD to find (x_j^{k+1},y_j^{k+1}).
2. For given \lambda_j^k and (x_j^{k+1},y_j^{k+1}), solve the ALD w.r.t. z, which has
the closed-form solution:
  \sum_{j=1}^N [\rho (z^{k+1} - x_j^k) - \lambda_j^k] = 0;
  that is, z^{k+1} = \sum_{j=1}^N \lambda_j^k / (N \rho) + \sum_{j=1}^N x_j^k / N
3. \lambda_j^{k+1} = \lambda_j^k - \rho (z^{k+1} - x_j^k)

Frank-Wolfe algorithm for step 1:
REPEAT the following steps.
1. For given (x_j^k,z^k,\lambda_j^k), solve
  min p_j (c^T u_j + d_j^T v_j) + (\lambda_j^k)^T u_j - \rho (z^k - x_j^k)^T u_j
  subject to
  (u_j,v_j) \in G_j
2. Find a step size \gamma (should be in a closed form).
=#

module ADMM

using JuMP
using MathProgBase
using CPLEX

export AdmmAlg, admm_addscenario, admm_setnonantvars, admm_solve

type Scenario
    m::JuMP.Model               # scenario model
    prob::Float64               # probability
    id::Integer

    A::SparseMatrixCSC{Float64} # constraint matrix
    c::Vector{Float64}          # linear objective coefficients
    x::Vector{Float64}          # primal solution
    w::Vector{Float64}          # multiplier
    qr::Vector{Int32}           # row indices of quadratic objective
    qc::Vector{Int32}           # column indices of quadratic objective
    qv::Vector{Float64}         # value of coefficients
    scratch::Vector{Float64}    # scratch pad
    Vs::Vector{Vector{Float64}} # samples

    auglag::JuMP.Model          # Augmented Lagrangian model

    function Scenario(m::JuMP.Model, prob::Float64, id::Integer)
        return new(m, prob, id)
    end
end

type AdmmAlg
    scen::Dict{Integer, Scenario} # scenarios
    nonant_names::Vector{Symbol}  # symbols of non-anticipativity variables
    nonant_inds::Vector{Int32}    # flattened indices of non-ant variables
    nonant_len::Int32             # the number of non-ant variables
    z::Vector{Float64}            # auxiliary variable for ADMM

    # Parameters
    auglag_mode::Symbol # either :SDM or :MIQP
    rho::Float64        # penalty parameter of augmented Lagrangian
    kmax::Integer       # the maximum number of ADMM iterations
    tmax::Integer       # the maximum number of SDM iterations
    tol::Float64        # convergence tolerance
    alpha::Float64      # convex combination of xs and z

    function AdmmAlg(;mode=:MIQP, rho=1.0, kmax=1000, tmax=1, tol=1e-6, alpha=1.0)
	return new(Dict(), [], [], 0, [], mode, rho, kmax, tmax, tol, alpha)
    end
end

###########################################################################
# Internal functions
###########################################################################

function init_nonantvars(admm::AdmmAlg)
    # ---------------------------------------------------------------------
    # Find out the flattened indices of non-anticipative variables.
    # Assume that each scenario has the same non-anticipative variables.
    # ---------------------------------------------------------------------

    scen_model = collect(values(admm.scen))[1].m

    for name in admm.nonant_names
        inds = getvariable(scen_model, name)

        if isa(inds, Variable)
            admm.nonant_len += 1
            push!(admm.nonant_inds, inds.col)
        else
            admm.nonant_len += length(inds)
            innerArray = isa(inds, Array{Variable}) ? inds : inds.innerArray

            cols = []
            for i in innerArray
                push!(cols, i.col)
            end

            sort!(cols)
            append!(admm.nonant_inds, cols)
        end
    end

    return true
end

function update_z(admm::AdmmAlg)

    # ---------------------------------------------------------------------
    # Compute z = ∑ p(i)*xs(i) where p(i) and xs(i) are the probability and
    # the non-anticipative variable value of scenario i, respectively.
    # ---------------------------------------------------------------------

    fill!(admm.z, 0)
    for (key,scen) in admm.scen
        for (i,j) in enumerate(admm.nonant_inds)
            admm.z[i] += scen.prob*scen.x[j]
        end
    end
end

function update_w(admm::AdmmAlg)

    # ---------------------------------------------------------------------
    # For each scenario i, compute w(i) = w(i) + ρ*(xs(i) - z).
    # ---------------------------------------------------------------------

    for (key, scen) in admm.scen
        for (i,j) in enumerate(admm.nonant_inds)
            scen.w[i] += admm.rho*(scen.x[j] - admm.z[i])
        end
    end
end

function init_scenarios(admm::AdmmAlg, auglag_solver)
    admm.z = zeros(admm.nonant_len)

    # ---------------------------------------------------------------------
    # For each scenario, initialize data structure and generate an initial
    # point by solving the scenario model. Initial points between scenarios
    # are generated to share the same 1st stage variable value.
    # ---------------------------------------------------------------------

    for (key, scen) in admm.scen
        JuMP.build(scen.m)
        in_m = internalmodel(scen.m)

        MathProgBase.optimize!(in_m)
        stat = MathProgBase.status(in_m)

        if stat != :Optimal
            println("Initial solve of scenario ", key, " has failed: ", stat)
            return false
        end

        scen.x = copy(MathProgBase.getsolution(in_m))
        scen.A = copy(MathProgBase.getconstrmatrix(in_m))
        scen.c = copy(MathProgBase.getobj(in_m))

        scen.w = zeros(admm.nonant_len)
        scen.scratch = zeros(MathProgBase.numvar(scen.m))
        scen.Vs = []
        scen.qr = []
        scen.qc = []
        scen.qv = []
        scen.auglag = Model(solver=auglag_solver)

        if admm.auglag_mode == :SDM
            # For simplicial decomposition, add the solution to the sample set.
            add_sample(admm, scen, scen.x)
        else
            init_auglag_miqp(admm, scen)
        end
    end

    # ---------------------------------------------------------------------
    # Update z and w in accordance with the primal solution of scenarios.
    # ---------------------------------------------------------------------

    update_z(admm)
    update_w(admm)

    return true
end

function update_quadsdm(admm::AdmmAlg, scen::Scenario)
    # Assume that the last sample in Vs was the new one just added.
    s = scen.Vs[end]
    num_samples = length(scen.Vs)

    for i in 1:length(scen.Vs)
        vs = scen.Vs[i][admm.nonant_inds]
        ss = s[admm.nonant_inds]
        qval = (admm.rho/2)*dot(vs,ss)
        push!(scen.qr, i)
        push!(scen.qc, num_samples)
        push!(scen.qv, qval)
    end

    MathProgBase.setquadobjterms!(internalmodel(scen.auglag),
                                  scen.qr, scen.qc, scen.qv)
end

function update_linobjsdm(admm::AdmmAlg, scen::Scenario)
    scratch = zeros(length(scen.Vs))

    for (i,s) in enumerate(scen.Vs)
        xs = s[admm.nonant_inds]
        scratch[i] = dot(scen.c, s) + dot(scen.w, xs) - admm.rho*dot(admm.z, xs)
    end

    MathProgBase.setobj!(internalmodel(scen.auglag), scratch)
end

function init_auglag_sdm(admm::AdmmAlg, scen::Scenario, s::Vector{Float64})
    auglag = scen.auglag
    auglag.internalModel = MathProgBase.LinearQuadraticModel(auglag.solver)

    # 1.0 is for the convex combination constraint
    num_rows = scen.A.m + 1
    A = sparse(collect(1:num_rows), ones(num_rows), [scen.A*s; 1.0]) # [I J V]
    f = dot(scen.c, s)

    linconstr = scen.m.linconstr::Vector{LinearConstraint}
    num_rows = length(linconstr)
    rowlb = fill(-Inf, num_rows+1)
    rowub = fill(+Inf, num_rows+1)

    for i in 1:num_rows
        rowlb[i] = linconstr[i].lb
        rowub[i] = linconstr[i].ub
    end

    # Convex combination constraint
    rowlb[num_rows+1] = 1
    rowub[num_rows+1] = 1

    MathProgBase.loadproblem!(auglag.internalModel, A, [0], [1],
                              [f], rowlb, rowub, scen.m.objSense)

    # Set quadratic objective coefficients.
    push!(scen.Vs, copy(s))
    update_quadsdm(admm, scen)
    update_linobjsdm(admm, scen)
    auglag.internalModelLoaded = true
end

function add_sample(admm::AdmmAlg, scen::Scenario, s::Vector{Float64})
    if !scen.auglag.internalModelLoaded
        init_auglag_sdm(admm, scen, s)
        return
    end

    in_auglag = internalmodel(scen.auglag)
    xs = s[admm.nonant_inds]
    f = dot(scen.c, s) + dot(scen.w, xs) - admm.rho*dot(admm.z, xs)
    constr_coeffs = [ scen.A*s; 1.0 ] # 1.0 for the convex combination
    MathProgBase.addvar!(in_auglag, collect(1:MathProgBase.numconstr(scen.m)+1),
                         constr_coeffs, 0, 1, f)

    push!(scen.Vs, copy(s))
    update_quadsdm(admm, scen)
    update_linobjsdm(admm, scen)
end

function solve_sdm(admm::AdmmAlg, scen::Scenario)
    xs = (1 - admm.alpha)*admm.z + admm.alpha*scen.x[admm.nonant_inds]

    for t in 1:admm.tmax
        if t > 1
            xs = scen.x[admm.nonant_inds]
        end

        ws = scen.w .+ admm.rho*(xs .- admm.z)
        in_m = internalmodel(scen.m)

        # Update the objective coefficients of the linearized AugLag.
	for j in 1:length(scen.c)
	    scen.scratch[j] = scen.c[j]
	end

        for (i,j) in enumerate(admm.nonant_inds)
            scen.scratch[j] += ws[i]
        end

        MathProgBase.setobj!(in_m, scen.scratch)

        # Solve the FW problem.
        MathProgBase.optimize!(in_m)
        stat = MathProgBase.status(in_m)

        if stat != :Optimal
            println("FW has failed with stat ", stat)
            return stat
        end

        s = MathProgBase.getsolution(in_m)
        add_sample(admm, scen, s)

        # Solve the augmented Lagrangian.
        in_auglag = internalmodel(scen.auglag)
        MathProgBase.optimize!(in_auglag)
        stat = MathProgBase.status(in_auglag)

        if stat != :Optimal
            println("AugLag has failed with stat ", stat)
            return stat
        end

        beta = MathProgBase.getsolution(in_auglag)
        fill!(scen.x, 0)

        for (i,s) in enumerate(scen.Vs)
            for j in 1:length(scen.x)
                scen.x[j] += beta[i]*s[j]
            end
        end
    end

    return :Optimal
end

function init_auglag_miqp(admm::AdmmAlg, scen::Scenario)

    # ---------------------------------------------------------------------
    # Formulate the Augmented Lagrangian of scenario scen. It shares the
    # same linear constraints and variables as scen. The only difference
    # is its quadratic objective function.
    # ---------------------------------------------------------------------

    auglag = scen.auglag
    auglag.internalModel = MathProgBase.LinearQuadraticModel(auglag.solver)

    linconstr = scen.m.linconstr::Vector{LinearConstraint}
    num_rows = length(linconstr)
    rowlb = fill(-Inf, num_rows)
    rowub = fill(+Inf, num_rows)

    for i in 1:num_rows
        rowlb[i] = linconstr[i].lb
        rowub[i] = linconstr[i].ub
    end

    MathProgBase.loadproblem!(auglag.internalModel, scen.A, scen.m.colLower,
                              scen.m.colUpper, scen.c, rowlb, rowub,
                              scen.m.objSense)

    # ---------------------------------------------------------------------
    # Set quadratic objective coefficients Q. Q is a diagonal matrix with
    # entries being all equal to ρ/2.
    # ---------------------------------------------------------------------

    for j in admm.nonant_inds
        push!(scen.qr, j)
        push!(scen.qc, j)
        push!(scen.qv, admm.rho/2)
    end

    MathProgBase.setquadobjterms!(auglag.internalModel,
                                  scen.qr, scen.qc, scen.qv)
    auglag.internalModelLoaded = true
end

function update_linobjmiqp(admm::AdmmAlg, scen::Scenario)

    # ---------------------------------------------------------------------
    # Update the objective coefficients in accordance with the updates of
    # w and z.
    # ---------------------------------------------------------------------

    num_vars = MathProgBase.numvar(scen.m)
    for j in 1:num_vars
	scen.scratch[j] = scen.c[j]
    end

    for (i,j) in enumerate(admm.nonant_inds)
        scen.scratch[j] += scen.w[i] - admm.rho*admm.z[i]
    end

    MathProgBase.setobj!(internalmodel(scen.auglag), scen.scratch)
end

function print_iterlog(admm::AdmmAlg, k::Integer, err::Float64=Inf)
    if k % 50 == 0
        @printf("Iteration Log\n")
        @printf("%5s   %12s\n", "iter", "deviation")
    end

    @printf("%5d   %12.6e\n", k, (err==Inf) ? 0 : err)
end

function print_summary(admm::AdmmAlg, k::Integer, err::Float64)
    objval = 0

    for (key, scen) in admm.scen
        objval += scen.prob*dot(scen.x, scen.c)
    end

    @printf("Objective value: %10.6e", objval)
end

function solve_miqp(admm::AdmmAlg, scen::Scenario)
    update_linobjmiqp(admm, scen)

    in_auglag = internalmodel(scen.auglag)
    MathProgBase.optimize!(in_auglag)
    stat = MathProgBase.status(in_auglag)

    if stat != :Optimal
        println("stat is not optimal ", stat)
        return stat
    end

    scen.x = MathProgBase.getsolution(in_auglag)
    return stat
end

# -------------------------------------------------------------------------
# Exported functions
# -------------------------------------------------------------------------

function admm_addscenario(admm::AdmmAlg, s::Integer, p::Float64, m::JuMP.Model)
    admm.scen[s] = Scenario(m, p, s)
end

function admm_setnonantvars(admm::AdmmAlg, names::Vector{Symbol})
    admm.nonant_names = names
end

function admm_solve(admm::AdmmAlg, solver=CplexSolver())
    if length(admm.scen) <= 0 || length(admm.nonant_names) <= 0
        println("empty scenarios or empty non-anticipativity variables.")
        return
    end

    if !init_nonantvars(admm)
        println("initialization of non-anticipative variables failed.")
        return
    end

    if !init_scenarios(admm, solver)
        println("initialization of scenarios failed.")
        return
    end

    if admm.auglag_mode == :SDM
        solve_routine = solve_sdm
    elseif admm.auglag_mode == :MIQP
        solve_routine = solve_miqp
    else
        println("unknown Augmented Lagrangian solve method ", admm.auglag_mode)
        return
    end

    k = 0
    err = Inf
    print_iterlog(admm, k)

    while k < admm.kmax && err >= admm.tol
        for (key, scen) in admm.scen
           solve_routine(admm, scen)
        end

        # Check the convergence: sqrt(∑ p(s)|x(s)-z|^2) <= ϵ
        err = 0
        for (key, scen) in admm.scen
	    for i=1:length(admm.nonant_inds)
		err += scen.prob*(scen.x[admm.nonant_inds[i]] - admm.z[i])^2
	    end
        end
        err = sqrt(err)

        update_z(admm)
        update_w(admm)

        k += 1
        print_iterlog(admm, k, err)
    end

    print_summary(admm, k, err)
end

end # module ADMM


