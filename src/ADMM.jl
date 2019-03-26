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
using LinearAlgebra
using SparseArrays
using Printf

export
    admm_addscenario,
    admm_setnonantvars,
    admm_solve

mutable struct Scenario
    m::JuMP.Model               # scenario model
    prob::Float64               # probability

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

    function Scenario(m_in::JuMP.Model, prob_in::Float64)
        return new(m_in, prob_in)
    end
end

mutable struct AdmmAlg
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

    function AdmmAlg()
	return new(Dict(), [], [], 0, [], :MIQP, 1.0, 700, 1, 1e-6)
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

    scen_model = admm.scen[1].m

    for name in admm.nonant_names
        inds = try
            getindex(scen_model, name)
        catch
            println("error: symbol name ", name, " does not exist.")
            admm.nonant_len = 0
            admm.nonant_inds = []
            return false
        end

        admm.nonant_len += length(inds)

        for i in inds.innerArray
            push!(admm.nonant_inds, i.col)
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
        i = 1
        for j in admm.nonant_inds
            admm.z[i] += scen.prob*scen.x[j]
            i += 1
        end
    end
end

function update_w(admm::AdmmAlg)

    # ---------------------------------------------------------------------
    # For each scenario i, compute w(i) = w(i) + ρ*(xs(i) - z).
    # ---------------------------------------------------------------------

    for (key, scen) in admm.scen
        i = 1
        for j in admm.nonant_inds
            scen.w[i] += admm.rho*(scen.x[j] - admm.z[i])
            i += 1
        end
    end
end

function init_scenarios(admm::AdmmAlg, auglag_solver)
    admm.z = zeros(admm.nonant_len)

    # ---------------------------------------------------------------------
    # For each scenario, initialize data structure and generate an initial
    # point by solving the scenario model.
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

        scen.x = MathProgBase.getsolution(in_m)
        scen.A = MathProgBase.getconstrmatrix(in_m)
        scen.c = MathProgBase.getobj(in_m)
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

function update_quadsdm(admm::AdmmAlg, scen::Scenario, s::Vector{Float64})
    num_samples = length(scen.Vs)

    for (i,v) in enumerate(scen.Vs)
        qval = (admm.rho/2)*dot(v,s)
        push!(scen.qr, i)
        push!(scen.qc, num_samples+1)
        push!(scen.qv, qval)
    end

    push!(scen.qr, num_samples+1)
    push!(scen.qc, num_samples+1)
    push!(scen.qv, (admm.rho/2)*dot(s,s))

    MathProgBase.setquadobjterms!(internalmodel(scen.auglag),
                                  scen.qr, scen.qc, scen.qv)
end

function update_auglagobj_sdm(admm::AdmmAlg, scen::Scenario)
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
    A = sparse(collect(1:num_rows), ones(num_rows), [scen.A*s; 1.0], num_rows, 1)
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
    update_quadsdm(admm, scen, s)
    auglag.internalModelLoaded = true

    MathProgBase.optimize!(auglag.internalModel)
    stat = MathProgBase.status(auglag.internalModel)

    println("stat: ", stat)
    println(MathProgBase.getsolution(auglag.internalModel))
end

function add_sample(admm::AdmmAlg, scen::Scenario, s::Vector{Float64})
    if !scen.auglag.internalModelLoaded
        init_auglag_sdm(admm, scen, s)
        push!(scen.Vs, s)
        return
    end

    in_auglag = internalmodel(scen.auglag)
    xs = s[admm.nonant_inds]
    f = dot(scen.c, s) + dot(scen.w, xs) - admm.rho*dot(admm.z, xs)
    constr_coeffs = [ scen.A*s; 1.0 ] # 1.0 for the convex combination
    MathProgBase.addvar!(in_auglag, collect(1:MathProgBase.numconstr(scen.m)+1),
                         constr_coeffs, 0, 1, f)
    update_quadsdm(admm, scen, s)
    push!(scen.Vs, s)
end

function solve_sdm(admm::AdmmAlg, scen::Scenario)
    update_auglagobj_sdm(admm, scen)

    for t in 1:admm.tmax
        xs = scen.x[admm.nonant_inds]
        ws = scen.w + admm.rho*(xs - admm.z)
        in_m = internalmodel(scen.m)

        # Update the objective coefficients of the linearized AugLag.
        copyto!(scen.scratch, 1:length(scen.c), scen.c, 1:length(scen.c))

        i = 1
        for j in admm.nonant_inds
            scen.scratch[j] += ws[i]
            i += 1
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

        alpha = MathProgBase.getsolution(in_auglag)
        fill!(scen.x, 0)

        for (i,s) in enumerate(scen.Vs)
            for j in 1:length(scen.x)
                scen.x[j] += alpha[i]*s[j]
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

function update_auglagobj_miqp(admm::AdmmAlg, scen::Scenario)

    # ---------------------------------------------------------------------
    # Update the objective coefficients in accordance with the updates of
    # w and z.
    # ---------------------------------------------------------------------

    num_vars = MathProgBase.numvar(scen.m)
    copyto!(scen.scratch, 1:num_vars, scen.c, 1:num_vars)

    i = 1
    for j in admm.nonant_inds
        scen.scratch[j] += scen.w[i] - admm.rho*admm.z[i]
        i += 1
    end

    MathProgBase.setobj!(internalmodel(scen.auglag), scen.scratch)
end

function print_iterlog(admm::AdmmAlg, k::Integer, err::Float64=Inf)
    @printf("%5d\t%.6e\t%s\n", k, (err==Inf) ? 0 : err, string(admm.z))
end

function solve_miqp(admm::AdmmAlg, scen::Scenario)
    update_auglagobj_miqp(admm, scen)

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
    admm.scen[s] = Scenario(m, p)
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

    print_iterlog(admm, 0)

    err = Inf
    k = 1
    while k < admm.kmax && err >= admm.tol
        for (key, scen) in admm.scen
           solve_routine(admm, scen)
        end

        update_z(admm)

        # Check the convergence: sqrt(∑ p(s)|x(s)-z|^2) <= ϵ
        err = 0
        for (key, scen) in admm.scen
            i = 1
            for j in admm.nonant_inds
                err += scen.prob*((scen.x[j] - admm.z[i])^2)
                i += 1
            end
        end
        err = sqrt(err)

        update_w(admm)
        print_iterlog(admm, k, err)
        k += 1
    end
end

end


