"""
    AdmmMaster

Implementation of projected subgradient method
"""
mutable struct AdmmMaster <: AbstractLagrangeMaster
    num_vars::Int
    num_functions::Int
    eval_f::Union{Nothing,Function}

    iter::Int # current iteration count
    maxiter::Int
    obj_limit::Float64

    f::Float64
    best_f::Float64
    u::Vector{Float64}
    best_u::Vector{Float64}

    u_mean::Vector{Float64}
    v::Vector{Float64}
    v_old::Vector{Float64}
    λ::Vector{Float64}
    ρ::Float64
    τ::Float64
    ξ::Float64
    μ::Float64
    pres::Float64
    dres::Float64
    tol::Float64

    coupling_ids::Array{Array{Int}}
    constraint_matrix::Union{Nothing,SparseMatrixCSC{Float64,Int}}

    iteration_time::Vector{Float64}

    function AdmmMaster()
        am = new()
        am.num_vars = 0
        am.num_functions = 0
        am.eval_f = nothing
        am.iter = 0
        am.maxiter = 1000
        am.obj_limit = +Inf

        am.f = -Inf
        am.best_f = -Inf
        am.u = []
        am.best_u = []

        am.u_mean = []
        am.v = []
        am.v_old = []
        am.λ = []
        am.ρ = 1.0
        am.τ = 2.0
        am.ξ = 10.0
        am.μ = 1.0
        am.pres = 0.0
        am.dres = 1.0
        am.tol = 1e-8
        am.coupling_ids = []

        am.constraint_matrix = nothing
        am.iteration_time = []
        return am
    end
end

function load!(method::AdmmMaster, num_coupling_variables::Int, num_blocks::Int, eval_function::Function, init_sol::Vector{Float64})
    method.num_vars = num_coupling_variables
    method.num_functions = num_blocks
    method.eval_f = eval_function
    method.u = init_sol
    method.best_u = init_sol

end

function add_constraints!(LD::AbstractLagrangeDual, method::AdmmMaster)
    for (i, c) in enumerate(LD.coupling_id_keys)
        vars = LD.block_model.variables_by_couple[c]
        couplings = Array{Int}()
        for v in vars
            push!(couplings, index_of_λ(LD, v))
        end
        push!(method.coupling_ids, couplings)
    end
    method.u_mean = zeros(length(method.coupling_ids))
    update_u_mean!(method)
    method.v = zeros(method.num_vars)
    update_v!(method)
    method.λ = zeros(length(method.coupling_ids))
end

function update_u_mean!(method::AdmmMaster)
    fill!(method.u_mean, 0.0)
    for i in 1:length(method.coupling_ids)
        for idx in method.coupling_ids[i]
            method.u_mean[i] += method.u[idx]
        end
        method.u_mean[i] /= length(method.coupling_ids[i])
    end
end

function update_v!(method::AdmmMaster)
    copy!(method.v_old, method.v)
    for i in 1:length(method.coupling_ids)
        for idx in method.coupling_ids[i]
            method.v[idx] = method.u[idx] - method.u_mean[i]
        end
    end
end

function update_λ!(method::AdmmMaster)
    method.λ .+= method.ρ * method.u_mean
end

function update_residuals_and_ρ!(method::AdmmMaster)
    magu = sum(abs2,method.u)
    magv = sum(abs2,method.v)
    magl = sum(abs2,method.λ)
    method.pres = sqrt(sum(abs2,method.u_mean)/max(magu,magv))
    method.dres = sqrt(sum(abs2,method.ρ*(method.v-method.v_old))/magl)

    if (method.pres > method.ξ*method.μ*method.dres)
        method.ρ *= method.τ
    elseif (method.dres > method.μ/method.ξ*method.pres)
        method.ρ /= method.τ
    end
end

function run!(method::AdmmMaster)

    total_time = 0.0
    total_master_time = 0.0
    total_stime = time()

    while method.iter < method.maxiter
        method.iter += 1
        if method.iter % 100 == 1
            @printf("%6s", "Iter")
            @printf("\t%13s", "fbest")
            @printf("\t%13s", "f")
            @printf("\t%13s", "pres")
            @printf("\t%13s", "dres")
            @printf("\t%13s", "ρ")
            @printf("\t%7s", "am time")
            @printf("\t%8s", "tot time")
            @printf("\n")
        end
        f, method.u = method.eval_f(method.ρ, method.v, method.λ)
        method.f = -sum(f)
        if method.best_f < method.f
            method.best_f = method.f
            copy!(method.best_u, method.u)
        end

        master_stime = time()

        update_u_mean!(method)
        update_v!(method)
        update_λ!(method)
        update_residuals_and_ρ!(method)

        # clock iteration time
        push!(method.iteration_time, time() - master_stime)
        total_master_time += time() - master_stime
        total_time = time() - total_stime

        @printf("%6d", method.iter)
        @printf("\t%+6e", method.best_f)
        @printf("\t%+6e", method.f)
        @printf("\t%+6e", method.pres)
        @printf("\t%+6e", method.dres)
        @printf("\t%+6e", method.ρ)
        @printf("\t%7.2f", total_master_time)
        @printf("\t%8.2f", total_time)
        @printf("\n")

        if max(method.pres, method.dres) < method.tol
            break
        end
    end
end

get_objective(method::AdmmMaster) = method.best_f
get_solution(method::AdmmMaster) = method.best_u
get_times(method::AdmmMaster)::Vector{Float64} = method.iteration_time
function set_obj_limit!(method::AdmmMaster, val::Float64)
    method.obj_limit = val
end
