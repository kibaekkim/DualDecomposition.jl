"""
    AdmmMaster

Implementation of projected subgradient method
"""
mutable struct AdmmMaster <: AbstractLagrangeMaster
    alg::Int
        #0: constant ρ
        #1: residual balancing
        #2: adaptive residual balancing
        #3: relaxed ADMM
        #4: adaptive relaxed ADMM
    num_vars::Int
    num_functions::Int
    eval_f::Union{Nothing,Function}

    iter::Int # current iteration count
    maxiter::Int
    maxtime::Float64
    obj_limit::Float64

    f::Float64
    final_f::Float64
    u::Vector{Float64}
    u_old::Vector{Float64}
    best_sol::Vector{Float64}
    best_res::Float64

    u_mean::Vector{Float64}
    v::Vector{Float64}
    v_old::Vector{Float64}
    λ::Vector{Float64}
    λ_old::Vector{Float64}
    ρ::Float64
    pres::Float64
    dres::Float64
    ϵ::Float64

    valid_step::Bool
    valid_step_prev::Bool

    #1: residual balancing
    τ::Float64
    ξ::Float64
    μ::Float64
    #2: adaptive residual balancing
    τmax::Float64
    #3: relaxed ADMM
    γ::Float64
    w::Vector{Float64}
    w_mean::Vector{Float64}
    #4: adaptive relaxed ADMM
    Tf::Int
    ϵ_cor::Float64
    C_cg::Float64
    u_k0::Vector{Float64}
    v_k0::Vector{Float64}
    λ_k0::Vector{Float64}
    λhat_k0::Vector{Float64}

    coupling_ids::Array{Array{Int}}
    constraint_matrix::Union{Nothing,SparseMatrixCSC{Float64,Int}}

    iteration_time::Vector{Float64}

    residual_primal_list::Vector{Float64}
    residual_dual_list::Vector{Float64}
    penalty_list::Vector{Float64}
    τ_list::Vector{Float64}
    γ_list::Vector{Float64}
    α_list::Vector{Float64}
    αcor_list::Vector{Float64}
    β_list::Vector{Float64}
    βcor_list::Vector{Float64}

    wallclock_time::Vector{Float64}

    function AdmmMaster(;alg=1, ρ=1.0, ϵ=1e-6, maxiter=1000, maxtime=3600.0)
        am = new()
        am.alg = alg
        am.num_vars = 0
        am.num_functions = 0
        am.eval_f = nothing
        am.iter = 0
        am.maxiter = maxiter
        am.maxtime = maxtime
        am.obj_limit = +Inf

        am.f = -Inf
        am.final_f = -Inf
        am.u = []
        am.u_old = []
        am.best_sol = []
        am.best_res = +Inf

        am.u_mean = []
        am.v = []
        am.v_old = []
        am.λ = []
        am.λ_old = []
        am.ρ = ρ
        am.pres = 0.0
        am.dres = 1.0
        am.ϵ = ϵ

        am.valid_step = false
        am.valid_step_prev = false
        
        #1: residual balancing
        am.τ = 2.0
        am.ξ = 10.0
        am.μ = 1.0
        #2: adaptive residual balancing
        am.τmax = 10.0
        #3: relaxed ADMM
        am.γ = 1.1
        am.w = []
        am.w_mean = []
        #4: adaptive relaxed ADMM
        am.Tf = 2
        am.ϵ_cor = 0.2
        am.C_cg = maxiter*maxiter
        am.u_k0 = []
        am.v_k0 = []
        am.λ_k0 = []
        am.λhat_k0 = []

        am.coupling_ids = []

        am.constraint_matrix = nothing
        am.iteration_time = []

        am.residual_primal_list = []
        am.residual_dual_list = []
        am.penalty_list = []
        am.τ_list = []
        am.γ_list = []
        am.α_list = []
        am.αcor_list = []
        am.β_list = []
        am.βcor_list = []

        am.wallclock_time = []
        
        return am
    end
end

function load!(method::AdmmMaster, num_coupling_variables::Int, num_blocks::Int, eval_function::Function, init_sol::Vector{Float64})
    method.num_vars = num_coupling_variables
    method.num_functions = num_blocks
    method.eval_f = eval_function
    method.u = init_sol
    method.best_sol = init_sol

end

function add_constraints!(LD::AbstractLagrangeDual, method::AdmmMaster)
    for (i, c) in enumerate(LD.coupling_id_keys)
        vars = LD.block_model.variables_by_couple[c]
        couplings = Array{Int}(undef,0)
        for v in vars
            push!(couplings, index_of_λ(LD, v))
        end
        push!(method.coupling_ids, couplings)
    end
    method.u_mean = zeros(length(method.coupling_ids))
    update_u_mean!(method, method.u)
    method.v = zeros(method.num_vars)
    update_v!(method, method.u, method.u_mean)
    method.λ = zeros(length(method.coupling_ids))

    if (method.alg == 3 || method.alg == 4)
        method.w = zeros(method.num_vars)
        method.w_mean = zeros(length(method.coupling_ids))
        update_w_mean!(method, method.w)
        method.u_k0 = zeros(method.num_vars)
        method.v_k0 = zeros(method.num_vars)
        method.λ_k0 = zeros(length(method.coupling_ids))
        method.λhat_k0 = zeros(method.num_vars)
    end
end

function update_u_mean!(method::AdmmMaster, u)
    fill!(method.u_mean, 0.0)
    for i in 1:length(method.coupling_ids)
        for idx in method.coupling_ids[i]
            method.u_mean[i] += u[idx]
        end
        method.u_mean[i] /= length(method.coupling_ids[i])
    end
end

function update_w_mean!(method::AdmmMaster, w)
    fill!(method.w_mean, 0.0)
    for i in 1:length(method.coupling_ids)
        for idx in method.coupling_ids[i]
            method.w_mean[i] += w[idx]
        end
        method.w_mean[i] /= length(method.coupling_ids[i])
    end
end

function update_v!(method::AdmmMaster, u, u_mean)
    copy!(method.v_old, method.v)
    for i in 1:length(method.coupling_ids)
        for idx in method.coupling_ids[i]
            method.v[idx] = u[idx] - u_mean[i]
        end
    end
end

function update_λ!(method::AdmmMaster, u_mean)
    copy!(method.λ_old, method.λ)
    method.λ .+= method.ρ * u_mean
end

function update_residuals!(method::AdmmMaster, u, v, v_old, λ)
    magu = sum(abs2,u)
    magv = sum(abs2,v)
    magl = 0.0
    for i in 1:length(method.coupling_ids)
        magl += length(method.coupling_ids[i]) * λ[i]^2
    end
    method.pres = sqrt(sum(abs2, u - v)/max(magu,magv))
    method.dres = method.ρ * sqrt(sum(abs2, v - v_old)/magl)
end

function relax_u!(method::AdmmMaster, u, v)
    for i in 1:length(method.coupling_ids)
        for idx in method.coupling_ids[i]
            method.w[idx] = method.γ * u[idx] + (1 - method.γ) * v[idx]
        end
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
            @printf("%8s", "status")
            @printf("\t%13s", "f")
            @printf("\t%13s", "best res")
            @printf("\t%13s", "pres")
            @printf("\t%13s", "dres")
            @printf("\t%13s", "ρ")
            @printf("\t%7s", "am time")
            @printf("\t%8s", "tot time")
            @printf("\n")
        end
        f, u_dict, status_dict = method.eval_f(method.ρ, method.v, method.λ, false)
        method.f = -sum(f)

        copy!(method.u_old, method.u)
        method.u = zeros(method.num_vars)
        for (id,vec) in u_dict
            method.u += vec
        end
        method.valid_step_prev = method.valid_step
        method.valid_step = true
        for (id, status) in status_dict
            if status != 1 # (1=optimal)
                method.valid_step = false
                break
            end
        end

        master_stime = time()

        #relax ρ
        #3: relaxed ADMM
        #4: adaptive relaxed ADMM
        if (method.alg == 3 || method.alg == 4)
            relax_u!(method, method.u, method.v)
            update_w_mean!(method, method.w)
            update_v!(method, method.w, method.w_mean)
            update_λ!(method, method.w_mean)
            update_residuals!(method, method.u, method.v, method.v_old, method.λ)
        else
            update_u_mean!(method, method.u)
            update_v!(method, method.u, method.u_mean)
            update_λ!(method, method.u_mean)
            update_residuals!(method, method.u, method.v, method.v_old, method.λ)
        end

        #update ρ
        #1: residual balancing
        #2: adaptive residual balancing
        if (method.alg == 1 || method.alg == 2)
            if (method.alg == 2)
                pres_raw = sqrt(sum(abs2, method.u - method.v))
                dres_raw = method.ρ*sqrt(sum(abs2, method.v - method.v_old))
                newτ = sqrt(pres_raw/dres_raw/method.ξ)
                if (newτ >= 1 && newτ < method.τmax)
                    method.τ = newτ
                elseif (newτ < 1 && newτ >= 1/method.τmax)
                    method.τ = 1/newτ
                end
            end
            if (method.pres > method.ξ*method.μ*method.dres)
                method.ρ *= method.τ
            elseif (method.dres > method.μ/method.ξ*method.pres)
                method.ρ /= method.τ
            end
        end
        #update ρ and γ
        #4: adaptive relaxed ADMM
        if (method.alg == 4 && method.iter % method.Tf == 1)
            λ_hat = zeros(Float64,method.num_vars)
            for i in 1:length(method.coupling_ids)
                for idx in method.coupling_ids[i]
                    λ_hat[idx] = method.λ_old[i] + method.ρ * (method.u[idx] - method.v_old[idx])
                end
            end

            Dh_hat = method.u_old - method.u_k0
            Dg_hat = method.v_old - method.v_k0
            Dλ     = zeros(Float64,method.num_vars)
            for i in 1:length(method.coupling_ids)
                for idx in method.coupling_ids[i]
                    Dλ[idx] = method.λ[i] - method.λ_k0[i]
                end
            end
            Dλ_hat =        λ_hat - method.λhat_k0

            DhhDhh = Dh_hat' * Dh_hat
            DghDgh = Dg_hat' * Dg_hat
            DλDλ   = Dλ'     * Dλ
            # DhhDλ  = Dh_hat' * Dλ
            DghDλ  = Dg_hat' * Dλ
            DλhDλh = Dλ_hat' * Dλ_hat
            DhhDλh = Dh_hat' * Dλ_hat
            # DghDλh = Dg_hat' * Dλ_hat

            α_SD = DλhDλh / DhhDλh
            α_MG = DhhDλh / DhhDhh
            β_SD = DλDλ   / DghDλ
            β_MG = DghDλ  / DghDgh

            if (2*α_MG > α_SD)
                α = α_MG
            else
                α = α_SD - α_MG/2
            end
            if (2*β_MG > β_SD)
                β = β_MG
            else
                β = β_SD - β_MG/2
            end

            α_cor = DhhDλh / sqrt(DhhDhh * DλhDλh)
            β_cor = DghDλ  / sqrt(DghDgh * DλDλ  )

            push!(method.α_list,α)
            push!(method.β_list,β)
            push!(method.αcor_list,α_cor)
            push!(method.βcor_list,β_cor)


            if α_cor > method.ϵ_cor && β_cor > method.ϵ_cor
                ρ_hat = sqrt(α * β)
                γ_hat = 1 + 2*sqrt(α * β)/(α + β)
            elseif α_cor > method.ϵ_cor && β_cor <= method.ϵ_cor
                ρ_hat = α
                γ_hat = 1.9
            elseif α_cor <= method.ϵ_cor && β_cor > method.ϵ_cor
                ρ_hat = β
                γ_hat = 1.1
            else
                ρ_hat = method.ρ
                γ_hat = 1.5
            end

            method.ρ = min(ρ_hat, (1+method.C_cg/method.iter^2)*method.ρ)
            method.γ = min(γ_hat, (1+method.C_cg/method.iter^2))
            copy!(method.u_k0, method.u_old)
            copy!(method.v_k0, method.v_old)
            copy!(method.λ_k0, method.λ_old)
            copy!(method.λhat_k0, λ_hat)
        end

        # clock iteration time
        push!(method.iteration_time, time() - master_stime)
        total_master_time += time() - master_stime
        total_time = time() - total_stime

        push!(method.residual_primal_list, method.pres)
        push!(method.residual_dual_list, method.dres)
        push!(method.penalty_list, method.ρ)
        if (method.alg == 2)
            push!(method.τ_list, method.τ)
        end
        if (method.alg == 4)
            push!(method.γ_list, method.γ)
        end
        push!(method.wallclock_time, total_time)

        if max(method.pres, method.dres) < method.best_res && method.valid_step_prev && method.valid_step
            method.best_res = max(method.pres, method.dres)
            copy!(method.best_sol, method.v)
        end

        @printf("%6d", method.iter)
        @printf("%8s", method.valid_step ? "valid" : "invalid")
        @printf("\t%+6e", method.f)
        @printf("\t%+6e", method.best_res)
        @printf("\t%+6e", method.pres)
        @printf("\t%+6e", method.dres)
        @printf("\t%+6e", method.ρ)
        @printf("\t%7.2f", total_master_time)
        @printf("\t%8.2f", total_time)
        @printf("\n")

        if max(method.pres, method.dres) < method.ϵ && method.valid_step_prev && method.valid_step
            break
        end
        if total_time > method.maxtime
            @printf("Time limit reached")
            break
        end
    end
    f, u_dict, status_dict = method.eval_f(method.ρ, method.best_sol, method.λ, true)
    method.final_f = -sum(f)
end

get_objective(method::AdmmMaster) = method.final_f
get_solution(method::AdmmMaster) = method.best_sol
get_times(method::AdmmMaster)::Vector{Float64} = method.iteration_time
function set_obj_limit!(method::AdmmMaster, val::Float64)
    method.obj_limit = val
end

function write_times(LM::AdmmMaster; dir = ".")
    write_file!(LM.wallclock_time, "wall_clock_time.txt", dir)
end

function write_all(LM::AdmmMaster; dir = ".")
    write_times(LM, dir = dir)
    write_file!(LM.residual_primal_list, "residual_primal.txt", dir)
    write_file!(LM.residual_dual_list, "residual_dual.txt", dir)
    write_file!(LM.penalty_list, "penalty.txt", dir)
    if (LM.alg == 2)
        write_file!(LM.τ_list, "penalty_multiplier.txt", dir)
    end
    if (LM.alg == 4)
        write_file!(LM.α_list, "alpha.txt", dir)
        write_file!(LM.β_list, "beta.txt", dir)
        write_file!(LM.αcor_list, "alpha_cor.txt", dir)
        write_file!(LM.βcor_list, "beta_cor.txt", dir)
        write_file!(LM.γ_list, "relaxation.txt", dir)
    end
end