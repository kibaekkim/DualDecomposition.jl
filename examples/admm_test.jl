using JuMP
using CPLEX
using ADMM

function main(;admm_options...)
    admm = ADMM.AdmmAlg(;admm_options...)
    nS = 3
    Pr = ones(nS)/nS

    admm_addscenarios(admm, nS, Pr, create_scenario_model)
    admm_setnonantvars(admm, nonanticipativity_vars())
    admm_solve(admm, CplexSolver(CPX_PARAM_SCRIND=0))
end

function create_scenario_model(s::Integer)
    m = Model(solver=CplexSolver(CPX_PARAM_SCRIND=0))

    @variable(m, 0 <= x <= 4, Int)
    @variable(m, -3 <= y <= 2, Int)

    @constraint(m, -x + 3*y <= 9/2)
    @constraint(m, -2*x + y >= -8)
    @constraint(m, x + y <= 7/2)

    if s == 1
        @objective(m, Min, -x -2*y)
    elseif s == 2
        @objective(m, Min, -x + 3*y)
    else
        @objective(m, Min, -x + 1/2*y)
    end

    return m
end

nonanticipativity_vars() = [:x]

main(; mode=:SDM, rho=50, tmax=1)
