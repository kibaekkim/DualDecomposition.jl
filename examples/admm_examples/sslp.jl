using DualDecomposition
using ArgParse
using JuMP
using Random

const DD = DualDecomposition
const parallel = DD.parallel

settings = ArgParseSettings()
@add_arg_table settings begin
    "--nJ"
        help = "number of servers"
        arg_type = Int
        default = 10
    "--nI"
        help = "number of clients"
        arg_type = Int
        default = 50
    "--nS"
        help = "number of scenarios"
        arg_type = Int
        default = 50
end

include("parser.jl")

nJ = parsed_args["nJ"]
nI = parsed_args["nI"]
nS = parsed_args["nS"]
NS = nS

seed::Int = 1
Random.seed!(seed)

sJ = 1:nJ
sI = 1:nI
sS = 1:nS

c = rand(40:80,nJ)
q = rand(0:25,nI,nJ,nS)
q0 = ones(nJ)*1000
d = q
u = 1.5*sum(d)/nJ
v = nJ
h = rand(0:1,nI,nS)
Pr = ones(nS)/nS


# This creates a Lagrange dual problem for each scenario s and adds coupling variables.
function create_sub_model!(s::Int64, coupling_variables::Vector{DD.CouplingVariableRef})
    model = Model()

    @variable(model, x[j=sJ], Bin)
    @variable(model, y[i=sI,j=sJ], Bin)
    @variable(model, y0[j=sJ] >= 0)

    @objective(model, Min, (
            sum(c[j]*x[j] for j in sJ)
        - sum(q[i,j,s]*y[i,j] for i in sI for j in sJ)
        + sum(q0[j]*y0[j] for j in sJ)))

    @constraint(model, sum(x[j] for j in sJ) <= v)
    @constraint(model, [j=sJ], sum(d[i,j,s]*y[i,j] for i in sI) - y0[j] <= u*x[j])
    @constraint(model, [i=sI], sum(y[i,j] for j in sJ) == h[i,s])

    xref = model[:x]
    for i in sJ
        push!(coupling_variables, DD.CouplingVariableRef(s, i, xref[i]))
    end

    return model
end

include("core.jl")



