#=
Dual Decomposition of Stochastic Programming in Julia

Assume: two-stage stochastic programming
=#

module JuDD

using Compat
using JuMP
using BundleMethod
include("parallel.jl")

const BM = BundleMethod

type LagrangeDuals
	num_scenarios::Int64			# total number of scenarios
	probability::Dict{Int64,Float64}	# probabilities
	model::Dict{Int64,Any}		# Dictionary of dual subproblems
	nonanticipativity_vars::Array{Symbol,1}
	num_nonant_vars::Int64
	nonant_indices::Array{Int64,1}
	master_algorithms::Dict{Symbol,Type}
	function LagrangeDuals(n::Int64)
		algo = Dict(
			# :ProximalBundle => BM.ProximalMethod,
			:ProximalDualBundle => BM.ProximalDualModel
		)
		parallel.init()
		parallel.partition(n)
		finalizer(LagrangeDuals, LagrangeDuals->parallel.finalize())
		global LD = new(n, Dict(), Dict(), [], 0, [], algo)
		return
	end
end

function add_Lagrange_dual_model(s::Int64, p::Float64, create_model)
	if s in parallel.getpartition()
		LD.probability[s] = p
		LD.model[s] = create_model(s)
	end
end

function set_nonanticipativity_vars(vars::Array{Symbol,1})
	LD.nonanticipativity_vars = vars
end

function solve(solver; master_alrogithm = :ProximalBundle)
	# check the validity of LagrangeDuals
	if LD.num_scenarios <= 0 || length(LD.model) <= 0 || length(LD.nonanticipativity_vars) == 0
		println("Invalid LagrangeDual structure.")
		return
	end

	# Get some model to retrieve model information
	some_model = collect(values(LD.model))[1]

	for v in LD.nonanticipativity_vars
		vi = getvariable(some_model, v)

		# Get the dimension of nonanticipativity variables
		LD.num_nonant_vars += length(vi)

	    # Get the indices for nonanticipativity variables
	    for i in vi.innerArray
	        push!(LD.nonant_indices, i.col)
	    end
	end

	# Number of variables in the bundle method
	nvars = LD.num_nonant_vars * LD.num_scenarios

	# Create bundle method instance
	bundle = BM.ProximalDualModel(nvars, LD.num_scenarios, solveLagrangeDual, true)

	# set the underlying solver
	# JuMP.setsolver(bundle.m, solver)
	bundle.solver = solver

	# parameters for BundleMethod
	# bundle.M_g = max(500, dv.nvars + nmodels + 1)
	bundle.maxiter = 500

	# Scale the objective coefficients by probability
	for (s,m) in LD.model
		affobj = getobjective(m).aff
		affobj.coeffs *= LD.probability[s]
	end

	# solve!
	BM.run(bundle)

	# print solution
	@show BM.getobjectivevalue(bundle)
	@show BM.getsolution(bundle)
end

function solveLagrangeDual(λ::Array{Float64,1})
	# output
	sindices = Int64[]
	objvals = Float64[]
	subgrads = Array{Float64,2}(0, length(λ))

	for s in parallel.getpartition()
		# get the current model
		m = LD.model[s]

		# @show s
		# initialize results
		objval = 0.0
		subgrad = zeros(length(λ))

		# Get the affine part of objective function
		affobj = getobjective(m).aff

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
    return -objvals2, -subgrads	
end

end  # modJuDD

