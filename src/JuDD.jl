#=
Dual Decomposition of Stochastic Programming in Julia

Assume: two-stage stochastic programming
=#

module JuDD

using JuMP
using BundleMethod

const BM = BundleMethod

type LagrangeDuals
	num_scenarios::Int64			# total number of scenarios
	probability::Dict{Int64,Float64}	# probabilities
	model::Dict{Int64,JuMP.Model}		# Dictionary of dual subproblems
	nonanticipativity_vars::Array{Symbol,1}
	num_nonant_vars::Int64
	nonant_indices::Array{Int64,1}
	master_algorithms::Dict{Symbol,Type}
	function LagrangeDuals(n::Int64)
		algo = Dict(
			# :ProximalBundle => BM.ProximalMethod,
			:ProximalDualBundle => BM.ProximalDualModel
		)
		global LD = new(n, Dict(), Dict(), [], 0, [], algo)
		return
	end
end

function add_Lagrange_dual_model(s::Int64, p::Float64, model::JuMP.Model)
	LD.probability[s] = p
	LD.model[s] = model
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
	JuMP.setsolver(bundle.m, solver)

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
	objvals = zeros(LD.num_scenarios)
	subgrads = zeros(LD.num_scenarios, length(λ))

	for (s,m) in LD.model
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
		objvals[s] = objval
		subgrads[s,:] = subgrad

		# Reset objective coefficients
		start_index = (s - 1) * LD.num_nonant_vars + 1
		for j in LD.nonant_indices
			var = Variable(m, j)
			objind = findfirst(x->x==var, affobj.vars)
			affobj.coeffs[objind] -= λ[start_index]
			start_index += 1
		end
	end

    return -objvals, -subgrads
end

end  # modJuDD
