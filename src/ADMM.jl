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
0. Obtain an initial cut for the MP
REPEAT the following steps.
1. For given \lambda_j^k and z^k, solve the ALD to find (x_j^{k+1},y_j^{k+1}).
2. For given \lambda_j^k and (x_j^{k+1},y_j^{k+1}), solve the ALD w.r.t. z, which has
the closed-form solution:
  \sum_{j=1}^N [\rho (z^{k+1} - x_j^k) - \lambda_j^k] = 0;
  that is, z^{k+1} = \sum_{j=1}^N \lambda_j^k / (N \rho) + \sum_{j=1}^N x_j^k / N
3. Conditionally update lambda and update model
  a. Compute provisional \widetilde{\lambda}_j^{k+1} = \lambda_j^k - \rho (z^{k+1} - x_j^k)
  b. Solve Lagrange (column generation) subproblem with \lambda=\widetilde{\lambda}^{k+1}
     min p_j (c^T u_j + d_j^T v_j) + (\lambda_j^k + \rho( x_j^k - z^k ))^T u_j 
     subject to
     (u_j,v_j) \in G_j
     use the solution (u^*,v^*) to update the MP model, adding cuts and possibly aggregating previous cuts, 
     use the optimal value \widetilde{\phi} to test the serious step condition (SSC)
  c. If SSC passes, set \lambda^{k+1} = \widetilde{\lambda}^{k+1}, otherwise \lambda^{k+1} = \lambda^k
=#

abstract type AdmmMethod <: AbstractMethod end

mutable struct AdmmModelExt
	m::Dict{Int64,JuMP.Model}

	function AdmmModelExt(n::Int64, N::Int64)
		ext = new()
		ext.m = Dict()
		return ext
	end
end

const AdmmModel = Model{AdmmMethod}

function initialize!(bundle::AdmmModel)
	# Attach the extended structure
	bundle.ext = AdmmModelExt(bundle.n, bundle.N)

	for j = 1:bundle.N
		bundle.ext.m[j] = JuMP.Model()
	end
end

function add_initial_bundles!(bundle::AdmmModel)
end

function solve_bundle_model(bundle::AdmmModel)
	return :Optimal
end

function termination_test(bundle::AdmmModel)
	return true
end

function evaluate_functions!(bundle::AdmmModel)
end

function manage_bundles!(bundle::AdmmModel)
end

function update_iteration!(bundle::AdmmModel)
end

function display_info!(bundle::AdmmModel)
end
