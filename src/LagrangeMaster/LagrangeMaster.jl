abstract type AbstractLagrangeMaster end

function load!(method::AbstractLagrangeMaster, num_coupling_variables::Int, num_blocks::Int, eval_function::Function, init_sol::Vector{Float64}) end

function add_constraints!(LD::AbstractLagrangeDual, method::AbstractLagrangeMaster) end

function run!(method::AbstractLagrangeMaster) end

function get_objective!(LD::AbstractLagrangeDual, method::AbstractLagrangeMaster) end

function get_solution!(LD::AbstractLagrangeDual, method::AbstractLagrangeMaster) end

get_times(method::AbstractLagrangeMaster)::Vector{Float64} = zeros(1)