To run sslp.jl in parallel with 36 threads:
mpich -np 36 julia --project path-to-examples/admm_examples/sslp.jl

see parser.jl for descriptions on optional arguments

notes:
-When using proximal bundle method, set --numcut to the number of subproblems. 

