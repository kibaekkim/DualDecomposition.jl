To run uc.jl, specify:

--DataDir           where power grid data is located e.g., path-to-examples/admm_examples/uc/data/UC_data/5BUS
--GraphDir          where tree.jld2 file is located. 
                    If tree.jld2 does not exist in GRAPHDIR, it generates one (but may take time).
--pRatio            Changes the scale of the load.