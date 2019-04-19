# JuDD.jl
Dual Decomposition of Stochastic Programming in Julia

This package can be installed by cloning this repository:
```julia
] add 
] using Pkg
] Pkg.add(PackageSpec(url="https://github.com/kibaekkim/JuDD.jl", rev="mpi"))
```

This package requires to install `BundleMethod.jl`:
```julia
] using Pkg
] Pkg.add(PackageSpec(url="https://github.com/kibaekkim/BundleMethod.jl", rev="structjump"))
```

# Using StructJuMP and PIPS

```
] using Pkg
] Pkg.add(PackageSpec(url="https://github.com/Argonne-National-Laboratory/StructJuMPSolverInterface.jl.git", rev="duals"))
] Pkg.add(PackageSpec(url="https://github.com/Argonne-National-Laboratory/StructJuMP.jl.git", rev="julia0.7"))
```
