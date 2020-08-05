# DualDecomposition.jl
[![Build Status](https://travis-ci.com/kibaekkim/DualDecomposition.jl.svg?branch=master)](https://travis-ci.com/kibaekkim/DualDecomposition.jl)
[![codecov](https://codecov.io/gh/kibaekkim/DualDecomposition.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kibaekkim/DualDecomposition.jl)

This package implements an algorithmic framework for parallel dual decomposition methods in Julia.
While not aiming to outperforming the decomposition solvers written in a low-level 
language (e.g., [DSP](https://github.com/Argonne-National-Laboratory/DSP)), this package provides
the following features that `DSP` does not provide:

- This is designed to solve structured MINLP (and thus NLP) too, as long as the objective function is linear or quadratic.
- One can use any optimization solvers through `MathOptInterface.jl`.
- Of course, user does not need to compile any code for parallel solutions with `MPI.jl`.

## Installation

This package can be installed by cloning this repository:
```julia
] add DualDecomposition
```

## Examples

Please see examples in `./examples`.

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
