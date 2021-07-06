# DualDecomposition.jl
![Run tests](https://github.com/kibaekkim/DualDecomposition.jl/workflows/Run%20tests/badge.svg)
[![codecov](https://codecov.io/gh/kibaekkim/DualDecomposition.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kibaekkim/DualDecomposition.jl)
[![DOI](https://zenodo.org/badge/169820113.svg)](https://zenodo.org/badge/latestdoi/169820113)

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

## Citing this package

```
@misc{DualDecomposition.jl.0.3.0,
  author       = {Kim, Kibaek and Nakao, Hideaki and Kim, Youngdae and Schanen, Michel and Zhang, Weiqi},
  title        = {{DualDecomposition.jl: Parallel Dual Decomposition in Julia}},
  month        = Mar,
  year         = 2021,
  doi          = {10.5281/zenodo.4574769},
  version      = {0.3.0},
  publisher    = {Zenodo},
  url          = {https://doi.org/10.5281/zenodo.4574769}
}
```

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
