# LinearAlgebraUtils [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://shobhan126.github.io/LinearAlgebraUtils.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://shobhan126.github.io/LinearAlgebraUtils.jl/dev/) [![Build Status](https://github.com/shobhan126/LinearAlgebraUtils.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/shobhan126/LinearAlgebraUtils.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Introduction

Collection of some Linear Algebra related utilities that I end up using often for creating operators and tensor product basis states and so on.
Many of these were inspired the old and deprecated `QSimulator.jl` by Raytheon BBN Technologies and the work I was doing at Rigetti Computing.

The key things this currently exports:

- `linear_index` and `nzindex` for `TensoredBasisStates`
- `embed` to Embed Operators into larger Hilbert Spaces; for example if you want $CZ(1,2) \otimes I(3)$, you can quickly embed it. Should work for non-uniform tensor products as - so if you have qudits of dims (2,3,3,2,4); and you'd like to embed CZ(1, 4), I(remaining), this should work.
- `mapvectors` for matching dressed states to bare states

There are probably more well developed alternative libraries available to do these things...
