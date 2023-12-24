module LinearAlgebraUtils

# Write your package code here.

using DocStringExtensions

include("basis.jl")
include("state_utils.jl")
include("embed.jl")

export site, basisvec, nzindex, excitationvec, mapvectors, overlap, linearindex
export TensoredBasis, TensoredBasisState, BasisState, basis_states, dims
export embed_add!, embed, embed_indices, embed!

end
