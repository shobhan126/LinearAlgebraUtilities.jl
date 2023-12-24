abstract type BasisState end

"""
A simple covenience wrapper for a tuple of subsystem dimensions for a tensor product space.
"""
struct TensoredBasis{N}
    dims::Tuple{Vararg{Int,N}}
end

dims(b::TensoredBasis) = b.dims

# helper contructor to convert from dimensions given in AbstractVector form
TensoredBasis(dims::AbstractVector{Int}) = TensoredBasis(Tuple(dims))

"""
A basis element of a tensor product space.
"""
struct TensoredBasisState{N} <: BasisState
    basis::TensoredBasis{N}
    states::Tuple{Vararg{Int,N}}
    function TensoredBasisState(b::TensoredBasis{N}, states::Tuple{Vararg{Int,N}}) where {N}
        # check that given subsystem basis states are compatible with TensorProductBasis dimensions
        @assert all(states .>= 0) && all(states .< dims(b))
        return new{N}(b, states)
    end
end

# helper contructor to convert from dimensions given in AbstractVector form
TensoredBasisState(b::TensoredBasis, states::AbstractVector{Int}) = TensoredBasisState(b, Tuple(states))

"""
    vec(state::TensorProductBasisState)

Create the complex basis vector correspoding to a given TensorProductBasisState.

## args
* `state`: a TensorProductBasisState to convert to a complex state vector.

## returns
The complex basis state vector.
"""
function Base.vec(state::TensoredBasisState, T::Type=ComplexF64)
    setindex!(zeros(T, prod(state.basis.dims)), one(T), nzindex(state.states, state.basis.dims))
end


"""
    basis_states(b::TensorProductBasis)

Enumerate all basis states in a tensor product space in the canonical order used by the Kronecker
product.

## args
* `b`: a TensorProductBasis.

## returns
A vector of TensorProductBasisState.

## example
`[bs.states for bs in basis_states(TensorProductBasis((2,2)))] == [(0,0), (0,1), (1,0), (1,1)]]`.
"""
function basis_states(b::TensoredBasis{N}) where {N}
    inds = permutedims(Tuple.(CartesianIndices(range.(0, b.dims .- 1))), reverse(1:N))
    return TensoredBasisState.((b,), inds)
end

"""
    index(bs::TensorProductBasisState)

Convert basis state of a tensor product space to an index into an enumerate of all basis states from
the canonical order used by the Kronecker product.

## args
* `bs`: a TensorProductBasisState.

## returns
The linear index of the state into the canonical tensor product basis states.

## example
Consider a tensor product of two qubits `b = TensorProductBasis((2,2))`. Then states `(0,0), (0,1),
(1,0), (1,1)` are numbered `1, 2, 3, 4` so that `index(TensorProductBasisState(b, (1,0)) == 3`.
"""
index(bs::TensoredBasisState) = nzindex(bs.states, bs.basis.dims)

"""
    getindex(b::TensorProductBasis, i::Int)

Convert an index into a tensor product space to the basis state. Note that
this function can be called with the notation `b[i]`.

## args
* `b`: the TensorProductBasis into which to index.
* `i`: the index of the basis state in the tensor product space.

## returns
The indexed TensorProductBasisState.

## example
Consider a tensor product of two qubits so that `b = TensorProductBasis((2,2))`. Then the basis
states `(0,0), (0,1), (1,0), (1,1)` are numbered `1, 2, 3, 4` so that
`b[3] == TensorProductBasisState(b, (1,0))`.
"""
Base.getindex(b::TensoredBasis, i::Int) = excitationvec(i, b.dims) |> Tuple
Base.getindex(b::TensoredBasis, i::AbstractVector{Int}) = TensoredBasisState.((b,), i)