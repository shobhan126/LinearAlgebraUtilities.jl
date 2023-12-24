using LinearAlgebra: I
using Base.Iterators: product, flatten

"""
    $(TYPEDEF)

    $(TYPEDFIELDS)
"""
struct EmbedMap
    """First column of the embedding Matrix"""
    firstcol::AbstractVecOrMat{<:Integer}
    """
    Strides for the embedding matrix. The first column is embedded at the first column of the larger matrix.
    """
    strides::AbstractVector{<:Integer}
end

"""
Embed a Matrix `M` acting on the given `subsystems` into a larger tensor product matrix.

    $(TYPEDSIGNATURES)
    
"""
function embed(m::AbstractMatrix{<:T}, subsystems::Base.AbstractVecOrTuple, sitedims::Base.AbstractVecOrTuple) where {T}
    @assert maximum(size(m, 1)) <= prod(sitedims[subsystems]) "Matrix being embedded is too large for the subsystem dimensions."
    D = prod(sitedims)
    M = zeros(T, (D, D))
    embed!(M, m, subsystems, sitedims)
    M
end

"""
In place version of `embed` that embeds a Matrix `m` onto the matrix M

    $(TYPEDSIGNATURES)
"""
function embed!(M, m, acting_on, sitedims)
    embed_map = embed_indices(acting_on, sitedims)
    @inbounds for i in eachindex(m)
        M[embed_map.firstcol[i].+embed_map.strides] .= m[i]
    end
end

"""
Inplace addition of a Matrix `m` of a subsystem into a larger system hilbert space Matrix `M`.

    $(TYPEDSIGNATURES)

    *Warning*: This function does perform any checks on matrix sizes. It is up to the user to add validations.
"""
function embed_add!(M::AbstractMatrix, m::AbstractMatrix, acting_on::AbstractVector, sitedims::AbstractVector)
    @inbounds for subindex in CartesianIndices(m)
        @inbounds M[global_indices(subindex, acting_on, sitedims)] .+= m[subindex]
    end
end


"""
Given an integer `subspace_index` and a vector of `acting_on` sites, return the global indices of 
Int -> Excitation Vector (1-based) -> Indices for a tensor product for a tensor product 
"""
function global_indices(subspace_index::Integer, acting_on::AbstractVector, sitedims::AbstractVector)
    # for each subindex find the corresponding excitation index
    subinds = excitationvec(subspace_index, sitedims[acting_on])
    ranges = UnitRange.(0, sitedims .- 1)
    ranges[acting_on] .= UnitRange.(subinds, subinds)
    Iterators.map(nzindex, Iterators.product(ranges...), Iterators.cycle((sitedims,)))
end

"""
Int -> Excitation Vector (1-based) -> Indices for a tensor product for a tensor product 
"""
function global_indices(subspace_index::Tuple{Integer,Integer}, acting_on::AbstractVector, sitedims::AbstractVector)
    z = Iterators.map(global_indices, subspace_index, Iterators.cycle((acting_on,)), Iterators.cycle((sitedims,)))
    Iterators.map(cartesian2linear, z..., Iterators.cycle(prod(sitedims)))
end

cartesian2linear(subspace_index::Tuple{Integer,Integer}, matrix_size::Integer) = cartesian2linear(subspace_index..., matrix_size)
cartesian2linear(col_index::Integer, row_index::Integer, matrix_size::Integer) = (row_index - 1) * matrix_size + col_index
global_indices(subspace_index::CartesianIndex{2}, acting_on::AbstractVector, sitedims::AbstractVector) = global_indices(subspace_index.I, acting_on, sitedims)

"""
Lazy embed mapping iterator
"""
function embed_indices_iter(acting_on::AbstractVector{<:Integer}, dims::AbstractVector{<:Integer})
    d = prod(dims[acting_on])
    global_indices.(Iterators.product(1:d, 1:d), (acting_on,), (dims,))
end


"""
Create an embed_mapping from a subsystem to a larger hilbert space.
"""
function embed_indices(acting_on::AbstractVector{<:Integer}, dims::AbstractVector{<:Integer})
    iter = embed_indices_iter(acting_on, dims)
    strides = iter |> first |> collect
    firstcol = first.(iter)
    # strides are referenced from first column so they're independent of previous values.
    strides .-= first(firstcol)
    EmbedMap(firstcol, strides)
end


"""
Add Matrix m to M, inplace, following an EmbedMap
"""
function embed_add!(M::AbstractMatrix, m::AbstractMatrix, embed_map::EmbedMap)
    for i in eachindex(m)
        M[embed_map.firstcol[i].+embed_map.strides] .+= m[i]
    end
    return nothing
end