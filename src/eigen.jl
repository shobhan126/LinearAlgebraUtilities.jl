using Base.Iterators: cycle
using Base: AbstractVecOrTuple
using SparseArrays
"""
Given a site, represent an excitation by excitation number `e`, and max possible excitation 

    $(SIGNATURES)

## Examples

```julia
julia> site(1, 2)
3-element SparseVector{Bool, Int64} with 1 stored entry:
  [2]  =  1
```
"""
site(e, emax) = sparsevec([e + 1], [true], emax)


"""
Create a sparse basis vector for a given device, `excitation` number on a given `siteindex.`

    $(SIGNATURES)

    This iterates through the device and constucts the max excitation number for each site;
        and then uses a kronecker product over sparse vectors to compute the basis vector.
"""
function basisvec(excitations, sites, sitedims)
    # create a properly padded excitations vector
    # check that all the excitates are less than the max excitation number
    # currently using a sparse vector but can replace it with an iterator
    spe = sparsevec(sites, excitations, length(sitedims))
    @assert all(spe .<= (sitedims .- 1))
    idx = nzindex(spe, sitedims)
    sparsevec([idx], [1], prod(sitedims))
end

"""
Given excitation and a site index, get a sparse basis vector.
"""
basisvec(excitation::Integer, site::Integer, sitedims::AbstractVector) = basisvec([excitation], [site], sitedims)
basisvec(excitations::AbstractVector, sitedims::AbstractVector) = sparsevec([nzindex(excitations, sitedims)], [1], prod(sitedims))


"""
Given the excitations and the sitedims, compute the index of the basis vector.

    $(SIGNATURES)
 
This should work identically over any Integer subtype; therefore one can pass UInts in cases where
the number of sites is large and we are worried about memory.
"""
function nzindex(excitations::Base.AbstractVecOrTuple{T}, sitedims::AbstractVecOrTuple{T}) where {T<:Integer}
    index = zero(T)
    for (e, d) in zip(excitations, sitedims)
        @assert 0 <= e < d "for each site, 0 <= excitation < sitedim"
        @fastmath index *= d
        @fastmath index += e
    end
    @fastmath index += one(T)
    return index
end
nzindex(excitations::CartesianIndex{N}, sitedims) where {N} = nzindex(excitations.I, sitedims)


function nzindex2(excitations::Base.AbstractVecOrTuple{T}, sitedims::AbstractVecOrTuple{T}) where {T<:Integer}
    index = zero(T)
    for (e, d) in zip(excitations, sitedims)
        @assert 0 < e <= d "for each site, 0 < excitation <= sitedim"
        index *= d
        index += (e - 1)
    end
    @fastmath index += one(T)
    return index
end


"""
Given the index of the basis vector find corresponding excitation vector.
This acts as an inverse of `nzindex`, for a fixed value of site dims.

    $(SIGNATURES)

## Details
A ket vector ⨂ᵢ|xᵢ⟩ᵢ in a tensor product space can be written as a column vector (0, 0, ..., 0, 1, ..., 0)ᵀ.
The `nzind` is the index of of the non-zero element in this column vector. 
Let `sitedims` be the dimension of of each hilbert space ℋᵢ, then $(FUNCTIONNAME)(nzind, sitedims) returns the corresponding xᵢ.
"""
function excitationvec(nzind::T, sitedims::AbstractVector{<:T}) where {T<:Integer}
    excitations = zero(sitedims)
    nzind -= 1
    @inbounds for i in Iterators.reverse(eachindex(sitedims))
        excitations[i] += nzind % sitedims[i]
        nzind ÷= sitedims[i]
    end
    return excitations
end

excitationvec(nzindex::T, sitedims::NTuple{N,T}) where {T<:Integer,N} = excitationvec(nzindex, collect(sitedims))

"""
Compute the overlap of 2 vectors x and y using Operator Norm

    $(SIGNATURES)

## Examples
```julia
julia> x = [1, 2, 2, 1] |> normalize
4-element Vector{Float64}:
 0.31622776601683794
 0.6324555320336759
 0.6324555320336759
 0.31622776601683794

julia> y = [1, 2, 2, 1] |> normalize
4-element Vector{Float64}:
 0.31622776601683794
 0.6324555320336759
 0.6324555320336759
 0.31622776601683794

julia> overlap(x, y)
1.0

julia> y = [1, -2, 2, -1] |> normalize
4-element Vector{Float64}:
  0.31622776601683794
 -0.6324555320336759
  0.6324555320336759
 -0.31622776601683794

julia> overlap(x, y)
0.0
```
"""
overlap(x::AbstractVecOrMat, y::AbstractVecOrMat) = abs.(x'y)  # l2 norm abs value

"""
Compute the overlap matrix of 2 sets of vectors x and y.

    $(SIGNATURES)

    Sometimes it is more convenient to have the vectors stored as Vector of Vectors instead 
    of a Matrix; for example during operations with sparse arrays. At such instants we can still
    return an overlap matrix.
"""
function overlap(x::T, y::U) where {T<:AbstractVector{<:AbstractVector},U<:AbstractVector{<:AbstractVector}}
    map(splat(overlap), Iterators.product(x, y))
end

mapvectors(X::AbstractMatrix, Y::AbstractMatrix) = mapvectors(eachcol(X), eachcol(Y))

"""
Establish an injective mapping between 2 sets of vectors X and Y such that the overlap is maximized.

    $(SIGNATURES)
    
    Returns a vector of indices y.
    If the initial map is surjective, i.e. the indices of y are not unique,
    then a tiebreak heuristic is used by invoking `breakties!`. breakties performance can be improved vastly.
    Currently this should be a good heuristic assuming tiebreaking is not required too often.
"""
function mapvectors(X::AbstractVector{<:AbstractVector}, Y::AbstractVector{<:AbstractVector})
    nx, ny = length(X), length(Y)
    if ny < nx
        throw(DimensionMismatch("For computing the injective mapping, length(Y)>=length(X)"))
    else
        return mapvectors(X, Y, 1:nx, 1:ny)
    end
end


"""
Establish an injective mapping between 2 sets of vectors X and Y such that the overlap is maximized. 

The returned mapping is a vector mapping `indsX` to `indsY`

    $(SIGNATURES)
"""
function mapvectors(X::AbstractVector{<:AbstractVector}, Y::AbstractVector{<:AbstractVector}, indsX, indsY)
    mapping = [
        # Iterate through each element in Y, and find the overlap; if it's larger than the previous item
        # then keep that one, else keep the previous one.
        @inbounds mapreduce(i -> (i, overlap(X[ix], Y[i])), (a, b) -> (last(a) > last(b) ? a : b), indsY) |> first
        for ix in indsX
    ]
    breakties!(mapping, X, Y)
end


"""
Given a surjective mapping, from X to Y, break using the second largest overlaps from X to Y.

    $(SIGNATURES)

This heuristic should well for most use cases we have
"""
function breakties!(mapping, X, Y, recursion_depth=0; max_recursion_depth=10)
    # todo: set max recursion depth?
    # todo: allow passing indices so we can close in on the solution faster
    if !allunique(mapping)
        # # find the indices of the non-unique mappings
        grouped_indices = group_indices(mapping)
        ny = length(Y)
        for (iy, iX) in grouped_indices
            if length(iX) > 1
                # returns [ix] such that overlap(X[ix], Y[iy]) is maximum for ix ∈ iX
                iiX = mapvectors(Y, X, [iy], iX) |> first
                popat!(iX, findfirst(==(iiX), iX))
                # now we find the second best match for the remaining ones
                except_iy = Iterators.flatten((1:iy-1, iy+1:ny))
                iX_second_choices = mapvectors(X, Y, iX, except_iy)
                # next we set the indices
                setindex!(mapping, iX_second_choices, iX)
            end
        end
        if recursion_depth < max_recursion_depth
            breakties!(mapping, X, Y, recursion_depth + 1)
        else
            # warn user 
            # TODO: also add the vectors which are causing trouble
            @warn "Maximum recursion depth reached.\n
            Returning the current mapping. Please check for degeneracies"
            return mapping
        end
    else
        return mapping
    end
end


function group_indices(indices::AbstractVector{<:T}) where {T}
    result = Dict{T,Vector{<:Integer}}()
    for (i, value) in enumerate(indices)
        if haskey(result, value)
            push!(result[value], i)
        else
            result[value] = [i]
        end
    end
    return result
end