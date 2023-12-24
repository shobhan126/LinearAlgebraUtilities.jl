using Test
using LinearAlgebraUtils
# @testset "basis utils" begin
    # check that constructors from AbstractVector work
    basis = TensoredBasis((2, 2))
    @test basis == TensoredBasis([2, 2])
    @test TensoredBasisState(basis, (0, 0)) == TensoredBasisState(basis, [0, 0])

    # simple tensor product expansion
    basis = TensoredBasis((2, 2))
    @test [bs.states for bs in basis_states(basis)[:]] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    basis = TensoredBasis((2, 3))
    @test [bs.states for bs in basis_states(basis)[:]] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    # go from state -> index
    basis = TensoredBasis((2, 3, 4, 5))
    all_states = basis_states(basis)
    @test all(index(state) == i for (i, state) in enumerate(all_states))

    # and from index to state
    @test all(basis[i] == state for (i, state) in enumerate(all_states))

    # check that TensoredBasisState construction catches errors
    basis = TensoredBasis((2, 2))
    # wrong number of subsystems
    @test_throws MethodError TensoredBasisState(basis, (0, 0, 0))
    # state index exceeds dimension
    @test_throws AssertionError TensoredBasisState(basis, (2, 0))
# end
