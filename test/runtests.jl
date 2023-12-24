using LinearAlgebraUtils
using Test
using Aqua

@testset "LinearAlgebraUtils.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(LinearAlgebraUtils)
    end
    # Write your tests here.
end
