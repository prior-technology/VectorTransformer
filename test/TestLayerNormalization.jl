
include("../data/probe_token.jl")
include("../data/pre_norm.jl")

@testset "SymbolicTransformer.jl" begin
    bias = 0.8328

    final_residual = LN(pre_norm)

    logit = sum(.*(probe_token, final_residual)) + bias
    # should return  11.4077
    @test logit ≈ 11.4077 atol=1e-3

    
end
