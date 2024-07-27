using VectorTransformer
using Test


@testset "attention_scores" begin
    include("../data/attention_inputs.jl")
    a = VectorTransformer.attention_scores(q, k)
    @test a ≈ expected_attention
end