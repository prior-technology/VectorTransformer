"This is a sketch of a transformer implementation based on standard linear algebra operations."
module VectorTransformer

using LinearAlgebra
import NNlib

export ModelConfig, Pythia70ModelConfig, center

struct ModelConfig
    seq_len :: Integer 
    rotary_pct
    rotary_emb_base
    d_model
    n_heads
    d_head
end


function Pythia70ModelConfig()
    
    #from https://github.com/EleutherAI/pythia/blob/main/models/70M/pythia-70m-deduped.yml"
    seq_len = 2048
    #From https://github.com/EleutherAI/pythia/blob/main/models/70M/pythia-70m-deduped.yml"
    rotary_pct=0.25
    #From https://github.com/EleutherAI/gpt-neox/blob/v2.0/megatron/neox_arguments/neox_args.py#L286"
    rotary_emb_base = 10000
    #more pythia-70 sizes
    d_model=512
    n_heads=8
    d_head=d_model/n_heads
    return ModelConfig(seq_len, rotary_pct, rotary_emb_base, d_model, n_heads, d_head)

 
end

struct AttentionParams
    W_Qs :: Array{Matrix}
    W_Ks :: Array{Matrix}
    W_Vs :: Array{Matrix}
    W_Os :: Array{Matrix}
    bias_Q :: Vector
    bias_K :: Vector
    bias_V :: Vector
    bias_O :: Vector

end



function apply_transformer(transformer, residuals)
    #assume that residuals are already embedded
    for block in transformer.blocks
        residuals = residuals + apply_transformer_block(transformer.config, block, residuals)
    end
end

function batch_layer_normalize(residuals)
    for residual in residuals
        residual = LN(residual)
    end
end

function apply_transformer_block(config::ModelConfig, transformer_block , residuals)
    block_input = batch_layer_normalize(residuals)
   
    attention_out = attention(config, transformer_block.attention, block_input)
    mlp_in = batch_layer_normalize(attention_out)
    mlp_out = mlp(transformer_block.feedforward, mlp_in)
    return mlp_out

end

function attention(config::ModelConfig, AttentionParams, attention_in)
    attention_out=attention_in
    for h = 1:n_heads
        attention_out = attention_out + attention_head(config, AttentionParams, attention_in, h)
    end
    return attention_out
end

"based on Attention.forward in https://github.com/neelnanda-io/TransformerLens/blob/v1.14.0/transformer_lens/components.py"
function attention_head(config::ModelConfig, attention::AttentionParams, residuals, h)
    W_Q = attention.W_Qs[h]
    q = ( W_Q * residuals ) + attention.bias_Q
    W_K = attention.W_Ks[h]
    k = ( W_K * residuals ) + attention.bias_K
    W_V = attention.W_Vs[h]
    v = ( W_V * residual ) + attention.bias_V

    q = apply_rotary(config, q)
    k = apply_rotary(config, k)
    
    attention_matrix = attention_scores(q, k)
    pattern = NNlib.softmax(attention_matrix, dims=ndims(attention_matrix))
    #TODO: check dimensions used here, transformer lens reorders through einsum
    z = pattern * v
    W_O = attention.W_Os[h]
    O = z * W_O +  attention.bias_O
    return O
end


"Based on transformer lens AFAIR"
function attention_scores(q::AbstractMatrix, k::AbstractMatrix)
    #q and k should be num_positions x head_dims
    
    input_len = size(q,1)
    d_head = size(q,2)
    attention_scores = -1 ./ zeros(input_len,input_len)


    for pos = 1:input_len
        for pos2 = 1:pos
            q_vec = q[pos,:]
            k_vec = k[pos2,:]
            attention_scores[pos, pos2] = q_vec' * k_vec
        end
    end
    return attention_scores ./ sqrt(d_head) 
end


function inverse_frequencies(base::Integer, ndims::Integer)
    r = collect(range(0, step=2, stop=ndims))
    return 1 ./ (base .^ (r ./ ndims))
end

"based on RotaryEmbedding from https://github.com/EleutherAI/gpt-neox/blob/v2.0/megatron/model/positional_embeddings.py"
function frequencies(base, ndims, sequence_length)
    base_frequencies = collect(inverse_frequencies(base, ndims))
    position_array = collect(range(1,sequence_length))
    frequencies = base_frequencies * position_array'
end


function rotate_half(x)
    # Split the array along the last dimension
    middle_index = size(x, ndims(x)) ÷ 2  
    x1 = x[:, 1:middle_index]
    x2 = x[:, middle_index+1:end]

    # Concatenate along the last dimension with negation of the second part
    return cat(-x2, x1, dims=ndims(x))
end

"based on https://github.com/EleutherAI/gpt-neox/blob/v2.0/megatron/model/transformer.py"
function apply_rotary(config::ModelConfig, x) 
    # x should have 2 dimensions, num_positions, d_head 
    #Trying to stay as close to got-neox implementation as possible. When trying to implement this 
    #symbolically we may think of it as if there are two head-spaces, one gets rotated
    #and one doesn't, and they are summed after MLP matrix before LN

    d_head = size(x,2)
    rotary_ndims = floor(Integer, d_head * config.rotary_pct)
    x_rot = x[:, 1:rotary_ndims]
    x_pass = x[:, rotary_ndims+1:end]

    
    sequence_length = size(x, 1)

    f = frequencies(config.rotary_emb_base, rotary_ndims, sequence_length)
    sin_array = sin.(f)
    cos_array = cos.(f)
    x_rotated = (x_rot .* cos_array) + (rotate_half(x_rot) .* sin_array)

    return cat(x_rotated, x_pass, dims=ndims(x))

end

end