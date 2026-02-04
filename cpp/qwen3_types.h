#pragma once

#include <stdint.h>
#include <vector>
#include <map>

// Common configurations based on the Python analysis

struct Qwen3TalkerConfig {
    int vocab_size = 3072; // Default from config analysis
    int hidden_size = 1024;
    int num_hidden_layers = 20;
    int num_attention_heads = 16;
    int num_key_value_heads = 2;
    int max_position_embeddings = 32768;
    int num_code_groups = 32; // Important for the code predictor
    
    // Special Tokens
    int tts_bos_token_id = 151672;
    int tts_eos_token_id = 151673;
    int tts_pad_token_id = 151671;
};

struct Qwen3AudioDecoderConfig {
    // Quantizer
    int codebook_dim = 256; 
    int num_quantizers = 8;
    int codebook_size = 2048;

    // Transformer (Pre-Net)
    int latent_dim = 1024; // Input to transformer
    int hidden_size = 1024;
    int num_hidden_layers = 8;
    int num_attention_heads = 16;
    int num_key_value_heads = 16;
    int max_position_embeddings = 8000;
    int intermediate_size = 3072;
    float rms_norm_eps = 1e-5f;
    float layer_scale_initial_scale = 0.01f;

    // Decoder / Upsampler
    int decoder_dim = 1536;
    std::vector<int> upsample_rates = {8, 5, 4, 3};
    std::vector<int> upsampling_ratios = {2, 2}; 
};

// Tensor data container (simple wrapper around float* for now)
struct TensorData {
    std::vector<float> data;
    std::vector<int64_t> shape;
};
