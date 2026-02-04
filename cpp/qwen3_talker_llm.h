#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "qwen3_types.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

namespace qwen3 {

// Extended config for the LLM part of the Talker
struct Qwen3TalkerLLMConfig {
    int32_t vocab_size = 151936;
    int32_t hidden_size = 1024;
    int32_t n_layers = 28;
    int32_t n_heads = 16;
    int32_t n_kv_heads = 8;
    int32_t intermediate_size = 3072;
    int32_t head_dim = 64;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    
    // MRoPE sections: temporal, height, width
    std::vector<int32_t> mrope_section = {16, 56, 56}; // Sum must be head_dim (128)
    
    // TTS specific special tokens (if needed for generation loop)
    int32_t boa_token_id = -1; // Begin of Audio
    int32_t eoa_token_id = -1; // End of Audio
};

struct Qwen3TalkerLayer {
    struct ggml_tensor * attn_norm = nullptr;
    
    struct ggml_tensor * attn_q = nullptr;
    struct ggml_tensor * attn_k = nullptr;
    struct ggml_tensor * attn_v = nullptr;
    struct ggml_tensor * attn_output = nullptr;
    
    // Qwen2/3 specific norms if present
    struct ggml_tensor * attn_q_norm = nullptr;
    struct ggml_tensor * attn_k_norm = nullptr;
    
    struct ggml_tensor * ffn_norm = nullptr;
    
    struct ggml_tensor * ffn_gate = nullptr;
    struct ggml_tensor * ffn_up = nullptr;
    struct ggml_tensor * ffn_down = nullptr;
};

// Talker model weights
struct Qwen3TalkerModel {
    Qwen3TalkerLLMConfig config;
    
    // Token embedding
    struct ggml_tensor * token_embd = nullptr;
    
    // Transformer layers
    std::vector<Qwen3TalkerLayer> layers;
    
    // Final RMSNorm
    struct ggml_tensor * output_norm = nullptr;
    
    // LM head
    struct ggml_tensor * output = nullptr;
    
    // GGML context for tensor metadata
    struct ggml_context * ctx = nullptr;
    
    // Backend buffer for weights
    ggml_backend_buffer_t buffer = nullptr;
    
    // Tensor name to tensor mapping
    std::map<std::string, struct ggml_tensor *> tensors;
};

// KV cache for autoregressive generation
struct Qwen3KVCache {
    std::vector<struct ggml_tensor *> k_cache;  // Per-layer K cache
    std::vector<struct ggml_tensor *> v_cache;  // Per-layer V cache
    
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    int32_t n_ctx = 0;      // Maximum context length
    int32_t n_used = 0;     // Current number of cached tokens
    int32_t head_dim = 0;
    int32_t n_kv_heads = 0;
    int32_t n_layers = 0;
};

// Runtime state
struct Qwen3TalkerState {
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    
    std::vector<uint8_t> compute_meta;
    
    Qwen3KVCache cache;
};

// Main Talker LLM class
class Qwen3TalkerLLM {
public:
    Qwen3TalkerLLM();
    ~Qwen3TalkerLLM();
    
    // Load model from GGUF file
    bool load_model(const std::string & model_path);
    
    // Initialize KV cache for given context length
    bool init_kv_cache(int32_t n_ctx);
    
    // Clear KV cache (for new sequence)
    void clear_kv_cache();
    
    // Forward pass: compute logits for input tokens
    // tokens: input token IDs [n_tokens]
    // pos_ids: multimodal position IDs [3, n_tokens] (temporal, height, width)
    // n_past: number of tokens already in KV cache
    // output: logits [n_tokens, vocab_size] (flattened)
    bool forward(const int32_t * tokens, const int32_t * pos_ids, int32_t n_tokens, int32_t n_past,
                 std::vector<float> & output);
                 
    const Qwen3TalkerLLMConfig & get_config() const { return model_.config; }
    const std::string & get_error() const { return error_msg_; }
    
private:
    // Build computation graph for forward pass
    struct ggml_cgraph * build_graph(const int32_t * tokens, const int32_t * pos_ids, int32_t n_tokens, int32_t n_past);
    
    // Parse hyperparameters from GGUF
    bool parse_config(struct gguf_context * ctx);
    
    // Create tensor structures
    bool create_tensors(struct gguf_context * ctx, struct ggml_context * meta_ctx);
    
    // Load tensor data from file
    bool load_tensor_data(const std::string & path, struct gguf_context * ctx);
    
    Qwen3TalkerModel model_;
    Qwen3TalkerState state_;
    std::string error_msg_;
    
    // Cleanup helpers
    void free_model();
    void free_kv_cache();
};

} // namespace qwen3
