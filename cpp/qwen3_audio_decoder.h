#pragma once

#include "ggml.h"
#include "qwen3_types.h"
#include <vector>
#include <string>

struct Qwen3AudioDecoder {
    Qwen3AudioDecoderConfig config;
    
    // Model weights (ggml tensors)
    // We use a map for flexibility during development
    std::map<std::string, struct ggml_tensor *> weights;

    struct ggml_context * ctx_w = nullptr; // Context for weights

    Qwen3AudioDecoder(const Qwen3AudioDecoderConfig & cfg);
    ~Qwen3AudioDecoder();

    // Load weights from a file (GGUF or raw dump)
    bool load_weights(const std::string & model_path);

    // Build the compute graph for decoding codes -> audio
    // codes: (n_codebooks, seq_len)
    struct ggml_cgraph * build_graph(struct ggml_context * ctx0, struct ggml_tensor * codes);
};

// Helper to build SnakeBeta activation subgraph
// x: input tensor
// alpha, beta: parameter tensors
struct ggml_tensor * ggml_snake_beta(
    struct ggml_context * ctx, 
    struct ggml_tensor * x, 
    struct ggml_tensor * alpha, 
    struct ggml_tensor * beta
);
