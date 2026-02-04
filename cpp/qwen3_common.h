#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <map>

namespace qwen3 {

// Shared configuration parameters
struct model_config {
    int32_t n_threads = 4;
    ggml_backend_dev_t device = nullptr;
};

// Common GGML helper functions
struct ggml_tensor * rms_norm(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * w, float eps);
struct ggml_tensor * silu(struct ggml_context * ctx, struct ggml_tensor * x);
struct ggml_tensor * softmax(struct ggml_context * ctx, struct ggml_tensor * x);

// Weight loading utilities
struct weight_map {
    std::map<std::string, struct ggml_tensor *> tensors;
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ~weight_map();
};

bool load_weights_gguf(const std::string & path, weight_map & wm, ggml_backend_t backend);

} // namespace qwen3
