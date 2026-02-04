#pragma once

#include "ggml.h"
#include "qwen3_types.h"
#include <vector>
#include <string>
#include <map>

struct Qwen3Talker {
    Qwen3TalkerConfig config;
    std::map<std::string, struct ggml_tensor *> weights;
    struct ggml_context * ctx_w = nullptr;

    Qwen3Talker(const Qwen3TalkerConfig & cfg);
    ~Qwen3Talker();

    bool load_weights(const std::string & model_path);

    // Generate audio codes from text
    // Returns: vector of codes for each frame. Shape: (n_codebooks, seq_len)
    // Note: This is simplified. Real output is flattened or shaped differently.
    // For Qwen3-TTS, we predict codes.
    std::vector<int32_t> generate(const std::string & text, const std::string & ref_audio_path);
};
