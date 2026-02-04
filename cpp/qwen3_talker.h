#pragma once

#include "qwen3_talker_llm.h"
#include <vector>
#include <string>

// High-level Talker class that orchestrates the LLM generation
class Qwen3Talker {
public:
    Qwen3Talker(const Qwen3TalkerConfig & cfg); // Kept for compat with main.cpp for now
    ~Qwen3Talker();

    bool load_weights(const std::string & model_path);

    // Generate audio codes from text
    // Returns flattened codes
    std::vector<int32_t> generate(const std::string & text, const std::string & ref_audio_path);

private:
    qwen3::Qwen3TalkerLLM llm;
    
    // Helper to tokenize text (placeholder for now)
    std::vector<int32_t> tokenize(const std::string & text);
};