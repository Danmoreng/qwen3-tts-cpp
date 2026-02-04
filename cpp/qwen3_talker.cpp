#include "qwen3_talker.h"
#include <iostream>
#include <algorithm>

Qwen3Talker::Qwen3Talker(const Qwen3TalkerConfig & cfg) {
    // Config logic now handled by LLM internally via GGUF
}

Qwen3Talker::~Qwen3Talker() = default;

bool Qwen3Talker::load_weights(const std::string & model_path) {
    return llm.load_model(model_path);
}

std::vector<int32_t> Qwen3Talker::tokenize(const std::string & text) {
    // Mock tokenizer: treat each char as a token (dumb) or just return fixed sequence
    // In reality we need sentencepiece/tiktoken.
    // For testing pipeline, return a few start tokens.
    // Qwen usually starts with specific BOS?
    // Let's return some dummy tokens that are within vocab range.
    std::vector<int32_t> tokens;
    tokens.push_back(151643); // PAD/BOS?
    for (char c : text) {
        tokens.push_back((int)c + 100); // Shift to avoid control codes, purely mock
    }
    return tokens;
}

std::vector<int32_t> Qwen3Talker::generate(const std::string & text, const std::string & ref_audio_path) {
    std::cout << "Talker: generating..." << std::endl;
    
    // 1. Tokenize text
    std::vector<int32_t> input_ids = tokenize(text);
    
    // 2. Generation Loop
    int max_new_tokens = 100; // Generate 100 frames
    std::vector<int32_t> generated_codes;
    
    // Initial Context
    llm.clear_kv_cache();
    
    // We need to maintain the sequence.
    // First pass: Process prompt
    // For now, we assume the LLM predicts *audio codes* directly?
    // Or text tokens?
    // Qwen3-TTS: Text -> Audio Codes.
    // So the input is text, output is audio code.
    // We feed text tokens. The next token predicted should be an audio code (after some special token?)
    // For 2.1, let's just run the prompt and generate N tokens auto-regressively.
    
    int n_past = 0;
    std::vector<int32_t> current_batch = input_ids;
    
    // Output container for logits
    std::vector<float> logits;
    
    for (int i = 0; i < max_new_tokens; ++i) {
        // Run LLM
        if (!llm.forward(current_batch.data(), current_batch.size(), n_past, logits)) {
            std::cerr << "LLM forward failed" << std::endl;
            break;
        }
        
        n_past += current_batch.size();
        
        // Sample next token (Greedy: Argmax)
        // Logits is [batch_size, vocab_size]. We want the last token's logits.
        int vocab_size = llm.get_config().vocab_size;
        size_t last_token_idx = current_batch.size() - 1;
        float * last_logits = logits.data() + last_token_idx * vocab_size;
        
        int best_id = 0;
        float best_val = -1e30f;
        for (int v = 0; v < vocab_size; ++v) {
            if (last_logits[v] > best_val) {
                best_val = last_logits[v];
                best_id = v;
            }
        }
        
        // Append to generated
        generated_codes.push_back(best_id);
        
        // Prepare next input
        current_batch = { best_id };
        
        // Optional: print progress
        if (i % 10 == 0) std::cout << "." << std::flush;
    }
    std::cout << " Done." << std::endl;
    
    // Expand to 8 codebooks (Mocking the predictor for now)
    // We generated Code 0. We need 8 codes per frame.
    // Output format: [c0_t0, c1_t0, ..., c7_t0, c0_t1, ...] or planar?
    // Decoder expects planar in memory usually, or interleaved?
    // Qwen3AudioDecoder: 
    //   codes: (seq_len, n_codebooks) - verified ne0=seq_len
    //   ggml_new_tensor_2d(ctx, TYPE_I32, seq_len, n_codebooks)
    //   ne0 is seq_len. In GGML, this means dim 0 stride is element size. dim 1 stride is seq_len * element size.
    //   So data is [CB0_all, CB1_all, ...] -> Planar.
    // Wait, let's check `ggml_new_tensor_2d`.
    // tensor->ne[0] = ne0 (seq_len)
    // tensor->ne[1] = ne1 (n_codebooks)
    // Memory layout is row-major (ne0 fastest).
    // So in memory: c0_t0, c0_t1, ... c0_tN, c1_t0, ...
    
    int seq_len = generated_codes.size();
    int n_codebooks = 8;
    std::vector<int32_t> final_codes(seq_len * n_codebooks, 0);
    
    // Fill CB0 with generated codes
    for (int t = 0; t < seq_len; ++t) {
        // Clamp to valid codebook range [0, 2047] for prototype safety
        final_codes[t] = std::max(0, std::min(generated_codes[t], 2047));
        // CB 1..7 are 0 (Mock)
    }
    
    return final_codes;
}