#include <iostream>
#include <vector>
#include <string>

#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "qwen3_types.h"
#include "qwen3_audio_decoder.h"
#include "qwen3_talker.h"

int main(int argc, char ** argv) {
    std::cout << "Qwen3-TTS C++ Inference (Pure CPP/CUDA)" << std::endl;
    std::cout << "Backend: llama.cpp / ggml" << std::endl;

    // 1. Initialize Backend
    llama_backend_init();
    
    // 2. Load Models
    std::string model_path = "models/qwen3_tts_full.gguf";

    Qwen3TalkerConfig talker_cfg;
    Qwen3Talker talker(talker_cfg);
    if (!talker.load_weights(model_path)) {
        std::cerr << "Failed to load talker weights from " << model_path << std::endl;
        return 1;
    }

    Qwen3AudioDecoderConfig decoder_cfg; 
    Qwen3AudioDecoder decoder(decoder_cfg); // Config will be updated from GGUF inside load_weights
    if (!decoder.load_weights(model_path)) {
        std::cerr << "Failed to load decoder weights from " << model_path << std::endl;
        return 1;
    }

    // 3. Inference Pipeline
    std::string text = "Hello, this is a test of Qwen3-TTS in C++.";
    std::string ref_audio = "ref.wav";

    // A. Generate Codes (Talker)
    // Returns flattened codes (planar: [cb0...], [cb1...])
    auto codes_vec = talker.generate(text, ref_audio);
    
    int seq_len = 100; // Mock length from talker
    int n_codebooks = 8; // Mock
    // Verify size
    if (codes_vec.size() != (size_t)(seq_len * n_codebooks)) {
        std::cerr << "Code vector size mismatch!" << std::endl;
        // Adjust for prototype safety
        seq_len = (int)(codes_vec.size() / n_codebooks);
    }

    // Convert codes to ggml tensor for decoding
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024 * 2ULL, // 2GB for graph nodes
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx_compute = ggml_init(params);
    
    // Create input tensor (seq_len, n_codebooks) -> planar in memory
    struct ggml_tensor * codes_tensor = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_I32, seq_len, n_codebooks);
    
    // Copy data
    memcpy(codes_tensor->data, codes_vec.data(), codes_vec.size() * sizeof(int32_t));
    
    // B. Decode Audio (Decoder)
    std::cout << "Decoder: building graph..." << std::endl;
    struct ggml_cgraph * gf = decoder.build_graph(ctx_compute, codes_tensor);
    
    std::cout << "Decoder: computing graph..." << std::endl;
    // Run graph
    ggml_graph_compute_with_ctx(ctx_compute, gf, 4); // 4 threads

    // Get output
    // The last node in graph usually is the output
    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    
    std::cout << "Output shape: " << out->ne[0] << ", " << out->ne[1] << std::endl;
    std::cout << "Inference complete. Output saved to 'out.wav' (simulated)." << std::endl;

    ggml_free(ctx_compute);
    llama_backend_free();
    return 0;
}
