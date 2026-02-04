#include "qwen3_talker.h"
#include "gguf.h"
#include <iostream>
#include <cstdio>

#ifdef _WIN32
#define fseeko _fseeki64
#define ftello _ftelli64
#endif

Qwen3Talker::Qwen3Talker(const Qwen3TalkerConfig & cfg) : config(cfg) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024 * 10ULL, // 10GB
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    ctx_w = ggml_init(params);
}

Qwen3Talker::~Qwen3Talker() {
    if (ctx_w) ggml_free(ctx_w);
}

bool Qwen3Talker::load_weights(const std::string & model_path) {
    std::cout << "Loading talker weights from " << model_path << "..." << std::endl;
    
    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };
    
    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx_gguf) {
        std::cerr << "Failed to load GGUF file" << std::endl;
        return false;
    }

    FILE * f = fopen(model_path.c_str(), "rb");
    if (!f) return false;
    
    fseeko(f, 0, SEEK_END);
    // size_t file_size = ftello(f);
    rewind(f);

    size_t data_offset = gguf_get_data_offset(ctx_gguf);
    
    struct ggml_tensor * cur = ggml_get_first_tensor(ctx_meta);
    int loaded_count = 0;

    while (cur) {
        const char * name = ggml_get_name(cur);
        
        if (strncmp(name, "talker.", 7) == 0) {
            struct ggml_tensor * tensor = ggml_get_tensor(ctx_w, name);
            if (!tensor) {
                int n_dims = ggml_n_dims(cur);
                tensor = ggml_new_tensor(ctx_w, cur->type, n_dims, cur->ne);
                ggml_set_name(tensor, name);
                
                int tensor_id = gguf_find_tensor(ctx_gguf, name);
                if (tensor_id >= 0) {
                    size_t offset = gguf_get_tensor_offset(ctx_gguf, tensor_id);
                    size_t size = ggml_nbytes(tensor);
                    
                    fseeko(f, data_offset + offset, SEEK_SET);
                    fread(tensor->data, 1, size, f);
                    
                    weights[std::string(name)] = tensor;
                    loaded_count++;
                }
            }
        }
        cur = ggml_get_next_tensor(ctx_meta, cur);
    }

    fclose(f);
    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);
    
    std::cout << "Loaded " << loaded_count << " talker tensors." << std::endl;
    return true;
}

std::vector<int32_t> Qwen3Talker::generate(const std::string & text, const std::string & ref_audio_path) {
    std::cout << "Talker generating (mock)..." << std::endl;
    // Return dummy codes: 100 frames * 8 codebooks
    // Flattened or structured?
    // Decoder expects (seq_len, n_codebooks) tensor.
    // If we return flat vector, main.cpp handles it.
    std::vector<int32_t> codes(100 * 8, 0);
    return codes;
}
