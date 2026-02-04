#include "qwen3_talker_llm.h"
#include "qwen3_common.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <iostream>

#define QWEN3_TALKER_MAX_NODES 8192

namespace qwen3 {

Qwen3TalkerLLM::Qwen3TalkerLLM() = default;

Qwen3TalkerLLM::~Qwen3TalkerLLM() {
    free_kv_cache();
    free_model();
}

void Qwen3TalkerLLM::free_model() {
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        ggml_backend_free(state_.backend);
        state_.backend = nullptr;
    }
    if (model_.buffer) {
        ggml_backend_buffer_free(model_.buffer);
        model_.buffer = nullptr;
    }
    if (model_.ctx) {
        ggml_free(model_.ctx);
        model_.ctx = nullptr;
    }
    model_.tensors.clear();
    model_.layers.clear();
}

void Qwen3TalkerLLM::free_kv_cache() {
    if (state_.cache.buffer) {
        ggml_backend_buffer_free(state_.cache.buffer);
        state_.cache.buffer = nullptr;
    }
    if (state_.cache.ctx) {
        ggml_free(state_.cache.ctx);
        state_.cache.ctx = nullptr;
    }
    state_.cache.k_cache.clear();
    state_.cache.v_cache.clear();
    state_.cache.n_ctx = 0;
    state_.cache.n_used = 0;
}

bool Qwen3TalkerLLM::load_model(const std::string & model_path) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    
    struct gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open GGUF file: " + model_path;
        return false;
    }
    
    if (!parse_config(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!create_tensors(ctx, meta_ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!load_tensor_data(model_path, ctx)) {
        free_model();
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);
    
    // Initialize backends
    std::vector<ggml_backend_t> backends;

#ifdef GGML_USE_CUDA
    ggml_backend_t cuda_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (cuda_backend) {
        backends.push_back(cuda_backend);
    } else {
        std::cerr << "Warning: CUDA backend failed to init." << std::endl;
    }
#endif

    // Always add CPU backend last
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_backend) {
        error_msg_ = "Failed to initialize CPU backend";
        return false;
    }
    backends.push_back(cpu_backend);
    
    // Store primary backend for simple access
    state_.backend = backends[0]; 
    
    // Create scheduler
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), QWEN3_TALKER_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }
    
    // Reserve space for compute meta
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TALKER_MAX_NODES + ggml_graph_overhead());
    
    return true;
}

bool Qwen3TalkerLLM::parse_config(struct gguf_context * ctx) {
    auto get_u32 = [&](const char * key, int32_t default_val) -> int32_t {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) return default_val;
        return (int32_t)gguf_get_val_u32(ctx, idx);
    };
    
    auto get_f32 = [&](const char * key, float default_val) -> float {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) return default_val;
        return gguf_get_val_f32(ctx, idx);
    };
    
    auto & cfg = model_.config;
    // Keys match cpp/convert_qwen3_to_gguf.py
    cfg.vocab_size = get_u32("talker.vocab_size", 151936);
    cfg.hidden_size = get_u32("talker.embedding_length", 1024);
    cfg.n_layers = get_u32("talker.block_count", 28);
    cfg.n_heads = get_u32("talker.attention.head_count", 16);
    cfg.n_kv_heads = get_u32("talker.attention.head_count_kv", 8);
    cfg.intermediate_size = get_u32("talker.feed_forward_length", 3072);
    
    // Some configs might not be present in the script yet, use defaults or infer
    cfg.head_dim = cfg.hidden_size / cfg.n_heads; 
    
    // Note: eps and rope_theta were not explicitly saved in convert script?
    // Using Qwen2 defaults if missing
    cfg.rms_norm_eps = get_f32("talker.attention.layer_norm_rms_epsilon", 1e-6f);
    cfg.rope_theta = get_f32("talker.rope.freq_base", 1000000.0f);
    
    return true;
}

bool Qwen3TalkerLLM::create_tensors(struct gguf_context * ctx, struct ggml_context * meta_ctx) {
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const auto & cfg = model_.config;
    
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to create GGML context";
        return false;
    }
    
    model_.layers.resize(cfg.n_layers);
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model_.ctx, name);
        
        if (!tensor) {
            // Get info from meta_ctx
            struct ggml_tensor * t_meta = ggml_get_tensor(meta_ctx, name);
            if (!t_meta) continue; // Should not happen if name is from gguf
            
            tensor = ggml_new_tensor(model_.ctx, t_meta->type, ggml_n_dims(t_meta), t_meta->ne);
            ggml_set_name(tensor, name);
        }
        
        model_.tensors[name] = tensor;
        
        // Map to struct fields
        // Names based on convert_qwen3_to_gguf.py
        if (strcmp(name, "talker.text_embedding.weight") == 0) {
            model_.token_embd = tensor;
        } else if (strcmp(name, "talker.output_norm.weight") == 0) {
            model_.output_norm = tensor;
        } else if (strcmp(name, "talker.codec_head.weight") == 0) {
            model_.output = tensor;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "talker.blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < cfg.n_layers) {
                auto & layer = model_.layers[layer_idx];
                
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        }
    }
    
    return true;
}

bool Qwen3TalkerLLM::load_tensor_data(const std::string & path, struct gguf_context * ctx) {
    // Need a temporary backend usage for allocation
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    // Actually we should use the state_.backend logic but that's not init yet.
    // Let's alloc on CPU for simplicity of loading, but if we want GPU, we should init it earlier.
    // NOTE: In load_model we init backend AFTER this. 
    // To support GPU offloading properly, we should init backend FIRST.
    // But let's assume CPU loading for now to be safe, or just move backend init up.
    // Actually, ggml_backend_alloc_ctx_tensors allocates memory on the backend.
    // If we want CUDA, we must provide CUDA backend here.
    
    // Re-doing backend init logic locally for loader
#ifdef GGML_USE_CUDA
    ggml_backend_t load_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!load_backend) load_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
#else
    ggml_backend_t load_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
#endif

    if (!load_backend) return false;

    model_.buffer = ggml_backend_alloc_ctx_tensors(model_.ctx, load_backend);
    if (!model_.buffer) {
        ggml_backend_free(load_backend);
        return false;
    }
    
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        ggml_backend_free(load_backend);
        return false;
    }
    
    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model_.ctx, name);
        if (!tensor) continue;
        
        size_t offset = gguf_get_tensor_offset(ctx, i);
        size_t nbytes = ggml_nbytes(tensor);
        
        // Direct read if backend supports it (CPU), else read to buffer
        if (ggml_backend_is_cpu(load_backend)) {
            fseek(f, data_offset + offset, SEEK_SET);
            fread(tensor->data, 1, nbytes, f);
        } else {
            // For GPU, read to host buf then copy
            if (read_buf.size() < nbytes) read_buf.resize(nbytes);
            fseek(f, data_offset + offset, SEEK_SET);
            fread(read_buf.data(), 1, nbytes, f);
            ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
        }
    }
    
    fclose(f);
    ggml_backend_free(load_backend);
    
    return true;
}

bool Qwen3TalkerLLM::init_kv_cache(int32_t n_ctx) {
    const auto & cfg = model_.config;
    
    free_kv_cache();
    
    state_.cache.n_ctx = n_ctx;
    state_.cache.n_used = 0;
    state_.cache.head_dim = cfg.head_dim;
    state_.cache.n_kv_heads = cfg.n_kv_heads;
    state_.cache.n_layers = cfg.n_layers;
    
    const size_t n_tensors = cfg.n_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    state_.cache.ctx = ggml_init(params);
    
    state_.cache.k_cache.resize(cfg.n_layers);
    state_.cache.v_cache.resize(cfg.n_layers);
    
    for (int il = 0; il < cfg.n_layers; ++il) {
        state_.cache.k_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16, // Use F16 for KV cache efficiency
            cfg.head_dim, cfg.n_kv_heads, n_ctx);
        ggml_format_name(state_.cache.k_cache[il], "k_cache_%d", il);
        
        state_.cache.v_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_kv_heads, n_ctx);
        ggml_format_name(state_.cache.v_cache[il], "v_cache_%d", il);
    }
    
    state_.cache.buffer = ggml_backend_alloc_ctx_tensors(state_.cache.ctx, state_.backend);
    
    return true;
}

void Qwen3TalkerLLM::clear_kv_cache() {
    state_.cache.n_used = 0;
}

struct ggml_cgraph * Qwen3TalkerLLM::build_graph(const int32_t * tokens, int32_t n_tokens, int32_t n_past) {
    const auto & cfg = model_.config;
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TALKER_MAX_NODES, false);
    
    // Inputs
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    // Embedding
    if (!model_.token_embd) {
        std::cerr << "Error: token_embd is NULL!" << std::endl;
        return nullptr;
    }
    struct ggml_tensor * cur = ggml_get_rows(ctx0, model_.token_embd, inp_tokens);
    
    const float eps = cfg.rms_norm_eps;
    const int n_head = cfg.n_heads;
    const int n_kv_head = cfg.n_kv_heads;
    const int head_dim = cfg.head_dim;
    const float rope_theta = cfg.rope_theta;
    
    for (int il = 0; il < cfg.n_layers; ++il) {
        // std::cout << "Building layer " << il << std::endl;
        const auto & layer = model_.layers[il];
        
        struct ggml_tensor * residual = cur;
        
        // Norm
        if (!layer.attn_norm) {
             std::cerr << "Error: attn_norm missing for layer " << il << std::endl;
             return nullptr;
        }
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        // QKV
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        // Reshape [N, head_dim * n_head] -> [N, n_head, head_dim] -> [head_dim, n_head, N] (transposed)
        // ggml_reshape_3d: [ne0, ne1, ne2]
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        
        // RoPE
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        // KV Cache Update
        struct ggml_tensor * k_cache = state_.cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.cache.v_cache[il];
        
        // view into cache for current batch
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache, 
            head_dim, n_kv_head, n_tokens, 
            k_cache->nb[1], k_cache->nb[2], 
            n_past * k_cache->nb[2]);
            
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache, 
            head_dim, n_kv_head, n_tokens, 
            v_cache->nb[1], v_cache->nb[2], 
            n_past * v_cache->nb[2]);
            
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        // Attention
        // View full context
        int n_ctx_curr = n_past + n_tokens;
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache, head_dim, n_kv_head, n_ctx_curr, k_cache->nb[1], k_cache->nb[2], 0);
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache, head_dim, n_kv_head, n_ctx_curr, v_cache->nb[1], v_cache->nb[2], 0);
        
        // Permute Q: [head_dim, n_head, N] -> [head_dim, N, n_head]
        // Actually typical ggml attention:
        // Q: [head_dim, N, n_head]
        // K: [head_dim, n_ctx, n_kv_head]
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        
        // K * Q
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q); // [n_ctx, N, n_head]
        
        // Scale
        KQ = ggml_scale(ctx0, KQ, 1.0f / sqrtf(float(head_dim)));
        
        // Mask (Causal)
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        
        // Softmax
        KQ = ggml_soft_max(ctx0, KQ);
        
        // V * KQ
        // V needs to be contiguous/transposed properly? 
        // ggml_mul_mat V: [head_dim, n_ctx, n_head] * KQ: [n_ctx, N, n_head] -> [head_dim, N, n_head]
        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        
        // Permute back: [head_dim, N, n_head] -> [head_dim, n_head, N] -> flatten to [hidden, N]
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, head_dim * n_head, n_tokens);
        
        // Output proj
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        
        // Residual
        cur = ggml_add(ctx0, cur, residual);
        
        // FFN
        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        cur = ggml_mul(ctx0, gate, up);
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
        
        cur = ggml_add(ctx0, cur, residual);
    }
    
    // Output Norm
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    
    // LM Head
    cur = ggml_mul_mat(ctx0, model_.output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    return gf;
}

bool Qwen3TalkerLLM::forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                             std::vector<float> & output) {
    if (!model_.ctx) return false;
    if (state_.cache.n_ctx == 0) init_kv_cache(2048); // Auto-init
    
    // std::cout << "Building graph..." << std::endl;
    struct ggml_cgraph * gf = build_graph(tokens, n_tokens, n_past);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        std::cerr << "Failed to allocate graph" << std::endl;
        return false;
    }
    
    // Set inputs
    // std::cout << "Setting inputs..." << std::endl;
    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    ggml_backend_tensor_set(inp_tokens, tokens, 0, n_tokens * sizeof(int32_t));
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    std::vector<int32_t> pos(n_tokens);
    for (int i=0; i<n_tokens; ++i) pos[i] = n_past + i;
    ggml_backend_tensor_set(inp_pos, pos.data(), 0, n_tokens * sizeof(int32_t));
    
    // Compute
    // std::cout << "Computing graph..." << std::endl;
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "Graph compute failed" << std::endl;
        return false;
    }
    
    // Output
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    output.resize(n_tokens * model_.config.vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    
    state_.cache.n_used = n_past + n_tokens;
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

} // namespace qwen3
