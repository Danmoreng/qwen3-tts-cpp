#include "qwen3_common.h"
#include "gguf.h"
#include <cstdio>
#include <iostream>

namespace qwen3 {

weight_map::~weight_map() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
}

struct ggml_tensor * rms_norm(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * w, float eps) {
    struct ggml_tensor * res = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, res, w);
}

// ... more helpers can be added ...

bool load_weights_gguf(const std::string & path, weight_map & wm, ggml_backend_t backend) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    
    struct gguf_context * ctx_gguf = gguf_init_from_file(path.c_str(), params);
    if (!ctx_gguf) return false;
    
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    const size_t meta_size = n_tensors * ggml_tensor_overhead();
    
    struct ggml_init_params init_params = {
        /*.mem_size   =*/ meta_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    wm.ctx = ggml_init(init_params);
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        int tensor_id = gguf_find_tensor(ctx_gguf, name);
        // In GGUF, tensor info is also in the metadata context if we loaded it
        // but here we can just use the name to find it in the gguf context.
        // Actually, ggml_new_tensor requires the shape.
        // We can get shape from gguf_get_tensor_n_dims etc IF available, but they might not be.
        // The most reliable way is to use the meta_ctx if we have it.
        struct ggml_tensor * t_meta = ggml_get_tensor(meta_ctx, name);
        if (!t_meta) continue;

        struct ggml_tensor * t = ggml_new_tensor(wm.ctx, t_meta->type, ggml_n_dims(t_meta), t_meta->ne);
        ggml_set_name(t, name);
        wm.tensors[name] = t;
    }
    
    wm.buffer = ggml_backend_alloc_ctx_tensors(wm.ctx, backend);
    
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        gguf_free(ctx_gguf);
        ggml_free(meta_ctx);
        return false;
    }
    
    size_t data_offset = gguf_get_data_offset(ctx_gguf);
    std::vector<uint8_t> read_buf;
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        size_t offset = gguf_get_tensor_offset(ctx_gguf, i);
        struct ggml_tensor * t = wm.tensors[name];
        size_t nbytes = ggml_nbytes(t);
        
        read_buf.resize(nbytes);
        fseek(f, (long)(data_offset + offset), SEEK_SET);
        fread(read_buf.data(), 1, nbytes, f);
        ggml_backend_tensor_set(t, read_buf.data(), 0, nbytes);
    }
    
    fclose(f);
    gguf_free(ctx_gguf);
    ggml_free(meta_ctx);
    return true;
}

} // namespace qwen3
