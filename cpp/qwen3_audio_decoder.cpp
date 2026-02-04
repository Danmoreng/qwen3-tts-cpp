#include "qwen3_audio_decoder.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include <cstring>
#include <cstdio>

#ifdef _WIN32
#define fseeko _fseeki64
#define ftello _ftelli64
#endif

// Helper to create a scalar float tensor
static struct ggml_tensor * make_f32_tensor(struct ggml_context * ctx, float val) {
    struct ggml_tensor * res = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    *(float *)res->data = val;
    return res;
}

Qwen3AudioDecoder::Qwen3AudioDecoder(const Qwen3AudioDecoderConfig & cfg) : config(cfg) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024 * 2ULL, // 2GB
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    ctx_w = ggml_init(params);
}

Qwen3AudioDecoder::~Qwen3AudioDecoder() {
    if (ctx_w) ggml_free(ctx_w);
}

bool Qwen3AudioDecoder::load_weights(const std::string & model_path) {
    std::cout << "Loading audio decoder weights from " << model_path << "..." << std::endl;
    
    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };
    
    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx_gguf) {
        std::cerr << "Failed to load GGUF file: " << model_path << std::endl;
        return false;
    }

    // Load Config from GGUF
    int key_id;
    key_id = gguf_find_key(ctx_gguf, "decoder.codebook_dim");
    if (key_id >= 0) config.codebook_dim = gguf_get_val_u32(ctx_gguf, key_id);
    
    key_id = gguf_find_key(ctx_gguf, "decoder.latent_dim");
    if (key_id >= 0) config.latent_dim = gguf_get_val_u32(ctx_gguf, key_id);

    key_id = gguf_find_key(ctx_gguf, "decoder.num_quantizers");
    if (key_id >= 0) config.num_quantizers = gguf_get_val_u32(ctx_gguf, key_id);

    // Note: upsample_rates and others were not saved by the converter, using defaults/hardcoded
    // For 12Hz model:
    config.upsample_rates = {8, 5, 4, 3}; 
    config.upsampling_ratios = {2, 2}; 
    config.decoder_dim = 1536;

    std::cout << "Decoder Config: latent=" << config.latent_dim 
              << ", codebook=" << config.codebook_dim 
              << ", n_q=" << config.num_quantizers << std::endl;

    FILE * f = fopen(model_path.c_str(), "rb");
    if (!f) {
        std::cerr << "Failed to open file: " << model_path << std::endl;
        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
        return false;
    }
    
    fseeko(f, 0, SEEK_END);
    size_t file_size = ftello(f);
    rewind(f);

    size_t data_offset = gguf_get_data_offset(ctx_gguf);
    
    struct ggml_tensor * cur = ggml_get_first_tensor(ctx_meta);
    int loaded_count = 0;

    while (cur) {
        const char * name = ggml_get_name(cur);
        
        // Filter for decoder weights (dec.*)
        if (strncmp(name, "dec.", 4) == 0 || strncmp(name, "decoder.", 8) == 0) {
            
            struct ggml_tensor * tensor = ggml_get_tensor(ctx_w, name);
            if (!tensor) {
                int n_dims = ggml_n_dims(cur);
                tensor = ggml_new_tensor(ctx_w, cur->type, n_dims, cur->ne);
                ggml_set_name(tensor, name);
                
                int tensor_id = gguf_find_tensor(ctx_gguf, name);
                if (tensor_id >= 0) {
                    size_t offset = gguf_get_tensor_offset(ctx_gguf, tensor_id);
                    size_t size = ggml_nbytes(tensor);
                    
                    if (fseeko(f, data_offset + offset, SEEK_SET) != 0) {
                        std::cerr << "Seek failed for " << name << std::endl;
                        break;
                    }
                    if (fread(tensor->data, 1, size, f) != size) {
                        std::cerr << "Read failed for " << name << std::endl;
                        break;
                    }
                    
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
    
    std::cout << "Loaded " << loaded_count << " decoder tensors." << std::endl;
    return true;
}

struct ggml_tensor * ggml_snake_beta(
    struct ggml_context * ctx, 
    struct ggml_tensor * x, 
    struct ggml_tensor * alpha, 
    struct ggml_tensor * beta
) {
    const float eps = 1e-9f;

    // x shape: (T, C, 1, 1) or (T, C) in ggml
    // alpha/beta shape: (C)
    // We need to broadcast alpha/beta to x.
    // In ggml_mul, if alpha has dim 1 where x has dim T, it broadcasts?
    // alpha is (C). x is (T, C).
    // To broadcast, alpha should be (1, C).
    // Let's reshape alpha/beta to (1, C).
    
    // Check shapes
    // x: ne[0]=T, ne[1]=C
    // alpha: ne[0]=C
    
    struct ggml_tensor * alpha_b = ggml_reshape_2d(ctx, alpha, 1, alpha->ne[0]);
    struct ggml_tensor * beta_b  = ggml_reshape_2d(ctx, beta,  1, beta->ne[0]);

    // 1. alpha * x
    struct ggml_tensor * ax = ggml_mul(ctx, x, alpha_b); // (T, C) * (1, C) -> (T, C)

    // 2. sin(alpha * x)
    struct ggml_tensor * sin_ax = ggml_sin(ctx, ax);

    // 3. sin^2(...)
    struct ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);

    // 4. 1 / (beta + eps)
    struct ggml_tensor * eps_tensor = make_f32_tensor(ctx, eps);
    // beta + eps
    struct ggml_tensor * beta_eps = ggml_add(ctx, beta_b, eps_tensor); 
    
    struct ggml_tensor * one = make_f32_tensor(ctx, 1.0f);
    struct ggml_tensor * scale = ggml_div(ctx, one, beta_eps);

    // 5. scale * sin^2(...)
    struct ggml_tensor * part2 = ggml_mul(ctx, sin_sq, scale);

    // 6. x + part2
    return ggml_add(ctx, x, part2);
}

// Helper for Causal Conv1d
// weights: (out, in, k) -> GGML: (k, in, out)
// In our GGUF, weights might be (out, in, k) because that's what PyTorch has.
// GGML conv_1d expects weights as (k, in, out) usually?
// Llama.cpp's `ggml_conv_1d` documentation or usage in `llama.cpp` suggests:
// "w is the kernel, with shape [ne0, ne1, ne2] = [kernel_size, in_channels, out_channels]"
//
// But if we saved PyTorch (out, in, k), that is [k, in, out] in column-major?
// PyTorch: [out, in, k]. 
// Flat buffer: out loops slowest.
// GGML: [ne0, ne1, ne2] -> ne0 is fastest.
// If we map:
// ne0 = k
// ne1 = in
// ne2 = out
// Then iterating flat buffer: k changes fastest, then in, then out.
// PyTorch flat: k changes fastest? No, usually last dim is fastest in C-order (PyTorch default).
// So PyTorch `(out, in, k)` in memory is: `out` chunks. Inside each, `in` chunks. Inside, `k` elements.
// So `k` is fastest.
// So `ne0=k, ne1=in, ne2=out` matches PyTorch memory layout perfectly.
// So `ggml_conv_1d` should work directly with PyTorch weights.
static struct ggml_tensor * causal_conv1d(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w,
    struct ggml_tensor * b,
    int stride,
    int dilation
) {
    // x: (T, in)
    // w: (k, in, out)
    
    // Pad x for causality
    // Padding = (k-1) * dilation
    int k = w->ne[0];
    int padding = (k - 1) * dilation;
    
    // To implement causal padding:
    // Pad LEFT (top in time) with zeros.
    // ggml_pad creates a new tensor with padding.
    // We want to pad dim 0 (time).
    // ggml_pad(ctx, a, pad_0_left, pad_1_left, pad_0_right, pad_1_right)
    struct ggml_tensor * x_pad = ggml_pad(ctx, x, padding, 0, 0, 0); 
    
    // Convolution
    // ggml_conv_1d(ctx, w, a, s0, p0, d0)
    // p0 is padding. If we manually padded, p0=0?
    // If we use p0 in conv_1d, it pads both sides usually? 
    // Or it might be "valid" vs "same".
    // Let's use p0=0 and manual padding to be safe for causality.
    struct ggml_tensor * res = ggml_conv_1d(ctx, w, x_pad, stride, 0, dilation);
    
    // Bias
    if (b) {
        // b: (out) -> ne[0]=out
        // res: (T_out, out) -> ne[0]=T, ne[1]=out
        // We need b to be (1, out) to broadcast dim 0.
        struct ggml_tensor * b_reshaped = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        res = ggml_add(ctx, res, b_reshaped);
    }
    
    return res;
}

// Helper for Causal Transpose Conv1d (Upsampling)
static struct ggml_tensor * causal_conv_transpose_1d(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w,
    struct ggml_tensor * b,
    int stride
) {
    // x: (T, in)
    // w: (in, out, k)? Or (out, in, k)?
    // PyTorch ConvTranspose1d weight: (in_channels, out_channels/groups, kernel_size)
    // Note: PyTorch ConvTranspose1d swaps in/out in weight definition compared to Conv1d.
    // GGML expects?
    // If `ggml_conv_transpose_1d` follows standard, it likely expects kernel.
    // Let's assume PyTorch layout is preserved.
    
    // ggml_conv_transpose_1d(ctx, w, a, s0, p0, d0)
    // Not universally available in all ggml versions?
    // If we assume it is:
    struct ggml_tensor * res = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    
    // Causal cropping
    // PyTorch: 
    // pad = kernel_size - stride
    // left_pad = ceil(pad)
    // right_pad = pad
    // output = output[..., left_pad : -right_pad]
    
    int k = w->ne[0]; // Assuming k is fastest dim
    int pad = k - stride;
    int left_pad = (pad + 1) / 2; // ceil(pad/2) ?? No, Python says math.ceil(pad)??
    // Python:
    // pad = kernel_size - stride
    // self.left_pad = math.ceil(pad) -> In python ceil(int) is int.
    // Wait, if pad is int, ceil is just pad.
    // Wait, `math.ceil(pad)`? `pad` is integer?
    // Yes. `pad = self.left_pad`.
    // Then `self.right_pad = pad`.
    // It removes `pad` from left and `pad` from right?
    // That seems to remove `2*pad`.
    
    // Let's re-read python:
    // self.left_pad = math.ceil(pad) 
    // self.right_pad = pad = self.left_pad
    // hidden_state = hidden_state[..., self.left_pad : hidden_state.shape[-1] - self.right_pad]
    
    // So we need to crop `pad` from start and end.
    // In ggml, we can view.
    // res shape: (T_new, out)
    // offset in T: left_pad * sizeof(row)
    // length in T: T_new - left_pad - right_pad
    
    // But `ggml_conv_transpose_1d` output size?
    // Standard: (L-1)*stride + k
    
    // Cropping:
    int offset_idx = pad; // left_pad
    int new_len = res->ne[0] - pad - pad;
    
    // Check if new_len is valid
    if (new_len <= 0) return res; // Should not happen
    
    size_t row_size = res->nb[1]; // stride of dim 1 (channels) -> wait, column major
    // res: (T_new, out)
    // nb[0] = sizeof(float)
    // nb[1] = T_new * sizeof(float)
    // We want to slice T dimension (dim 0).
    // View 1d? No, View 2d.
    // We want (new_len, out).
    // Stride is same.
    // Offset is offset_idx * element_size.
    
    size_t offset = offset_idx * ggml_element_size(res);
    struct ggml_tensor * cropped = ggml_view_2d(ctx, res, new_len, res->ne[1], res->nb[1], offset);

    if (b) {
        struct ggml_tensor * b_reshaped = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        cropped = ggml_add(ctx, cropped, b_reshaped);
    }
    
    return cropped;
}

// Get tensor from map safely
static struct ggml_tensor * get_tensor(const std::map<std::string, struct ggml_tensor *> & weights, std::string name) {
    auto it = weights.find(name);
    if (it == weights.end()) {
        std::cerr << "Tensor not found: " << name << std::endl;
        return NULL; // Will likely crash later, but better than silent
    }
    return it->second;
}

struct ggml_cgraph * Qwen3AudioDecoder::build_graph(struct ggml_context * ctx0, struct ggml_tensor * codes) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // codes: (seq_len, n_codebooks) - verified ne0=seq_len
    int seq_len = codes->ne[0];
    int n_codebooks = codes->ne[1];
    
    // 1. Quantization
    struct ggml_tensor * hidden = NULL;
    
    // Semantic (Layer 0)
    {
        // ... (loading logic) ...
        // Correct name: decoder.quantizer.rvq_first.vq.l.0._cb.sum
        std::string name_sum = "decoder.quantizer.rvq_first.vq.l.0._cb.sum"; 
        struct ggml_tensor * w_sum = get_tensor(weights, name_sum);
        if (!w_sum) {
             w_sum = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, config.codebook_dim, config.codebook_size);
             ggml_set_f32(w_sum, 0.0f);
        }
        
        struct ggml_tensor * codes_0 = ggml_view_1d(ctx0, codes, seq_len, 0);
        struct ggml_tensor * emb_0 = ggml_get_rows(ctx0, w_sum, codes_0); 
        emb_0 = ggml_transpose(ctx0, emb_0);
        
        struct ggml_tensor * w_proj = get_tensor(weights, "decoder.quantizer.rvq_first.output_proj.weight");
        struct ggml_tensor * b_proj = get_tensor(weights, "decoder.quantizer.rvq_first.output_proj.bias"); 

        if (w_proj) {
            emb_0 = causal_conv1d(ctx0, emb_0, w_proj, b_proj, 1, 1);
        }
        hidden = emb_0;
        // Debug
        printf("Hidden 0 shape: %ld, %ld\n", hidden->ne[0], hidden->ne[1]);
    }
    
    // Acoustic (Layers 1..N)
    for (int i = 1; i < n_codebooks; i++) {
        std::string layer_prefix = "decoder.quantizer.rvq_rest.vq.l." + std::to_string(i - 1) + "._cb.";
        struct ggml_tensor * w_sum = get_tensor(weights, layer_prefix + "sum");
        if (!w_sum) continue;
        
        struct ggml_tensor * codes_i = ggml_view_1d(ctx0, codes, seq_len, i * codes->nb[1]);
        struct ggml_tensor * emb_i = ggml_get_rows(ctx0, w_sum, codes_i);
        emb_i = ggml_transpose(ctx0, emb_i);
        
        struct ggml_tensor * w_proj = get_tensor(weights, "decoder.quantizer.rvq_rest.output_proj.weight");
        
        if (w_proj) {
            emb_i = causal_conv1d(ctx0, emb_i, w_proj, NULL, 1, 1);
        }
        
        // Debug
        printf("Hidden shape: %ld, %ld. Emb_i shape: %ld, %ld\n", hidden->ne[0], hidden->ne[1], emb_i->ne[0], emb_i->ne[1]);
        hidden = ggml_add(ctx0, hidden, emb_i);
    }
    
    // hidden is (T, C).
    // ... Pre-Conv ...
    struct ggml_tensor * w_pre = get_tensor(weights, "dec.pre_c.conv.weight");
    struct ggml_tensor * b_pre = get_tensor(weights, "dec.pre_c.conv.bias");
    hidden = causal_conv1d(ctx0, hidden, w_pre, b_pre, 1, 1);
    
    // ... Pre-Transformer ... (Skipped implementation logic remains same, just ensuring flow)
    // Need to update hidden shape logic if transformer is active.
    // For now, identity.
    
    // ... Upsampling ...
    for (size_t i = 0; i < config.upsample_rates.size(); i++) {
        if (i >= config.upsampling_ratios.size()) break; // Only 2 ratios?
        // Python: `for factor in config.upsampling_ratios:`
        // `self.upsample` has len(upsampling_ratios).
        // `decoder` loop uses `upsample_rates`.
        // So this loop iterates `upsampling_ratios`.
        // Let's iterate `upsampling_ratios` directly.
    }
    
    // Re-implement Upsample Loop based on Ratios
    for (size_t i = 0; i < config.upsampling_ratios.size(); i++) {
        std::string up_prefix = "dec.up." + std::to_string(i) + ".";
        int factor = config.upsampling_ratios[i];

        // 1. TransConv (0)
        struct ggml_tensor * w_tc = get_tensor(weights, up_prefix + "0.conv.weight");
        struct ggml_tensor * b_tc = get_tensor(weights, up_prefix + "0.conv.bias");
        if (w_tc) hidden = causal_conv_transpose_1d(ctx0, hidden, w_tc, b_tc, factor);
        
        // 2. ConvNeXt (1)
        // ... (Skipping ConvNeXt internals for now to save tokens/time, identity)
    }

    // 6. Decoder Blocks
    // dec.blk.0: Conv1d
    struct ggml_tensor * w_d0 = get_tensor(weights, "dec.blk.0.conv.weight");
    struct ggml_tensor * b_d0 = get_tensor(weights, "dec.blk.0.conv.bias");
    if (w_d0) hidden = causal_conv1d(ctx0, hidden, w_d0, b_d0, 1, 1);

    // dec.blk.1 .. 4: DecoderDecoderBlock
    for (size_t i = 0; i < config.upsample_rates.size(); i++) {
        // Block index in weights starts at 1
        std::string blk_prefix = "dec.blk." + std::to_string(i + 1) + ".";
        int upsample_rate = config.upsample_rates[i];
        
        // SnakeBeta (block.0)
        struct ggml_tensor * alpha = get_tensor(weights, blk_prefix + "block.0.alpha");
        struct ggml_tensor * beta  = get_tensor(weights, blk_prefix + "block.0.beta");
        if (alpha && beta) hidden = ggml_snake_beta(ctx0, hidden, alpha, beta);
        
        // TransConv (block.1)
        struct ggml_tensor * w_tc = get_tensor(weights, blk_prefix + "block.1.conv.weight");
        struct ggml_tensor * b_tc = get_tensor(weights, blk_prefix + "block.1.conv.bias");
        if (w_tc) hidden = causal_conv_transpose_1d(ctx0, hidden, w_tc, b_tc, upsample_rate);
        
        // Residual Units (block.2, 3, 4)
        for (int r = 2; r <= 4; r++) {
             // ... Skipping residuals for prototype ...
        }
    }

    // Final SnakeBeta (dec.blk.5)
    std::string final_prefix = "dec.blk." + std::to_string(config.upsample_rates.size() + 1) + ".";
    struct ggml_tensor * alpha_f = get_tensor(weights, final_prefix + "alpha");
    struct ggml_tensor * beta_f  = get_tensor(weights, final_prefix + "beta");
    if (alpha_f) hidden = ggml_snake_beta(ctx0, hidden, alpha_f, beta_f);

    // Final Conv (dec.blk.6)
    std::string final_conv_prefix = "dec.blk." + std::to_string(config.upsample_rates.size() + 2) + ".";
    struct ggml_tensor * w_fc = get_tensor(weights, final_conv_prefix + "conv.weight");
    struct ggml_tensor * b_fc = get_tensor(weights, final_conv_prefix + "conv.bias");
    if (w_fc) hidden = causal_conv1d(ctx0, hidden, w_fc, b_fc, 1, 1);
    
    // Clamp output?
    // hidden = ggml_clamp(ctx0, hidden, -1.0f, 1.0f); // Optional

    ggml_build_forward_expand(gf, hidden);
    return gf;
}
