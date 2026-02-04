# Technical Notes: Qwen3-TTS C++ Implementation

## Build Environment (Windows)
- **Compiler**: Visual Studio 2022 (MSVC) is required for the C++ build.
- **PowerShell**: Use `cpp/build.ps1` to build. It automatically handles Visual Studio environment variable loading (`vcvars64.bat`) via the `Import-VSEnv` function.
- **Generator**: Ninja is the preferred generator (`-G Ninja`).
- **CUDA**: Supported via `-DGGML_CUDA=ON`. Currently tested with CUDA v13.0.

## Implementation Details

### Model Architecture
- **Mimi Decoder**: Uses a CNN-based architecture with `SnakeBeta` activations and an upsampling chain.
- **Talker**: A 1.7B Llama-based transformer.
- **RoPE**: Implements Multimodal RoPE (3D position IDs) for text and audio integration.

### Custom GGUF Format
- **Tensor Names**: Shortened to < 64 characters to comply with GGUF standards (e.g., `decoder.quantizer...` -> `dec.q...`).
- **Large File Support**: Loading the ~8GB model on Windows requires using `_fseeki64` and `_ftelli64` to bypass 32-bit `long` limits.

### Activation Functions
- **SnakeBeta**: Implemented via `ggml` composition (`sin`, `sqr`, `add`, `div`). Requires explicit `ggml_repeat` for broadcasting constants (like epsilon) and the final additive term.

### Convolution Workarounds
- **F16 Restriction**: The local `ggml_conv_1d` implementation hardcodes `GGML_TYPE_F16` for its internal `im2col` operation. This triggers assertions if the CPU/Backend doesn't support F16 im2col kernels for F32 weights. 
- **Solution**: Implemented `fixed_ggml_conv_1d` which explicitly uses `GGML_TYPE_F32` for `im2col`.
- **Batch Reshaping**: `ggml_conv_1d` and `ggml_conv_transpose_1d` expect 3D/4D inputs. Tensors must be reshaped from `(T, C)` to `(T, C, 1)` before processing.

## Runtime Quirks
- **Memory Context**: The decoder graph, especially the 4-stage upsampling chain, consumes significant memory during graph expansion. `ctx_compute` should be at least 2GB for a sequence length of 100 frames.
- **Transposed Tensors**: Must be made contiguous via `ggml_cont` after `ggml_transpose` to avoid shape assertions in downstream ops.
