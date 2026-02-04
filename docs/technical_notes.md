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
- **SnakeBeta**: Implemented via `ggml` composition (`sin`, `sqr`, `add`, `div`) as a temporary measure until a custom CUDA kernel is optimized.

## Runtime Quirks
- **Shared Libraries**: The project builds `llama.cpp` as shared libraries (`llama.dll`, `ggml.dll`). The build script automatically deploys these to the build root for easier execution.
- **Python Reference**: Used for weight extraction and verification. Reference scripts should use `device="cpu"` and `dtype=torch.float32` for maximum compatibility in non-CUDA Python environments.
