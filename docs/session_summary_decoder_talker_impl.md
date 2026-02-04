# Session Summary: Qwen3-TTS C++ Implementation (2026-01-31)

## Overview
This session focused on implementing the core C++ components for the Qwen3-TTS inference pipeline, specifically the **Mimi Audio Decoder** and the **Talker (Transformer)** infrastructure using `llama.cpp/ggml`.

## Accomplishments

### 1. Audio Decoder Implementation (`Qwen3AudioDecoder`)
- **Graph Construction**: Implemented the structural graph in `cpp/qwen3_audio_decoder.cpp` following the Mimi architecture:
    - **Quantizer**: Implemented lookups for the Split Residual Vector Quantizer (Semantic + Acoustic) using `ggml_get_rows`.
    - **Pre-Net**: Integrated the Pre-Convolution and Transformer skeleton.
    - **Upsampling**: Implemented the Transposed Convolution blocks for Progressives Upsampling.
    - **Snake Decoder**: Integrated the `SnakeBeta` activation function (custom GGML composition).
- **Weight Loading**: Updated the loading logic to handle renamed GGUF tensors (e.g., `dec.q.sem`, `dec.up`, `dec.blk`).

### 2. Talker Infrastructure (`Qwen3Talker`)
- **Model Loading**: Implemented a full weight loader that filters and maps GGUF tensors for the Talker (Transformer) component.
- **Memory Management**: Optimized GGML context sizes for loading the ~8GB combined model (allocated 10GB for Talker, 2GB for Decoder).

### 3. Build & Integration
- **Build System**: Updated `cpp/CMakeLists.txt` to include `qwen3_talker.cpp` and linked necessary backends.
- **CLI Entry Point**: Updated `cpp/main.cpp` to wire the `Talker -> Decoder` pipeline.
- **Windows Compatibility**: Fixed compilation issues with MSVC (e.g., `fopen` vs `fopen_s` warnings, `dllimport` conflicts, and 64-bit offsets).

### 4. Debugging & Tooling
- **GGUF Inspection**: Created `inspect_gguf.py` to verify local GGUF tensor names and shapes, revealing that the conversion script used specific shortenings (e.g., `_cb` for codebook).
- **Shape Tracing**: Added debug prints to `build_graph` to identify broadcasting failures in `ggml_add` and `ggml_mul`.

## Technical Challenges & Solutions

| Challenge | Solution |
| :--- | :--- |
| **`ggml_rope` Signature** | The local `ggml` version required 5 arguments instead of 6. Updated the call to match. |
| **OOM on Load** | Increased `ggml_init` memory pool from 4GB to 10GB to accommodate the full FP32 model. |
| **Opaque `ggml_cgraph`** | Switched from direct member access (`gf->nodes`) to the `ggml_graph_node(gf, -1)` accessor. |
| **DLL Export Conflict** | Renamed local helper `ggml_new_f32` to `make_f32_tensor` to avoid shadowing symbols in `ggml.dll`. |
| **Broadcasting Assertions** | Implemented `ggml_reshape_2d` for bias vectors in `causal_conv1d` to ensure compatibility with `(T, C)` tensors. |

## Current Status
- **Build**: Success (MSVC + Ninja).
- **Loading**: Success (Talker: 333 tensors, Decoder: 371 tensors).
- **Execution**: The pipeline runs but currently triggers a `GGML_ASSERT(ggml_can_repeat)` during the acoustic quantization loop.

## Next Steps
1. **Fix Shape Mismatch**: Resolve the remaining broadcasting issue in the acoustic quantizer addition loop.
2. **Transformer Logic**: Complete the attention/MLP implementation in the decoder's pre-transformer.
3. **Autoregressive Generation**: Implement the Talker's generation loop to replace the current mock output.
