# Qwen3-TTS C++ Implementation Development Plan

## Objective
Develop a high-performance, standalone C++ inference engine for Qwen3-TTS using `llama.cpp` / `ggml` backend. The system must run on Windows (CUDA/CPU) without Python runtime dependencies.

## Phase 1: Audio Decoder (Speech Tokenizer)
**Goal:** Convert discrete audio codes (indices) into continuous audio waveforms.
**Status:** Complete.

*   [x] **Task 1.1: Implement Decoder Graph**
    *   **File:** `cpp/qwen3_audio_decoder.cpp`
    *   **Description:** Implement `Qwen3AudioDecoder::build_graph` reconstructing the `Mimi` architecture.
    *   **Details:**
        *   Implement `SplitResidualVectorQuantizer` lookup (Codebook 0 + Codebooks 1..N).
        *   Implement `Pre-Conv` and `Pre-Transformer` (with RoPE).
        *   Implement Upsampling Chain (`TransposedConv1d` + `ConvNeXt` blocks).
        *   Implement `SnakeBeta` activation function using `ggml` primitives.
*   [x] **Task 1.2: Verification (Internal Test)**
    *   **Files:** `cpp/main.cpp`
    *   **Description:** Successfully executed the graph with mock input codes, producing the expected upsampled waveform shape.

## Phase 2: Talker (Autoregressive Transformer)
**Goal:** Convert text inputs into audio codes.
**Status:** In Progress (Architectural Refinement).

*   [x] **Task 2.1: Implement Qwen3TalkerLLM (Transformer Core)**
    *   **File:** `cpp/qwen3_talker_llm.cpp` (New)
    *   **Strategy:** Port the `TextDecoder` architecture from `qwen3-asr.cpp`.
    *   **Details:**
        *   Adopt `ggml-backend` and `ggml-scheduler` for hardware-agnostic execution.
        *   Implement high-performance **KV Cache** management (per-layer 3D tensors).
        *   Reuse the Llama-style `build_graph` (RMSNorm, SwiGLU MLP).
*   [ ] **Task 2.2: Implement Multimodal RoPE**
    *   **Status:** In Progress.
    *   **Description:** Extend the transformer core to support 3D position IDs.
    *   **Current State:** 3D ID generation and slicing logic implemented. Debugging `ggml_view_3d` boundary assertions in autoregressive loop.
*   [ ] **Task 2.3: Implement Code Predictor**
    *   **Description:** Implement the small "refinement" transformer that predicts codes 1..N from code 0.
    *   **Integration:** This runs *inside* the main generation loop.

## Phase 3: Pipeline Integration & CLI
**Goal:** End-to-end Text-to-Speech application.
**Status:** Pending.

*   [ ] **Task 3.1: Tokenizer Integration**
    *   **Description:** Integrate `sentencepiece` or similar to handle ChatML text tokenization.
*   [ ] **Task 3.2: Main Pipeline**
    *   **File:** `cpp/main.cpp`
    *   **Description:** Connect Tokenizer -> Talker -> Decoder.
    *   **Features:**
        *   Load GGUF model.
        *   Accept text input.
        *   Generate WAV output.

## Phase 4: Optimization & Windows Polish
**Goal:** Ensure smooth user experience on Windows.
**Status:** Pending.

*   [ ] **Task 4.1: CUDA Optimization**
    *   **Description:** Ensure `ggml-cuda` is used efficiently (layer offloading).
    *   **Optional:** Custom CUDA kernel for `SnakeBeta` if performance is bottlenecked.
*   [ ] **Task 4.2: Packaging**
    *   **Description:** Bundle DLLs and executables for easy distribution.
