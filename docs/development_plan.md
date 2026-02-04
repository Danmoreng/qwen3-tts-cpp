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
**Status:** Pending.

*   [ ] **Task 2.1: Implement Main Talker Class**
    *   **File:** `cpp/qwen3_talker.cpp` (New)
    *   **Description:** Implement the 1.7B Llama-based transformer.
    *   **Key Challenges:**
        *   **Multimodal RoPE:** Implement 3D position ID logic (Text, Audio, Width/Height).
        *   **KV Cache:** Manage state for autoregressive generation.
*   [ ] **Task 2.2: Implement Code Predictor**
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
