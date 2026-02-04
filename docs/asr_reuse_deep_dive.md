# Deep Dive: Reusing qwen3-asr.cpp Logic

## Motivation
The `qwen3-asr.cpp` project implements a production-grade inference engine for the ASR variant of Qwen3. Since both ASR and TTS share the same base Transformer architecture (Llama-style), we can port highly optimized infrastructure components to Qwen3-TTS.

## Reusable Components

### 1. Transformer Architecture (`TextDecoder`)
- **Logic**: The `build_graph` in ASR's `TextDecoder` correctly implements the Llama block (RMSNorm -> Attention -> Res -> RMSNorm -> MLP -> Res).
- **Benefit**: Using this as a baseline ensures we follow proven `ggml` patterns for this specific model family.
- **Porting**: Will be adapted into `Qwen3TalkerLLM`.

### 2. Modern KV Cache Management
- **Implementation**: ASR uses 3D tensors (`head_dim`, `n_kv_head`, `n_ctx`) per layer for KV storage.
- **Backend Integration**: Correctly utilizes `ggml_backend_alloc_ctx_tensors` to ensure the cache stays on the correct device (GPU/CPU).
- **Benefit**: Essential for autoregressive generation in the TTS Talker.

### 3. Unified Backend Orchestration
- **API**: ASR uses `ggml_backend_sched` (Scheduler).
- **Why**: This automatically handles:
    - Splitting large graphs between multiple GPUs or CPU/GPU.
    - Managing scratch buffers for intermediate nodes.
    - Handling tensor synchronization between backends.
- **Action**: TTS will adopt the Scheduler-based execution model instead of raw `ggml_graph_compute`.

## Non-Reusable Components
- **Audio Encoder**: ASR's Mel-based encoder is incompatible with Mimi's VQ-based decoder.
- **Audio Injection**: The coordinate-based Multimodal RoPE in Qwen3-TTS is more complex than the simple concatenation/injection used in ASR.

## Implementation Plan
1.  **Phase 2.1**: Extract `text_decoder.h/cpp` logic into `cpp/qwen3_talker_llm.h/cpp`.
2.  **Refactor**: Rename namespaces and structures to fit the TTS project.
3.  **Enhance**: Modify the position embedding logic to support the 3D IDs required by the Qwen3-TTS paper.
