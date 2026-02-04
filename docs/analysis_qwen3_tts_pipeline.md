# Qwen3-TTS Pipeline Analysis & C++ Implementation Guide

## 1. High-Level Overview

The Qwen3-TTS system is a two-stage text-to-speech pipeline:
1.  **Auto-Regressive (AR) Transformer ("Talker")**: Converts input text (and optional reference audio prompts) into discrete audio codes (tokens). It uses a hierarchical generation approach where a main transformer predicts the first layer of codes, and a smaller "Code Predictor" transformer predicts the subsequent layers of codes (Refinement).
2.  **Audio Decoder ("Speech Tokenizer")**: Converts the generated discrete codes back into a continuous audio waveform. The 12Hz version (likely the primary target) uses a `Mimi`-based architecture with a custom decoder involving a small internal transformer and upsampling blocks.

To implement this in C++/CUDA without Python dependencies, you need to replicate these three distinct neural network components and the orchestration logic connecting them.

---

## 2. Component Details

### A. Text Tokenizer
*   **Role**: Converts input text strings into integer `input_ids`.
*   **Implementation**: Likely a BPE-based tokenizer (HuggingFace `AutoProcessor`/`AutoTokenizer`).
*   **C++ Requirement**: Use `sentencepiece` or a port of `tokenizers` library. You need to load the `tokenizer.json` or `tokenizer.model` file.
*   **Special Logic**: 
    *   Input text is wrapped in ChatML format: `<|im_start|>assistant\n{text}<|im_end|>\n`.
    *   Special tokens are added for TTS control (`tts_bos`, `tts_eos`, `tts_pad`, `codec_think_start`, etc.).

### B. Speaker Encoder (Optional / "Base" Model only)
*   **Role**: Extracts a fixed-size speaker embedding vector from a reference audio file (for zero-shot voice cloning).
*   **Architecture**: ECAPA-TDNN (Time Delay Neural Network).
    *   Input: Mel Spectrogram (requires Audio STFT preprocessing).
    *   Layers: 1D Convolutions, Res2Net blocks, Squeeze-Excitation, Attentive Statistical Pooling.
*   **C++ Requirement**: Implement Mel Spectrogram calculation (STFT) and the TDNN model.

### C. The "Talker" (Main Inference Engine)
This is the core logic `Qwen3TTSForConditionalGeneration`.

#### 1. Input Processing
*   **Embeddings**:
    *   **Text Embeddings**: Standard lookup table.
    *   **Speaker Embeddings**: Linear projection of the extracted speaker vector.
    *   **Codec Embeddings**: Embeddings for the audio codes. Since there are multiple codebooks (layers), they are projected/summed.
*   **Prompt Construction**: 
    *   Concatenates [Speaker Embed, Text Embeds, Prompt Audio Code Embeds].
    *   Uses a "Non-streaming mode" logic where text is fully available for attention.

#### 2. Main Transformer (`Qwen3TTSTalkerModel`)
*   **Type**: Causal Decoder-only Transformer (Llama-like).
*   **Key Features**:
    *   **Normalization**: RMSNorm.
    *   **Activation**: SwiGLU.
    *   **Attention**: Multi-Head Attention (MHA) or GQA (Grouped Query Attention) depending on config.
    *   **Positional Embeddings**: **Multimodal RoPE** (Rotary Positional Embeddings). 
        *   *Crucial Detail*: It handles "temporal", "height", "width" dimensions. For TTS, this effectively handles the alignment between text and audio. The `get_rope_index` function calculates 3D position IDs.
    *   **KV Cache**: Standard Key-Value cache for efficient autoregressive generation.

#### 3. Code Predictor (`Qwen3TTSTalkerCodePredictorModel`)
*   **Role**: Predicts code layers 1..N given layer 0.
*   **Architecture**: A smaller, standard Causal Transformer.
*   **Integration**:
    *   Inside the main generation loop, after the Main Transformer predicts Code 0.
    *   The Code Predictor runs autoregressively *per step* or in parallel for the sub-codes (depending on the `forward` implementation, usually it's a small chain).
    *   In `Qwen3TTSTalkerForConditionalGeneration`, the `forward` method shows: `inputs_embeds` for the main model at step `t` is the sum of embeddings of ALL codes generated at step `t-1`.

### D. The Speech Tokenizer (Decoder)
This converts the sequence of codes into audio. Focusing on the **12Hz (v2)** version:

#### 1. Quantizer (`SplitResidualVectorQuantizer`)
*   **Input**: `(Batch, Sequence, Num_Quantizers)` integers.
*   **Operation**: Look up vectors in codebooks. Sum the vectors from all quantizers. 
    *   Uses a split logic: First quantizer might have a separate projection.

#### 2. Pre-Net
*   **Conv1D**: Initial processing.
*   **Transformer**: `Qwen3TTSTokenizerV2DecoderTransformerModel`. Yes, there is a transformer *inside* the tokenizer decoder. It uses RoPE and Sliding Window Attention.

#### 3. Upsampler
*   **Structure**: Series of Upsampling blocks.
*   **Blocks**: `Qwen3TTSTokenizerV2CausalTransConvNet` (Transposed Conv 1D) + `Qwen3TTSTokenizerV2ConvNeXtBlock`.

#### 4. Final Decoder
*   **Blocks**: `Qwen3TTSTokenizerV2DecoderDecoderBlock`.
*   **Activation**: **SnakeBeta** (`x + (1/b)*sin^2(a*x)`). This is non-standard and must be implemented manually in CUDA/C++.

---

## 3. Data Flow & Implementation Steps

To execute `generate_voice_clone`:

1.  **Preprocessing (C++)**:
    *   Load Reference Audio -> STFT -> Mel Spectrogram -> **Speaker Encoder** -> `Speaker Embedding`.
    *   Load Reference Audio -> **Speech Tokenizer Encode** -> `Reference Codes`.
    *   Tokenize Input Text -> `Input IDs`.

2.  **Prompt Assembly**:
    *   Construct `input_embeds`: `[Speaker Emb] + [Text Emb] + [Reference Code Emb]`.
    *   Create 3D Position IDs for RoPE.

3.  **Autoregressive Loop (Main Transformer)**:
    *   **Input**: Current sequence embeddings.
    *   **Forward**: Main Transformer Block.
    *   **Output**: Logits for **Code 0** (First Quantizer).
    *   **Sample**: Select Code 0 (Greedy/Top-k/Top-p).

4.  **Sub-Generation (Code Predictor)**:
    *   **Input**: Embedding of Code 0 + Past Hidden State.
    *   **Forward**: Code Predictor Transformer (runs for `Num_Quantizers - 1` steps).
    *   **Sample**: Select Codes 1..N.
    *   **Update**: Sum embeddings of Codes 0..N to form the input for the *next* step of the Main Transformer.

5.  **Post-Processing (Speech Tokenizer Decode)**:
    *   Collect all generated codes `(T, Num_Quantizers)`.
    *   **Quantize**: Convert to vectors.
    *   **Decode**: Run through Pre-Conv -> Decoder Transformer -> Upsamplers -> Snake Decoder.
    *   **Output**: Float array (Audio Waveform).

---

## 4. Key Implementation Challenges for C++/CUDA

1.  **Multimodal RoPE**: Implementing the specific 3D position ID logic and applying rotary embeddings correctly to Q/K vectors.
2.  **Snake Activation**: Custom CUDA kernel needed for `SnakeBeta` activation.
3.  **Nested Generation**: The generation loop is not a simple "next token" prediction. It's "predict token 0, then run sub-model to predict tokens 1..N, then combine".
4.  **Model Loading**: You will need to parse the SafeTensors/PyTorch weights and map them to your C++ structs. The parameter names in `modeling_qwen3_tts.py` are your map.
5.  **FFT/STFT**: For the Speaker Encoder (if used), you need a signal processing library (like cuFFT or a simple STFT implementation).

## 5. File Mapping

| Python Module | Logic | C++ Goal |
| :--- | :--- | :--- |
| `qwen_tts/inference/qwen3_tts_model.py` | API Entry, Prompting | `main.cpp` / `pipeline.cpp` |
| `qwen_tts/core/models/modeling_qwen3_tts.py` | Talker & Code Predictor | `model_talker.cpp`, `model_predictor.cpp` |
| `qwen_tts/core/models/modeling_qwen3_tts.py` | Speaker Encoder (ECAPA) | `model_speaker_encoder.cpp` |
| `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` | Speech Tokenizer (Mimi+Snake) | `model_audio_decoder.cpp` |
| `transformers` (library) | Tokenizer (BPE) | `tokenizers.cpp` |
