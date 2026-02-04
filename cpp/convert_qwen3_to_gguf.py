import argparse
import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
import gguf

def convert(model_id, output_path):
    print(f"Loading model: {model_id}")
    # Load the wrapper to get all components
    wrapper = Qwen3TTSModel.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    model = wrapper.model
    tokenizer = wrapper.processor
    
    print("Model loaded. Starting GGUF conversion...")
    
    gguf_writer = gguf.GGUFWriter(output_path, "qwen3-tts")
    
    # --- Arch Params ---
    # We can save config params as KV pairs
    cfg = model.config.talker_config
    gguf_writer.add_uint32("talker.vocab_size", cfg.vocab_size)
    gguf_writer.add_uint32("talker.context_length", cfg.max_position_embeddings)
    gguf_writer.add_uint32("talker.embedding_length", cfg.hidden_size)
    gguf_writer.add_uint32("talker.block_count", cfg.num_hidden_layers)
    gguf_writer.add_uint32("talker.feed_forward_length", cfg.intermediate_size)
    gguf_writer.add_uint32("talker.attention.head_count", cfg.num_attention_heads)
    gguf_writer.add_uint32("talker.attention.head_count_kv", cfg.num_key_value_heads)
    
    # Save Decoder Config if available
    if hasattr(model, "speech_tokenizer"):
        dec_cfg = model.speech_tokenizer.model.config.decoder_config
        gguf_writer.add_uint32("decoder.latent_dim", dec_cfg.latent_dim)
        gguf_writer.add_uint32("decoder.codebook_dim", dec_cfg.codebook_dim)
        gguf_writer.add_uint32("decoder.num_quantizers", dec_cfg.num_quantizers)

    # --- Weights ---
    state_dict = model.state_dict()
    
    # Add Speech Tokenizer weights manually since they are separate
    if hasattr(model, "speech_tokenizer"):
        st_state_dict = model.speech_tokenizer.model.state_dict()
        for k, v in st_state_dict.items():
            # Prefix with 'speech_tokenizer.' to distinguish
            state_dict[f"speech_tokenizer.{k}"] = v

    print(f"Total tensors to process: {len(state_dict)}")

    for name, data in state_dict.items():
        # Standardize name for C++ loading
        # 1. Talker
        if name.startswith("talker.model."):
            # Map standard Llama-like keys
            # talker.model.layers.0.self_attn.q_proj.weight -> talker.blk.0.attn_q.weight
            new_name = name.replace("talker.model.", "talker.")
            new_name = new_name.replace("layers.", "blk.")
            new_name = new_name.replace(".self_attn.q_proj", ".attn_q")
            new_name = new_name.replace(".self_attn.k_proj", ".attn_k")
            new_name = new_name.replace(".self_attn.v_proj", ".attn_v")
            new_name = new_name.replace(".self_attn.o_proj", ".attn_output")
            new_name = new_name.replace(".mlp.gate_proj", ".ffn_gate")
            new_name = new_name.replace(".mlp.up_proj", ".ffn_up")
            new_name = new_name.replace(".mlp.down_proj", ".ffn_down")
            new_name = new_name.replace(".input_layernorm", ".attn_norm")
            new_name = new_name.replace(".post_attention_layernorm", ".ffn_norm")
            # Norms
            new_name = new_name.replace("talker.norm.weight", "talker.output_norm.weight")
            
        elif name.startswith("talker.code_predictor."):
            new_name = name.replace("talker.code_predictor.model.", "predictor.")
            new_name = new_name.replace("layers.", "blk.")
            # ... apply similar mappings for predictor ...
            
        elif name.startswith("speech_tokenizer."):
            # speech_tokenizer.decoder. -> decoder.
            new_name = name.replace("speech_tokenizer.", "")
            
            # Shorten long names for GGML_MAX_NAME (64 chars) compatibility
            # encoder.quantizer.semantic_residual_vector_quantizer -> dec.q.sem
            new_name = new_name.replace("encoder.quantizer.semantic_residual_vector_quantizer", "dec.q.sem")
            new_name = new_name.replace("encoder.quantizer.acoustic_residual_vector_quantizer", "dec.q.ac")
            
            new_name = new_name.replace("decoder.upsample.", "dec.up.")
            new_name = new_name.replace("decoder.pre_transformer.", "dec.pre_tf.")
            new_name = new_name.replace("decoder.pre_conv.", "dec.pre_c.")
            new_name = new_name.replace("decoder.decoder.", "dec.blk.")
            
            # Codebook specific shortenings
            new_name = new_name.replace("layers.", "l.")
            new_name = new_name.replace("codebook", "cb")
            new_name = new_name.replace("cluster_usage", "usage")
            new_name = new_name.replace("embedding_sum", "sum")
            new_name = new_name.replace("initialized", "init")
            
        else:
            new_name = name # Keep as is (speaker_encoder etc)

        # Convert to FP16 or FP32 for GGUF
        # Using FP32 for safety now, user can quantize later
        data_np = data.detach().cpu().numpy().astype(np.float32)
        
        # Add to GGUF
        gguf_writer.add_tensor(new_name, data_np)

    print(f"Writing GGUF to {output_path}...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Model ID or path")
    parser.add_argument("--output", type=str, default="qwen3_tts_full.gguf", help="Output GGUF file")
    args = parser.parse_args()
    
    convert(args.model, args.output)
