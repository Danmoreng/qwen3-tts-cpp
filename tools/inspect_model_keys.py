from transformers import AutoModel
import torch
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

print(f"Loading model from {model_id}...")
# Use Qwen3TTSModel.from_pretrained wrapper which handles registration
# We use device_map="cpu" to avoid CUDA issues
model_wrapper = Qwen3TTSModel.from_pretrained(model_id, device_map="cpu")
model = model_wrapper.model

print("\n--- Tensor Keys ---")
keys = list(model.state_dict().keys())
# Print first 20 and last 20 to get an idea, plus filter for specific components
for k in keys[:20]:
    print(k)
print("...")
for k in keys[-20:]:
    print(k)

print(f"\nTotal keys: {len(keys)}")

# Check for specific components
print("\n--- Component Check ---")
components = ["talker", "speaker_encoder", "code_predictor"]
for comp in components:
    matches = [k for k in keys if comp in k]
    print(f"{comp}: {len(matches)} tensors found. First match: {matches[0] if matches else 'None'}")

print("\n--- Speech Tokenizer Check ---")
if model.speech_tokenizer:
    print("Speech Tokenizer found in model!")
    st_keys = list(model.speech_tokenizer.model.state_dict().keys())
    print(f"Total Speech Tokenizer keys: {len(st_keys)}")
    for k in st_keys[:20]:
        print(f"ST: {k}")
else:
    print("Speech Tokenizer NOT found in model object.")
