import gguf
import sys

def inspect(path):
    print(f"Opening {path}...")
    reader = gguf.GGUFReader(path)
    
    print(f"Tensors: {len(reader.tensors)}")
    
    for tensor in reader.tensors:
        name = tensor.name
        # if "dec" in name or "quant" in name:
        print(f"{name} | Shape: {tensor.data.shape}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        print("Usage: python inspect_gguf.py <file.gguf>")
