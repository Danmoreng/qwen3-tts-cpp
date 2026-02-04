# Qwen3-TTS.cpp

> **Note**: This project is an experiment in AI-assisted software development. The core C++ implementation and architecture mapping were developed by an AI agent through iterative loops, exploring how effectively agents can translate complex transformer architectures into optimized GGML-based C++.

A high-performance C++ implementation of the **Qwen3-TTS** model using `llama.cpp` and `ggml`. This project aims to provide standalone, dependency-free inference for state-of-the-art text-to-speech on Windows, Linux, and macOS.

> [!WARNING]  
> **Work in Progress**: This project is currently in early development (Phase 1). The audio decoder is being implemented, and the full pipeline is not yet functional.

## Features
- **Pure C++**: No Python runtime required for inference.
- **GGML Backend**: Leverages `llama.cpp` for optimized CPU and CUDA performance.
- **GGUF Support**: Custom conversion tools to handle Qwen3's multi-model architecture.
- **Windows Optimized**: Dedicated PowerShell build scripts and MSVC support.
- **Multimodal RoPE**: Support for 3D position IDs used in the Qwen3 architecture.

## Repository Structure
- `cpp/`: Main C++ source code, CMake configuration, and build scripts.
- `docs/`: Design documents, technical notes, and development plans.
- `tools/`: Python scripts for GGUF model conversion and weight inspection.
- `python_ref/`: (Submodule) The original [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) reference implementation.
- `third_party/`:
    - `llama.cpp`: The core tensor library.
    - `qwen3-asr.cpp`: Reference implementation for Qwen3 ASR logic.

## Getting Started

### Prerequisites
- CMake (>= 3.14)
- Visual Studio 2022 (Windows) or GCC/Clang (Linux/macOS)
- Ninja (optional, but recommended)

### Building (Windows)
```powershell
cd cpp
.\build.ps1 -ExecutionPolicy Bypass
```

### Building (Linux/macOS)
```bash
cd cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON  # Optional
make -j
```

## Documentation
- [Development Plan](docs/development_plan.md) - Project roadmap and tasks.
- [Technical Notes](docs/technical_notes.md) - Deep dive into Windows builds and architecture quirks.
- [Architecture Analysis](docs/analysis_qwen3_tts_pipeline.md) - Detailed breakdown of the Qwen3-TTS pipeline.

## License
The C++ code in this repository is licensed under the MIT License. Please refer to the `python_ref/` submodule for the license governing the original Qwen3-TTS model weights and architecture.

## Acknowledgments
- [Alibaba Qwen Team](https://github.com/QwenLM) for the original Qwen3-TTS models.
- [ggerganov](https://github.com/ggerganov/llama.cpp) for the incredible `llama.cpp` ecosystem.
- [predict-woo](https://github.com/predict-woo/qwen3-asr.cpp) for the ASR reference implementation.