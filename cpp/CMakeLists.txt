cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure llama.cpp options before adding it
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

# Enable CUDA if available (user must pass -DGGML_CUDA=ON to cmake)
if (GGML_CUDA)
    add_definitions(-DGGML_USE_CUDA)
endif()

# Add llama.cpp as a subdirectory
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/CMakeLists.txt")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp")
else()
    message(FATAL_ERROR "llama.cpp submodule not found at ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp")
endif()

# Define our executable
add_executable(qwen3-tts-cli 
    main.cpp
    qwen3_audio_decoder.cpp
    qwen3_talker.cpp
)

# Link against llama (high level) and ggml (low level for custom ops)
if (TARGET common)
    target_link_libraries(qwen3-tts-cli PRIVATE common)
endif()

target_link_libraries(qwen3-tts-cli PRIVATE llama ggml)

# Include paths
target_include_directories(qwen3-tts-cli PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp"
)
