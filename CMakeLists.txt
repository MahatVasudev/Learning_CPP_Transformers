cmake_minimum_required(VERSION 3.15)
project(SampleProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(ENABLE_SANITIZERS "Build with ASan + UBSan" ON)

add_library(core_lib STATIC
  src/core/tensor.cpp
  src/core/utils
  src/core/errors.cpp
  src/core/random.cpp
  src/core/tokenized_tensor.cpp
)

add_library(nn_lib STATIC
 src/nn/initialization.cpp
 src/nn/embedding.cpp
 src/nn/layer_norm.cpp
 src/nn/linear.cpp
 src/nn/module.cpp
 src/nn/positional_encoding.cpp
 src/nn/utils.cpp
)

add_library(transformers_lib STATIC
 src/nn/transformers/decoder.cpp
 src/nn/transformers/decoder_gpt.cpp
 src/nn/transformers/encoder.cpp
 src/nn/transformers/feedforward.cpp
 src/nn/transformers/multiheadattention.cpp
 src/nn/transformers/transformer.cpp
)

add_library(activations_lib STATIC
  src/nn/activations/relu.cpp
  src/nn/activations/softmax.cpp
)

add_library(ops_lib STATIC
  src/ops/cross_entropy.cpp
  src/ops/matmul.cpp
  src/ops/utils.cpp
)

add_library(testing_lib STATIC
  src/testing/gradcheck.cpp
)

add_library(test_lib STATIC
  src/tests/test_activations.cpp
  src/tests/test_cross_entropy.cpp
  src/tests/test_embedding.cpp
  src/tests/test_layernorm.cpp
  src/tests/test_linear.cpp
  src/tests/test_matmul.cpp
  src/tests/test_tensor.cpp
  src/tests/test_utils_and_initialization.cpp
  src/tests/test_positional_encoding.cpp
  src/tests/test_feedforward.cpp
  src/tests/test_module_contract.cpp
)

# dependencies, bottom to top (no cycles)
target_link_libraries(ops_lib          PUBLIC core_lib)
target_link_libraries(nn_lib           PUBLIC ops_lib core_lib)
target_link_libraries(transformers_lib PUBLIC nn_lib core_lib activations_lib)
target_link_libraries(activations_lib  PUBLIC nn_lib core_lib)
target_link_libraries(testing_lib      PUBLIC core_lib)
target_link_libraries(test_lib         PUBLIC core_lib model_lib ops_lib nn_lib testing_lib)

# settings shared by every library
foreach(lib core_lib nn_lib transformers_lib activations_lib ops_lib testing_lib test_lib)
  target_include_directories(${lib} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(${lib} PRIVATE -Wall -Wextra)
  if(ENABLE_SANITIZERS)
    target_compile_options(${lib} PUBLIC -g -fsanitize=address,undefined)
    target_link_options(${lib} PUBLIC -fsanitize=address,undefined)
  endif()
endforeach()

add_executable(main_app main.cpp)
target_link_libraries(main_app PRIVATE test_lib)

# ctest, and a `check` target that rebuilds first
enable_testing()
add_test(NAME all_tests COMMAND main_app --test)
add_custom_target(check COMMAND main_app --test DEPENDS main_app)
