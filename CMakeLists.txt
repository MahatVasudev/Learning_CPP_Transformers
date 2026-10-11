cmake_minimum_required(VERSION 3.15)
project(SampleProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(ENABLE_SANITIZERS "Build with ASan + UBSan" ON)

add_library(tensor_lib STATIC src/functional/tensor.cpp)
add_library(utils_lib STATIC src/utils/utils.cpp)
add_library(functional_lib STATIC
  src/functional/activations.cpp
  src/functional/ops.cpp
  src/functional/gradcheck.cpp
  src/functional/module.cpp
  src/functional/cross_entropy.cpp
  src/functional/positional_encoding.cpp
)
add_library(model_lib STATIC
  src/model/model.cpp src/model/decoder.cpp src/model/encoder.cpp)
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
)

# dependencies, bottom to top (no cycles)
target_link_libraries(utils_lib      PUBLIC tensor_lib)
target_link_libraries(functional_lib PUBLIC tensor_lib utils_lib)
target_link_libraries(model_lib      PUBLIC functional_lib)
target_link_libraries(test_lib       PUBLIC model_lib functional_lib utils_lib tensor_lib)

# settings shared by every library
foreach(lib tensor_lib utils_lib functional_lib model_lib test_lib)
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
