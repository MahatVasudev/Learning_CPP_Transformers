cmake_minimum_required(VERSION 3.15)

project(SampleProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(functional_lib STATIC
  src/functional/activations.cpp
  src/functional/tensor.cpp
  src/functional/ops.cpp
  src/functional/gradcheck.cpp
  src/functional/module.cpp
)

add_library(model_lib STATIC src/model/model.cpp src/model/decoder.cpp src/model/encoder.cpp)

target_include_directories(functional_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(model_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_executable(main_app main.cpp)

target_link_libraries(main_app PRIVATE model_lib functional_lib)
