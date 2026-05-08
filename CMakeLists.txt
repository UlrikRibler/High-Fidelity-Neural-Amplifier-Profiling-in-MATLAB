cmake_minimum_required(VERSION 3.24)

project(NeuralAmpCpp VERSION 1.0.0 LANGUAGES CXX)

if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)

FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()
add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})

FetchContent_MakeAvailable(nlohmann_json)

add_library(neural_amp_core
  src/Artifacts.cpp
  src/Benchmark.cpp
  src/Config.cpp
  src/Dataset.cpp
  src/Dsp.cpp
  src/Model.cpp
  src/Pipeline.cpp
  src/Trainer.cpp
  src/VirtualTubeAmp.cpp
)

target_include_directories(neural_amp_core PUBLIC src)
target_link_libraries(neural_amp_core PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json)

if(MSVC)
  target_compile_options(neural_amp_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(neural_amp_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(neural_amp src/main.cpp)
target_link_libraries(neural_amp PRIVATE neural_amp_core)

enable_testing()

add_executable(neural_amp_tests tests/test_core.cpp)
target_link_libraries(neural_amp_tests PRIVATE neural_amp_core)
add_test(NAME neural_amp_tests COMMAND neural_amp_tests)
