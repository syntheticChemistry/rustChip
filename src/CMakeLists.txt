cmake_minimum_required(VERSION 3.16)
project(akida_inference CXX)

# Set C++17 standard (required by Akida Engine API)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Configure CMake module path
# ============================================================================
# The Akida Engine provides CMake modules in engine/cmake/ for finding
# the akida library. We need to add this path to CMAKE_MODULE_PATH.
#
# IMPORTANT: This CMakeLists.txt assumes it's placed at:
#   engine/test/customer_deployment_package/src/
#
# From there, we reach engine/cmake/ via: ../../cmake/
# If you place this package elsewhere, adjust this path accordingly.

set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/../../../cmake"
    ${CMAKE_MODULE_PATH}
)

# ============================================================================
# Find the Akida library
# ============================================================================
# The akida-model module provides the 'akida' target which links:
# - The core Akida Engine library (libakida.a)
# - Host PCIe drivers (for device discovery via akida::get_drivers())
# - Required system libraries

include(akida-model)

# ============================================================================
# Build the inference executable
# ============================================================================
# Collect all source files in this directory, including:
# - main.cpp (entry point)
# - inference.cpp (inference pipeline)
# - system_linux.cpp (system.h implementation)
# - program.cpp (generated model binary)

file(GLOB SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/*.h")

add_executable(akida_inference ${SOURCES})

# Link against the Akida library
target_link_libraries(akida_inference PRIVATE akida)

# ============================================================================
# Configure include paths
# ============================================================================
# The Akida Engine headers use paths like:
#   #include "api/akida/hardware_device.h"
#   #include "host/hardware_drivers.h"
#   #include "infra/system.h"
#
# We need to add the engine root to the include path so these resolve correctly.
# Assuming engine deployment structure:
#   engine/
#   ├── api/
#   ├── host/
#   ├── infra/
#   └── test/customer_deployment_package/src/  <- we are here
#
# We reach engine/ via: ../../..

target_include_directories(akida_inference PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../.."
)

# ============================================================================
# Build output
# ============================================================================
# Executable will be created as: build/akida_inference
# Run it with: ./build/akida_inference

message(STATUS "Configuring Akida Inference Demo")
message(STATUS "  Executable: akida_inference")
message(STATUS "  Engine root: ${CMAKE_CURRENT_LIST_DIR}/../../..")
message(STATUS "  CMake module path: ${CMAKE_MODULE_PATH}")
