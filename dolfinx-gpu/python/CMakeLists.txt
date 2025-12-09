# Copyright (C) 2025 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT

cmake_minimum_required(VERSION 3.21)

project(dolfinx_gpu_nanobind LANGUAGES CXX)

find_package(
  Python
  COMPONENTS Interpreter Development ${SKBUILD_SABI_COMPONENT}
  REQUIRED
)

# Options
include(FeatureSummary)
feature_summary(WHAT ALL)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# If CUDA is available, the user should set CUDA_ARCH environment variable
if(DEFINED ENV{CUDA_ARCH})
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
  set(CMAKE_CUDA_STANDARD 20)
  enable_language(CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -march=native -std=c++20 -O3 -DNDEBUG --extended-lambda --expt-relaxed-constexpr -lineinfo")
endif()

# If HIP is available, the user should set HIP_ARCH environment variable
if(DEFINED ENV{HIP_ARCH})
  set(CMAKE_HIP_ARCHITECTURES ${HIP_ARCH})
  set(CMAKE_HIP_STANDARD 20)
  enable_language(HIP)
endif()

find_package(DOLFINX REQUIRED CONFIG)

if(DOLFINX_FOUND)
  message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")
endif()

# Create the binding library nanobind handles its own calls to
# target_link_libraries

if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(NANOBIND_SABI "STABLE_ABI")
endif()

nanobind_add_module(
  gpucpp
  NB_SUPPRESS_WARNINGS
  NOMINSIZE
  ${NANOBIND_SABI}
  wrappers/gpu.cu
)

set(CMAKE_CXX_EXTENSIONS OFF)
target_compile_definitions(gpucpp PRIVATE cxx_std_20)

if(DEFINED ENV{CUDA_ARCH})
  # Add compile flag
  target_compile_definitions(gpucpp PUBLIC GPUCPP_SUBMODULE)
  # CUDA direct solver
  find_package(cudss)
  target_link_libraries(gpucpp PRIVATE cudss)
  target_link_libraries(gpucpp PRIVATE dolfinx)
endif()

if(DEFINED ENV{HIP_ARCH})
  set_source_files_properties(wrappers/gpu.cu PROPERTIES LANGUAGE HIP)
  target_link_libraries(gpucpp PRIVATE dolfinx)
endif()

find_package(DOLFINX REQUIRED CONFIG)

if(DOLFINX_FOUND)
  message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")
endif()

# Check for petsc4py
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import dolfinx; print(dolfinx.get_include())"
  OUTPUT_VARIABLE DOLFINX_INCLUDE_DIR
  RESULT_VARIABLE DOLFINX_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT DOLFINX_COMMAND_RESULT)
  message(STATUS "Found dolfinx include directory at ${DOLFINX_INCLUDE_DIR}")
  target_include_directories(gpucpp PRIVATE ${DOLFINX_INCLUDE_DIR})
else()
  message(STATUS "dolfinx python headers not found.")
endif()


set_target_properties(gpucpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS gpucpp DESTINATION dolfinx_gpu)
