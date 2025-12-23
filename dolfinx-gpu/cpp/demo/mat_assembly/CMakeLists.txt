# Copyright (C) 2025 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT

cmake_minimum_required(VERSION 3.21)

set(PROJECT_NAME mat_assembly)
project(${PROJECT_NAME} LANGUAGES C CXX CUDA)

if(NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -march=native -std=c++20 -O3 -DNDEBUG --extended-lambda --expt-relaxed-constexpr -lineinfo")

set_source_files_properties(main.cpp sparsity.cpp PROPERTIES LANGUAGE CUDA)

add_executable(${PROJECT_NAME} main.cpp sparsity.cpp)
target_link_libraries(${PROJECT_NAME} dolfinx)

# Set C++20 standard
set(CMAKE_CXX_EXTENSIONS OFF)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
