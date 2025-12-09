# Copyright (C) 2025 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT

cmake_minimum_required(VERSION 3.21)

set(PROJECT_NAME demo_heat)
project(${PROJECT_NAME} LANGUAGES C CXX CUDA)

if(NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

add_custom_command(
  OUTPUT heat.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/heat.py
  VERBATIM
  DEPENDS heat.py
  COMMENT "Compile heat.py using FFCx"
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -march=native -std=c++20 -O3 -DNDEBUG --extended-lambda --expt-relaxed-constexpr -lineinfo")

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

find_package(cudss)
find_package(CUDAToolkit)

add_executable(${PROJECT_NAME} main.cpp ${CMAKE_CURRENT_BINARY_DIR}/heat.c)
target_link_libraries(${PROJECT_NAME} dolfinx cudss ${CUDA_cusparse_LIBRARY})

# Set C++20 standard
set(CMAKE_CXX_EXTENSIONS OFF)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
