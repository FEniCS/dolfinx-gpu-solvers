# Copyright (C) 2025 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

set(PROJECT_NAME demo_dg_convect)
project(${PROJECT_NAME} LANGUAGES C CXX)

if(NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Compile UFL form with ffcx
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT  dg_convect.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/dg_convect.py
  VERBATIM
  DEPENDS dg_convect.py
  COMMENT "Compiling dg_convect.py with ffcx"
)

# ---------------------------------------------------------------------------
# GPU back-end selection (optional for this CPU demo; mirrors project style)
# ---------------------------------------------------------------------------
if(DEFINED ENV{CUDA_ARCH})
  set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCH})
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} -g -march=native -std=c++20 -O3 -DNDEBUG \
       --extended-lambda --expt-relaxed-constexpr -lineinfo")
  message(STATUS "CUDA architecture: $ENV{CUDA_ARCH}")
  enable_language(CUDA)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
elseif(DEFINED ENV{HIP_ARCH})
  set(CMAKE_HIP_ARCHITECTURES $ENV{HIP_ARCH})
  set(CMAKE_HIP_STANDARD 20)
  message(STATUS "HIP architecture: $ENV{HIP_ARCH}")
  enable_language(HIP)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
else()
  message(STATUS "No GPU back-end selected; building plain C++20")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(${PROJECT_NAME}
  main.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/dg_convect.c
)

target_link_libraries(${PROJECT_NAME} dolfinx)

set(CMAKE_CXX_EXTENSIONS OFF)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
