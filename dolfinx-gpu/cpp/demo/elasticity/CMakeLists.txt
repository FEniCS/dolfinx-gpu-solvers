# Copyright (C) 2026 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

set(PROJECT_NAME demo_elasticity)
project(${PROJECT_NAME} LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT TARGET dolfinx::dolfinx)
  find_package(DOLFINX REQUIRED)
endif()
find_package(Boost 1.70 REQUIRED program_options)

if(DEFINED ENV{CUDA_ARCH})
  set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCH})
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} -g -march=native -std=c++20 -O3 -DNDEBUG \
       --extended-lambda --expt-relaxed-constexpr -lineinfo")
  message(STATUS "CUDA architecture: $ENV{CUDA_ARCH}")
  enable_language(CUDA)
  set_source_files_properties(main.cpp geometry.cpp PROPERTIES LANGUAGE CUDA)
elseif(DEFINED ENV{HIP_ARCH})
  set(CMAKE_HIP_ARCHITECTURES $ENV{HIP_ARCH})
  set(CMAKE_HIP_STANDARD 20)
  message(STATUS "HIP architecture: $ENV{HIP_ARCH}")
  enable_language(HIP)
  set_source_files_properties(main.cpp geometry.cpp PROPERTIES LANGUAGE HIP)
else()
  message(STATUS "No GPU back-end selected; building plain C++20")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/coarse_elasticity.c
    ${CMAKE_CURRENT_BINARY_DIR}/coarse_elasticity.h

  COMMAND
    ffcx ${CMAKE_CURRENT_SOURCE_DIR}/coarse_elasticity.py

  WORKING_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/coarse_elasticity.py

  COMMENT
    "Compiling coarse_elasticity.py using FFCx"

  VERBATIM
)

add_executable(${PROJECT_NAME}
  main.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/coarse_elasticity.c
)

target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx::dolfinx Boost::program_options)

set(CMAKE_CXX_EXTENSIONS OFF)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
