// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <cstdio>
#include <sstream>
#include <string>

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#elif USE_CUDA
#include <cuda_runtime.h>
#endif

#if defined(USE_CUDA) || defined(USE_HIP)

// Some useful utilities for error checking and synchronisation
// for each hardware type

namespace benchdolfinx
{
#ifdef USE_HIP
#define err_check(command)                                                     \
  {                                                                            \
    hipError_t status = command;                                               \
    if (status != hipSuccess)                                                  \
    {                                                                          \
      printf("(%s:%d) Error: Hip reports %s\n", __FILE__, __LINE__,            \
             hipGetErrorString(status));                                       \
      exit(1);                                                                 \
    }                                                                          \
  }
#elif USE_CUDA
#define err_check(command)                                                     \
  {                                                                            \
    cudaError_t status = command;                                              \
    if (status != cudaSuccess)                                                 \
    {                                                                          \
      printf("(%s:%d) Error: CUDA reports %s\n", __FILE__, __LINE__,           \
             cudaGetErrorString(status));                                      \
      exit(1);                                                                 \
    }                                                                          \
  }
#endif

#ifdef USE_HIP
#define non_temp_load(addr) __builtin_nontemporal_load(addr)
#define deviceMemcpyToSymbol(symbol, addr, count)                              \
  hipMemcpyToSymbol(symbol, addr, count)
inline void check_device_last_error() { err_check(hipGetLastError()); }
inline void device_synchronize() { err_check(hipDeviceSynchronize()); }
#elif USE_CUDA
#define non_temp_load(addr) __ldg(addr)
#define deviceMemcpyToSymbol(symbol, addr, count)                              \
  cudaMemcpyToSymbol(symbol, addr, count)
inline void check_device_last_error() { err_check(cudaGetLastError()); }
inline void device_synchronize() { err_check(cudaDeviceSynchronize()); }
#else
#error "Unsupported platform"
#endif

std::string get_device_information();
} // namespace benchdolfinx
#endif
