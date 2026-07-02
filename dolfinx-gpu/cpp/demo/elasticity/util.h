// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <cstdio>
#include <sstream>
#include <string>

#if defined(__HIP__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#if defined(__HIP__)
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
#else
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

#if defined(__HIP__)
#define non_temp_load(addr) __builtin_nontemporal_load(addr)
#define deviceMemcpyToSymbol(symbol, addr, count)                              \
  hipMemcpyToSymbol(symbol, addr, count)
inline void check_device_last_error() { err_check(hipGetLastError()); }
inline void device_synchronize() { err_check(hipDeviceSynchronize()); }
#else
#define non_temp_load(addr) __ldg(addr)
#define deviceMemcpyToSymbol(symbol, addr, count)                              \
  cudaMemcpyToSymbol(symbol, addr, count)
inline void check_device_last_error() { err_check(cudaGetLastError()); }
inline void device_synchronize() { err_check(cudaDeviceSynchronize()); }
#endif
