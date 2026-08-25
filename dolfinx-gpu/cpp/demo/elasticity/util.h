// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include <cstdio>
#include <sstream>
#include <string>
#include <mpi.h>

#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/memory.h>

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


struct GPUSelection
{
  int local_rank;
  int device;
};

inline GPUSelection select_gpu_for_rank(MPI_Comm comm)
{
  MPI_Comm local_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);

  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);

  int num_devices = 0;

  #if defined(__HIP__)
    hipGetDeviceCount(&num_devices);
    const int device = local_rank % num_devices;
    hipSetDevice(device);
  #else
    cudaGetDeviceCount(&num_devices);
    const int device = local_rank % num_devices;
    cudaSetDevice(device);
  #endif

  MPI_Comm_free(&local_comm);

  const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");

  std::cout << "local rank " << local_rank
            << ", num visible GPUs = " << num_devices
            << ", CUDA_VISIBLE_DEVICES = "
            << (visible ? visible : "not set")
            << "\n";

  char pci_bus_id[32];
  cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device);

  std::cout << "local rank " << local_rank
            << " CUDA device " << device
            << " PCI bus " << pci_bus_id
            << "\n";

  return {local_rank, device};
}


struct device_pack
{
  template <typename IndexIt, typename InputIt, typename OutputIt>
  void operator()(IndexIt idx_first, IndexIt idx_last, InputIt in_first, OutputIt out_first)
  {
    thrust::gather(thrust::device, idx_first, idx_last, in_first, out_first);
  }
};


struct device_unpack
{
  template <typename IndexIt, typename InputIt, typename OutputIt>
  void operator()(IndexIt idx_first, IndexIt idx_last, InputIt in_first, OutputIt out_first)
  {
    const std::size_t n = idx_last - idx_first;
    thrust::scatter(thrust::device, in_first, in_first + n, idx_first, out_first);
  }
};


struct device_get_ptr
{
  template <typename Container>
  auto operator()(Container& x) const
  {
    return thrust::raw_pointer_cast(x.data());
  }
};


template <typename T>
__global__ void scatter_add_kernel(
  T* output,
  const T* input,
  const std::int32_t* indices,
  std::size_t n)
{
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < n)
  {
    const std::int32_t idx = indices[i];
    atomicAdd(&output[idx], input[i]);
  }
};


struct device_unpack_add{
  template <typename IndexIt, typename InputIt, typename OutputIt>
  void operator()(IndexIt idx_first, IndexIt idx_last, InputIt in_first, OutputIt out_first) const
  {
    const std::size_t n = idx_last - idx_first;

    if (n == 0) return;

    const auto* indicies = thrust::raw_pointer_cast(&*idx_first);
    const auto* input = thrust::raw_pointer_cast(&*in_first);
    auto* output = thrust::raw_pointer_cast(&*out_first);

    constexpr int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    scatter_add_kernel<<<num_blocks, block_size>>>(output, input, indicies, n);
  }
};


template <typename Vector>
void scatter_fwd(Vector& x){
  x.scatter_fwd_begin(device_pack{}, device_get_ptr{});
  x.scatter_fwd_end(device_unpack{});
}


template <typename Vector>
void scatter_rev_add(Vector& x){
  x.scatter_rev_begin(device_unpack{}, device_get_ptr{});
  x.scatter_rev_end(device_unpack_add{});
}
