// Copyright (C) 2025 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <cassert>
#include <complex>
#include <cudss.h>

/// Interfaces to CUDA library functions
/// using dolfinx Vector and MatrixCSR which are
/// allocated on-device
namespace dolfinx::la::cuda
{

/// cuDSS provides a direct sparse solver for CUDA
/// solving the problem A.u = b
/// @tparam MatType A Matrix type stored on-device
/// @tparam VecType A Vector type stored on-device
template <typename MatType, typename VecType>
class cudssSolver
{

public:
  /// @brief Create a solver for A.u = b
  /// @param A_device CSR Matrix stored on device
  /// @param b_device RHS Vector stored on device
  /// @param u_device Solution Vector stored on device
  cudssSolver(MatType& A_device, VecType& b_device, VecType& u_device);

  /// Destructor
  ~cudssSolver();

  /// @brief Perform analysis of the matrix
  /// @note Prerequisite for factorization
  void analyze();

  /// @brief Perform factorization of the matrix
  /// @note Matrix must be analyzed before calling factorize
  void factorize();

  /// @brief Solve the problem A.u = b
  /// @note Matrix must be factorized before calling solve
  void solve();

private:
  /// Data structures wrapping matrix and vectors
  cudssMatrix_t A_dss;
  cudssMatrix_t b_dss;
  cudssMatrix_t u_dss;

  /// Internal data of cuDSS
  cudssHandle_t handle;
  cudssConfig_t config;
  cudssData_t data;
};
//-----------------------------------------------------------------------------
template <typename MatType, typename VecType>
cudssSolver<MatType, VecType>::cudssSolver(MatType& A_device, VecType& b_device,
                                           VecType& u_device)
{
  using T = MatType::value_type;
  using U = VecType::value_type;
  static_assert(std::is_same_v<T, U>, "Incompatible data types");

  cudaDataType_t data_type;
  if constexpr (std::is_same_v<T, double>)
    data_type = CUDA_R_64F;
  else if constexpr (std::is_same_v<T, float>)
    data_type = CUDA_R_32F;
  else if constexpr (std::is_same_v<T, std::complex<float>>)
    data_type = CUDA_C_32F;
  else if constexpr (std::is_same_v<T, std::complex<double>>)
    data_type = CUDA_C_64F;
  else
    throw std::runtime_error("CUDSS: invalid data type");

  cudssStatus_t status = cudssCreate(&handle);
  if (status != CUDSS_STATUS_SUCCESS)
    throw std::runtime_error("CUDSS: error creating handle");
  status = cudssConfigCreate(&config);
  if (status != CUDSS_STATUS_SUCCESS)
    throw std::runtime_error("CUDSS: error creating config");
  status = cudssDataCreate(handle, &data);
  if (status != CUDSS_STATUS_SUCCESS)
    throw std::runtime_error("CUDSS: error creating data");

  status = cudssMatrixCreateCsr(
      &A_dss, A_device.index_map(0)->size_local(),
      A_device.index_map(1)->size_local(), A_device.cols().size(),
      (void*)(A_device.row_ptr().data().get()), NULL,
      (void*)(A_device.cols().data().get()),
      (void*)(A_device.values().data().get()), CUDA_R_32I, data_type,
      CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);
  if (status != CUDSS_STATUS_SUCCESS)
    throw std::runtime_error("CUDSS: error wrapping MatrixCSR");

  std::int64_t nrows = b_device.index_map()->size_local();
  status = cudssMatrixCreateDn(&b_dss, nrows, 1, nrows,
                               (void*)b_device.array().data().get(), data_type,
                               CUDSS_LAYOUT_COL_MAJOR);
  if (status != CUDSS_STATUS_SUCCESS)
    throw std::runtime_error("CUDSS: error wrapping b vector");
  status = cudssMatrixCreateDn(&u_dss, nrows, 1, nrows,
                               (void*)u_device.array().data().get(), data_type,
                               CUDSS_LAYOUT_COL_MAJOR);
  if (status != CUDSS_STATUS_SUCCESS)
    throw std::runtime_error("CUDSS: error wrapping u vector");
}
//-----------------------------------------------------------------------------
template <typename MatType, typename VecType>
void cudssSolver<MatType, VecType>::analyze()
{
  cudssStatus_t status = cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config,
                                      data, A_dss, u_dss, b_dss);
  if (status != CUDSS_STATUS_SUCCESS)
    throw std::runtime_error("CUDSS: error in analyze");
}
//-----------------------------------------------------------------------------
template <typename MatType, typename VecType>
void cudssSolver<MatType, VecType>::factorize()
{
  cudssStatus_t status = cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config,
                                      data, A_dss, u_dss, b_dss);
  if (status != CUDSS_STATUS_SUCCESS)
    throw std::runtime_error("CUDSS: error in factorize");
}
//-----------------------------------------------------------------------------
template <typename MatType, typename VecType>
void cudssSolver<MatType, VecType>::solve()
{
  cudssStatus_t status = cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data,
                                      A_dss, u_dss, b_dss);
  if (status != CUDSS_STATUS_SUCCESS)
    throw std::runtime_error("CUDSS: error in solve");
}
//-----------------------------------------------------------------------------
template <typename MatType, typename VecType>
cudssSolver<MatType, VecType>::~cudssSolver()
{
  cudssStatus_t status = cudssConfigDestroy(config);
  assert(status == CUDSS_STATUS_SUCCESS);
  status = cudssDataDestroy(handle, data);
  assert(status == CUDSS_STATUS_SUCCESS);
  status = cudssMatrixDestroy(A_dss);
  assert(status == CUDSS_STATUS_SUCCESS);
  status = cudssMatrixDestroy(u_dss);
  assert(status == CUDSS_STATUS_SUCCESS);
  status = cudssMatrixDestroy(b_dss);
  assert(status == CUDSS_STATUS_SUCCESS);
  status = cudssDestroy(handle);
  assert(status == CUDSS_STATUS_SUCCESS);
}

} // namespace dolfinx::la::cuda
