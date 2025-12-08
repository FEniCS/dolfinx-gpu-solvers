// Copyright (C) 2025 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cassert>
#include <complex>
#include <cstdint>

#include <hipsparse/hipsparse.h>

/// Interfaces to HIP library functions
/// using dolfinx Vector and MatrixCSR which are
/// allocated on-device
namespace dolfinx::la::hip
{

/// hipsparse provides a sparse MatVec for HIP
/// providing the operator  y = Ax
/// @tparam MatType Matrix type stored on-device
/// @tparam VecType Vector type stored on-device
template <typename MatType, typename VecType>
class hipsparseMatVec
{

public:
  /// @brief Create an operator for y = Ax
  /// @param A_device CSR Matrix stored on device
  /// @param y_device Output Vector stored on device
  /// @param x_device Input Vector stored on device
  hipsparseMatVec(MatType& A_device, VecType& y_device, VecType& x_device);

  /// Destructor
  ~hipsparseMatVec();

  /// Apply the operator, computing y = Ax
  void apply();

private:
  // HIP representation of scalar type
  hipDataType data_type;

  // Data structures wrapping matrix
  hipDataType hipValueType;
  hipsparseHandle_t handle;
  hipsparseSpMatDescr_t matA;

  // Data structures wrapping vectors
  hipsparseDnVecDescr_t vecX, vecY;

  // scratch
  void* dBuffer;

  // Coefficients in axpy
  MatType::value_type alpha, beta;
};
//-----------------------------------------------------------------------------
template <typename MatType, typename VecType>
hipsparseMatVec<MatType, VecType>::hipsparseMatVec(MatType& A_device,
                                                 VecType& y_device,
                                                 VecType& x_device)
{
  using T = MatType::value_type;
  using U = VecType::value_type;
  static_assert(std::is_same_v<T, U>, "Incompatible data types");

  if constexpr (std::is_same_v<T, double>)
    data_type = HIP_R_64F;
  else if constexpr (std::is_same_v<T, float>)
    data_type = HIP_R_32F;
  else if constexpr (std::is_same_v<T, std::complex<float>>)
    data_type = HIP_C_32F;
  else if constexpr (std::is_same_v<T, std::complex<double>>)
    data_type = HIP_C_64F;
  else
    throw std::runtime_error("Value type not supported");

  hipsparseCreate(&handle);

  int nnz = A_device.values().size();
  assert(A_device.values().size() == A_device.cols().size());
  int nrows = A_device.row_ptr().size() - 1;

  assert(nrows == y_device.array().size());
  int ncols = x_device.array().size();

  hipsparseStatus_t status = hipsparseCreateCsr(
      &matA, nrows, ncols, nnz, (void*)A_device.row_ptr().data().get(),
      (void*)A_device.cols().data().get(),
      (void*)A_device.values().data().get(), HIPSPARSE_INDEX_32I,
      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, data_type);
  assert(status == HIPSPARSE_STATUS_SUCCESS);

  // Create dense vector X
  status = hipsparseCreateDnVec(&vecX, x_device.array().size(),
                               x_device.array().data().get(), data_type);
  assert(status == HIPSPARSE_STATUS_SUCCESS);
  // Create dense vector y
  status = hipsparseCreateDnVec(&vecY, y_device.array().size(),
                               y_device.array().data().get(), data_type);
  assert(status == HIPSPARSE_STATUS_SUCCESS);

  alpha = 1.0;
  beta = 0.0;

  // allocate an external buffer if needed
  std::size_t bufferSize;
  status = hipsparseSpMV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, matA, vecX, &beta, vecY, data_type,
                                   HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  assert(status == HIPSPARSE_STATUS_SUCCESS);

  hipError_t err = hipMalloc(&dBuffer, bufferSize);
  assert(err == hipSuccess);
}
//-----------------------------------------------------------------------------
template <typename MatType, typename VecType>
void hipsparseMatVec<MatType, VecType>::apply()
{
  hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
               &beta, vecY, data_type, HIPSPARSE_SPMV_ALG_DEFAULT, dBuffer);
}
//-----------------------------------------------------------------------------
template <typename MatType, typename VecType>
hipsparseMatVec<MatType, VecType>::~hipsparseMatVec()
{
  hipsparseStatus_t status = hipsparseDestroySpMat(matA);
  assert(status == HIPSPARSE_STATUS_SUCCESS);
  status = hipsparseDestroy(handle);
  assert(status == HIPSPARSE_STATUS_SUCCESS);
  status = hipsparseDestroyDnVec(vecX);
  assert(status == HIPSPARSE_STATUS_SUCCESS);
  status = hipsparseDestroyDnVec(vecY);
  assert(status == HIPSPARSE_STATUS_SUCCESS);
  hipError_t err = hipFree(dBuffer);
  assert(err == hipSuccess);
}
} // namespace dolfinx::la::hip
