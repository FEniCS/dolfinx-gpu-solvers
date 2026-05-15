/// Solve block diagonal mass-matrix system on GPU
#pragma once
#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/la/Vector.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

// ── GPU BLAS portability ──────────────────────────────────────────────────────
// Define a unified gpublas* interface so the rest of the file is
// vendor-agnostic.  Compile with hipcc to get the HIP path.

#ifdef __HIPCC__
#include <hipblas/hipblas.h>
using gpublasHandle_t = hipblasHandle_t;
using gpublasStatus_t = hipblasStatus_t;
#define GPUBLAS_STATUS_SUCCESS  HIPBLAS_STATUS_SUCCESS
#define gpublasCreate           hipblasCreate
#define gpublasDestroy          hipblasDestroy
#define gpublasDmatinvBatched   hipblasDmatinvBatched
#define gpublasSmatinvBatched   hipblasSmatinvBatched
#define gpublasDgemvBatched     hipblasDgemvBatched
#define gpublasSgemvBatched     hipblasSgemvBatched
#define GPUBLAS_OP_T            HIPBLAS_OP_T
#else
#include <cublas_v2.h>
using gpublasHandle_t = cublasHandle_t;
using gpublasStatus_t = cublasStatus_t;
#define GPUBLAS_STATUS_SUCCESS  CUBLAS_STATUS_SUCCESS
#define gpublasCreate           cublasCreate
#define gpublasDestroy          cublasDestroy
#define gpublasDmatinvBatched   cublasDmatinvBatched
#define gpublasSmatinvBatched   cublasSmatinvBatched
#define gpublasDgemvBatched     cublasDgemvBatched
#define gpublasSgemvBatched     cublasSgemvBatched
#define GPUBLAS_OP_T            CUBLAS_OP_T
#endif

// ── Error-checking macro ──────────────────────────────────────────────────────
// Throws std::runtime_error on any non-SUCCESS status, embedding the
// call site so the error is actionable without a debugger.

#define BLAS_CHECK(call)                                                       \
  do {                                                                         \
    gpublasStatus_t _blas_status = (call);                                     \
    if (_blas_status != GPUBLAS_STATUS_SUCCESS)                                \
      throw std::runtime_error(                                                \
          "GPU BLAS error (status="                                            \
          + std::to_string(static_cast<int>(_blas_status))                    \
          + ") at " __FILE__ ":" + std::to_string(__LINE__));                  \
  } while (false)

// ── assemble_dg ───────────────────────────────────────────────────────────────

/// Assemble a form in DG style, simply appending each dense local element
/// matrix to an array, inverting blocks after assembly
/// @param a Form
/// @param bcs Dirichlet BCs
/// @returns Ainv in DG style, suitable to solve with a MatVec call
template <dolfinx::scalar T, std::floating_point U>
thrust::device_vector<T> assemble_dg(
    const dolfinx::fem::Form<T, U>& a,
    const std::vector<
        std::reference_wrapper<const dolfinx::fem::DirichletBC<T, U>>>& bcs)
{
  // Vector to assemble into
  std::vector<T> A_cpu;

  auto mat_add = [&A_cpu](std::span<const std::int32_t> rows,
                           std::span<const std::int32_t> cols,
                           std::span<const T> data) -> int
  {
    A_cpu.insert(A_cpu.end(), data.begin(), data.end());
    return 0;
  };

  assemble_matrix(mat_add, a, bcs);

  // Create a GPU BLAS context
  gpublasHandle_t handle;
  BLAS_CHECK(gpublasCreate(&handle));

  // FIXME: get from element/dofmap
  int n = 4; // 4 dofs per cell
  int ncells = static_cast<int>(A_cpu.size()) / (n * n);

  // Allocate memory for A, A^-1
  thrust::device_vector<T> A(A_cpu.begin(), A_cpu.end());
  thrust::device_vector<T> Ainv(A.size());

  std::vector<const T*> ptrA(ncells);
  std::vector<T*> ptrAinv(ncells);
  for (int i = 0; i < ncells; ++i)
  {
    ptrA[i]    = A.data().get()    + n * n * i;
    ptrAinv[i] = Ainv.data().get() + n * n * i;
  }
  thrust::device_vector<const T*> ptrA_device(ptrA.begin(), ptrA.end());
  thrust::device_vector<T*> ptrAinv_device(ptrAinv.begin(), ptrAinv.end());
  thrust::device_vector<int> info(ncells, 0);

  // Invert A blockwise
  if constexpr (std::is_same_v<double, T>)
  {
    BLAS_CHECK(gpublasDmatinvBatched(handle, n,
                                     ptrA_device.data().get(), n,
                                     ptrAinv_device.data().get(), n,
                                     info.data().get(), ncells));
  }
  else if constexpr (std::is_same_v<float, T>)
  {
    BLAS_CHECK(gpublasSmatinvBatched(handle, n,
                                     ptrA_device.data().get(), n,
                                     ptrAinv_device.data().get(), n,
                                     info.data().get(), ncells));
  }
  else
    throw std::runtime_error("Unsupported scalar type");

  // Check that every block was non-singular (info[i] == 0 means success).
  // Copy to host first; the device vector is not directly iterable on CPU.
  std::vector<int> h_info(ncells);
  thrust::copy(info.begin(), info.end(), h_info.begin());
  if (auto it = std::ranges::find_if_not(h_info, [](int v) { return v == 0; });
      it != h_info.end())
  {
    throw std::runtime_error(
        "matinvBatched: singular matrix at block "
        + std::to_string(std::distance(h_info.begin(), it)));
  }

  BLAS_CHECK(gpublasDestroy(handle));

  return Ainv;
}

// ── solve_block_diag_system ───────────────────────────────────────────────────

/// Solve A.u = b for a block diagonal system
/// Assumes DG style matrix laid out in cell order in A
/// and that dofs of b and u follow the same layout
/// @param Ainv inverted block-diagonal matrix in DG form
/// @param b RHS input vector
/// @param u Solution vector
template <dolfinx::scalar T>
void solve_block_diag_system(
    const thrust::device_vector<T>& Ainv,
    const dolfinx::la::Vector<T, thrust::device_vector<T>>& b,
    dolfinx::la::Vector<T, thrust::device_vector<T>>& u)
{
  gpublasHandle_t handle;
  BLAS_CHECK(gpublasCreate(&handle));

  int n = 4;
  int ncells = static_cast<int>(b.array().size()) / n;

  std::vector<const T*> ptrAinv(ncells);
  std::vector<const T*> ptrb(ncells);
  std::vector<T*> ptru(ncells);
  for (int i = 0; i < ncells; ++i)
  {
    ptrAinv[i] = Ainv.data().get()         + n * n * i;
    ptrb[i]    = b.array().data().get()    + n * i;
    ptru[i]    = u.array().data().get()    + n * i;
  }
  thrust::device_vector<const T*> ptrAinv_device(ptrAinv.begin(), ptrAinv.end());
  thrust::device_vector<const T*> ptrb_device(ptrb.begin(), ptrb.end());
  thrust::device_vector<T*>       ptru_device(ptru.begin(), ptru.end());

  // NB use transpose operator, since original A was Row Major.
  // alpha and beta must match the scalar type T.
  const T alpha = T{1};
  const T beta  = T{0};

  if constexpr (std::is_same_v<double, T>)
  {
    BLAS_CHECK(gpublasDgemvBatched(
        handle, GPUBLAS_OP_T, n, n, &alpha,
        ptrAinv_device.data().get(), n,
        ptrb_device.data().get(), 1, &beta,
        ptru_device.data().get(), 1, ncells));
  }
  else if constexpr (std::is_same_v<float, T>)
  {
    BLAS_CHECK(gpublasSgemvBatched(
        handle, GPUBLAS_OP_T, n, n, &alpha,
        ptrAinv_device.data().get(), n,
        ptrb_device.data().get(), 1, &beta,
        ptru_device.data().get(), 1, ncells));
  }
  else
    throw std::runtime_error("Unsupported scalar type");

  BLAS_CHECK(gpublasDestroy(handle));
}
