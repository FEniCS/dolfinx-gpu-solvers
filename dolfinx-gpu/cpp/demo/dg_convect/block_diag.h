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

// ── GPU BLAS portability
// ────────────────────────────────────────────────────── Define a unified
// gpublas* interface so the rest of the file is vendor-agnostic.  Compile with
// hipcc to get the HIP path.

#ifdef __HIPCC__
#include <hipblas/hipblas.h>
using gpublasHandle_t = hipblasHandle_t;
using gpublasStatus_t = hipblasStatus_t;
#define GPUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define gpublasCreate hipblasCreate
#define gpublasDestroy hipblasDestroy
#define gpublasDgetrfBatched hipblasDgetrfBatched
#define gpublasDgetriBatched hipblasDgetriBatched
#define gpublasSgetrfBatched hipblasSgetrfBatched
#define gpublasSgetriBatched hipblasSgetriBatched
#define gpublasDgemvBatched hipblasDgemvBatched
#define gpublasSgemvBatched hipblasSgemvBatched
#define GPUBLAS_OP_T HIPBLAS_OP_T
#else
#include <cublas_v2.h>
using gpublasHandle_t = cublasHandle_t;
using gpublasStatus_t = cublasStatus_t;
#define GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define gpublasCreate cublasCreate
#define gpublasDestroy cublasDestroy
#define gpublasDgetrfBatched cublasDgetrfBatched
#define gpublasDgetriBatched cublasDgetriBatched
#define gpublasSgetrfBatched cublasSgetrfBatched
#define gpublasSgetriBatched cublasSgetriBatched
#define gpublasDgemvBatched cublasDgemvBatched
#define gpublasSgemvBatched cublasSgemvBatched
#define GPUBLAS_OP_T CUBLAS_OP_T
#endif

// ── Error-checking macro
// ────────────────────────────────────────────────────── Throws
// std::runtime_error on any non-SUCCESS status, embedding the call site so the
// error is actionable without a debugger.

#define BLAS_CHECK(call)                                                       \
  do                                                                           \
  {                                                                            \
    gpublasStatus_t _blas_status = (call);                                     \
    if (_blas_status != GPUBLAS_STATUS_SUCCESS)                                \
      throw std::runtime_error(                                                \
          "GPU BLAS error (status="                                            \
          + std::to_string(static_cast<int>(_blas_status))                     \
          + ") at " __FILE__ ":" + std::to_string(__LINE__));                  \
  } while (false)

// ── assemble_dg
// ───────────────────────────────────────────────────────────────

template <dolfinx::scalar T>
class BlockDiagonalSolver
{
  using U = dolfinx::scalar_value_t<T>;

public:
  BlockDiagonalSolver(
      const dolfinx::fem::Form<T, U>& a,
      const std::vector<
          std::reference_wrapper<const dolfinx::fem::DirichletBC<T, U>>>& bcs)
  {
    // Assemble a in DG style
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
    BLAS_CHECK(gpublasCreate(&_handle));

    ndofs = a.function_spaces()[0]->dofmap()->element_dof_layout().num_dofs();
    int ncells = static_cast<int>(A_cpu.size()) / (ndofs * ndofs);

    // Allocate memory for A, A^-1
    thrust::device_vector<T> A(A_cpu.begin(), A_cpu.end());
    _Ainv.resize(A.size());

    std::vector<T*> ptrA(ncells);
    std::vector<T*> ptrAinv(ncells);
    for (int i = 0; i < ncells; ++i)
    {
      ptrA[i] = A.data().get() + ndofs * ndofs * i;
      ptrAinv[i] = _Ainv.data().get() + ndofs * ndofs * i;
    }

    // A is changed by getrfBatched
    thrust::device_vector<T*> ptrA_device(ptrA.begin(), ptrA.end());
    // Use a non-const pointer array here: matinvBatched writes T* output, not
    // const T*.  The member ptrAinv_device (also T*) is assigned after
    // inversion so that solve() can pass it directly to gemvBatched.
    ptrAinv_device = thrust::device_vector<T*>(ptrAinv.begin(), ptrAinv.end());
    thrust::device_vector<int> info(ncells, 0);

    // Invert A blockwise
    if constexpr (std::is_same_v<double, T>)
    {
      thrust::device_vector<int> ipiv(ndofs * ncells);
      BLAS_CHECK(gpublasDgetrfBatched(_handle, ndofs, ptrA_device.data().get(),
                                      ndofs, ipiv.data().get(),
                                      info.data().get(), ncells));
      BLAS_CHECK(gpublasDgetriBatched(
          _handle, ndofs, ptrA_device.data().get(), ndofs, ipiv.data().get(),
          ptrAinv_device.data().get(), ndofs, info.data().get(), ncells));
    }
    else if constexpr (std::is_same_v<float, T>)
    {
      thrust::device_vector<int> ipiv(ndofs * ncells);
      BLAS_CHECK(gpublasSgetrfBatched(_handle, ndofs, ptrA_device.data().get(),
                                      ndofs, ipiv.data().get(),
                                      info.data().get(), ncells));
      BLAS_CHECK(gpublasSgetriBatched(
          _handle, ndofs, ptrA_device.data().get(), ndofs, ipiv.data().get(),
          ptrAinv_device.data().get(), ndofs, info.data().get(), ncells));
    }
    else
      throw std::runtime_error("Unsupported scalar type");

    // Check that every block was non-singular (info[i] == 0 means success).
    // Copy to host first; the device vector is not directly iterable on CPU.
    std::vector<int> h_info(ncells);
    thrust::copy(info.begin(), info.end(), h_info.begin());
    if (auto it
        = std::ranges::find_if_not(h_info, [](int v) { return v == 0; });
        it != h_info.end())
    {
      throw std::runtime_error(
          "matinvBatched: singular matrix at block "
          + std::to_string(std::distance(h_info.begin(), it)));
    }
  }

  ~BlockDiagonalSolver() { gpublasDestroy(_handle); }

  /// Solve A.u = b for a block diagonal system
  /// Assumes DG style matrix laid out in cell order in A
  /// and that dofs of b and u follow the same layout
  /// @param b RHS input vector
  /// @param u Solution vector
  /// @param alpha
  /// @param beta Factor to apply to u, so we can
  /// solve u = alpha(A^-1 b) + beta u
  void solve(const thrust::device_vector<T>& b, thrust::device_vector<T>& u,
             T alpha = T{1}, T beta = T{0})
  {
    if (b.size() * ndofs != _Ainv.size() or b.size() != u.size())
      throw std::runtime_error("Size mismatch in BlockDiagonalSolver");

    int ncells = static_cast<int>(b.size()) / ndofs;
    std::vector<const T*> ptrb(ncells);
    std::vector<T*> ptru(ncells);
    for (int i = 0; i < ncells; ++i)
    {
      ptrb[i] = b.data().get() + ndofs * i;
      ptru[i] = u.data().get() + ndofs * i;
    }
    thrust::device_vector<const T*> ptrb_device(ptrb.begin(), ptrb.end());
    thrust::device_vector<T*> ptru_device(ptru.begin(), ptru.end());

    // NB use transpose operator, since original A was Row Major.
    if constexpr (std::is_same_v<double, T>)
    {
      BLAS_CHECK(gpublasDgemvBatched(_handle, GPUBLAS_OP_T, ndofs, ndofs,
                                     &alpha, ptrAinv_device.data().get(), ndofs,
                                     ptrb_device.data().get(), 1, &beta,
                                     ptru_device.data().get(), 1, ncells));
    }
    else if constexpr (std::is_same_v<float, T>)
    {
      BLAS_CHECK(gpublasSgemvBatched(_handle, GPUBLAS_OP_T, ndofs, ndofs,
                                     &alpha, ptrAinv_device.data().get(), ndofs,
                                     ptrb_device.data().get(), 1, &beta,
                                     ptru_device.data().get(), 1, ncells));
    }
    else
      throw std::runtime_error("Unsupported scalar type");
  }

private:
  // GPU blas handle
  gpublasHandle_t _handle;

  // Number of dofs per cell
  int ndofs;

  // Inverse of A, stored blockwise
  thrust::device_vector<T> _Ainv;
  thrust::device_vector<T*> ptrAinv_device;
};
