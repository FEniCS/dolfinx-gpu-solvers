/// Solve block diagonal mass-matrix system on GPU
#include <cublas_v2.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/la/Vector.h>
#include <thrust/device_vector.h>

/// Assemble a form in DG style, simply appending each dense local element
/// matrix to an array.
template <dolfinx::scalar T, std::floating_point U>
thrust::device_vector<T> assemble_dg(
    const dolfinx::fem::Form<T, U>& a,
    const std::vector<
        std::reference_wrapper<const dolfinx::fem::DirichletBC<T, U>>>& bcs)
{
  // Vector to assemble into
  std::vector<T> A;

  auto mat_add
      = [&A](std::span<const std::int32_t> rows,
             std::span<const std::int32_t> cols, std::span<const T> data) -> int
  {
    int n = static_cast<int>(rows.size());
    // transpose data to column major
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        A.push_back(data[j * n + i]);
    return 0;
  };

  assemble_matrix(mat_add, a, bcs);
  return thrust::device_vector<T>(A.begin(), A.end());
}

// Solve A.u = b for a block diagonal system
template <dolfinx::scalar T>
void solve_block_diag_system(
    const thrust::device_vector<T>& A,
    const dolfinx::la::Vector<T, thrust::device_vector<T>>& b,
    dolfinx::la::Vector<T, thrust::device_vector<T>>& u)
{
  // Create a cublas context
  cublasHandle_t handle;
  cublasCreate(&handle);

  // FIXME: get from element/dofmap
  int n = 4;
  int ncells = A.size() / (n * n);

  // Allocate memory for A^-1
  thrust::device_vector<T> Ainv(A.size());

  std::vector<T*> ptrA(ncells);
  std::vector<T*> ptrAinv(ncells);
  for (int i = 0; i < ncells; ++i)
  {
    ptrA[i] = A.data().get() + n * n * i;
    ptrAinv[i] = Ainv.data().get() + n * n * i;
  }
  thrust::device_vector<T*> ptrA_device(ptrA.begin(), ptrA.end());
  thrust::device_vector<T*> ptrAinv_device(ptrAinv.begin(), ptrAinv.end());
  thrust::device_vector<int> info(ncells, 0);

  // Invert A blockwise
  cublasStatus_t status = cublasDmatinvBatched(
      handle, n, ptrA_device.data().get(), n, ptrAinv_device.data().get(), n,
      info.data().get(), ncells);

  // Now do batch gemv with Ainv

  const double alpha = 1.0;
  const double beta = 0.0;

  status = cublasDgemvBatched(
      handle, CUBLAS_OP_N, n, n, &alpha, ptrAinv_device.data().get(), n,
      b.array().data().get(), 1,        // Input vectors
      &beta, u.array().data().get(), 1, // Output vectors
      ncells);

  cublasDestroy(handle);
}
