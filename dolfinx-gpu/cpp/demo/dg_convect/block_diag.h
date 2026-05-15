/// Solve block diagonal mass-matrix system on GPU
#include <cublas_v2.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/la/Vector.h>
#include <thrust/device_vector.h>

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

  // Create a cublas context
  cublasHandle_t handle;
  cublasCreate(&handle);

  // FIXME: get from element/dofmap
  int n = 4; // 4 dofs per cell
  int ncells = A_cpu.size() / (n * n);

  // Allocate memory for A, A^-1
  thrust::device_vector<T> A(A_cpu.begin(), A_cpu.end());
  thrust::device_vector<T> Ainv(A.size());

  std::vector<const T*> ptrA(ncells);
  std::vector<T*> ptrAinv(ncells);
  for (int i = 0; i < ncells; ++i)
  {
    ptrA[i] = A.data().get() + n * n * i;
    ptrAinv[i] = Ainv.data().get() + n * n * i;
  }
  thrust::device_vector<const T*> ptrA_device(ptrA.begin(), ptrA.end());
  thrust::device_vector<T*> ptrAinv_device(ptrAinv.begin(), ptrAinv.end());
  thrust::device_vector<int> info(ncells, 0);

  // Invert A blockwise
  cublasStatus_t status;
  if constexpr (std::is_same_v<double, T>)
  {
    status = cublasDmatinvBatched(handle, n, ptrA_device.data().get(), n,
                                  ptrAinv_device.data().get(), n,
                                  info.data().get(), ncells);
  }
  else if constexpr (std::is_same_v<float, T>)
  {
    status = cublasSmatinvBatched(handle, n, ptrA_device.data().get(), n,
                                  ptrAinv_device.data().get(), n,
                                  info.data().get(), ncells);
  }
  else
    throw std::runtime_error("Unsupported scalar type");

  // TODO: check status and info

  cublasDestroy(handle);

  return Ainv;
}

/// Solve A.u = b for a block diagonal system
/// Assumes DG style matrix laid out in cell order in A
/// and that dofs of b and u follow the same layout
/// @param A matrix in DG form
/// @param b RHS input vector
/// @param u Solution vector
template <dolfinx::scalar T>
void solve_block_diag_system(
    const thrust::device_vector<T>& Ainv,
    const dolfinx::la::Vector<T, thrust::device_vector<T>>& b,
    dolfinx::la::Vector<T, thrust::device_vector<T>>& u)
{
  // Create a cublas context
  cublasHandle_t handle;
  cublasCreate(&handle);

  int n = 4;
  int ncells = b.array().size() / n;

  std::vector<const T*> ptrAinv(ncells);
  std::vector<const T*> ptrb(ncells);
  std::vector<T*> ptru(ncells);
  for (int i = 0; i < ncells; ++i)
  {
    ptrAinv[i] = Ainv.data().get() + n * n * i;
    ptrb[i] = b.array().data().get() + n * i;
    ptru[i] = u.array().data().get() + n * i;
  }
  thrust::device_vector<const T*> ptrAinv_device(ptrAinv.begin(),
                                                 ptrAinv.end());
  thrust::device_vector<const T*> ptrb_device(ptrb.begin(), ptrb.end());
  thrust::device_vector<T*> ptru_device(ptru.begin(), ptru.end());

  // NB use transpose operator, since original A was Row Major
  const double alpha = 1.0;
  const double beta = 0.0;
  cublasStatus_t status = cublasDgemvBatched(
      handle, CUBLAS_OP_T, n, n, &alpha, ptrAinv_device.data().get(), n,
      ptrb_device.data().get(), 1, &beta, ptru_device.data().get(), 1, ncells);

  cublasDestroy(handle);
}
