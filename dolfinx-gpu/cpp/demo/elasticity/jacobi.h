#pragma once

#include <cmath>
#include <cstdint>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>

#include <dolfinx/common/MPI.h>

#include "cg.h"


// invert the diagonal of a matrix with clamped boundary conditions
template <typename Scalar>
struct invert_jacobi_diagonal{
  __host__ __device__ Scalar operator()(Scalar diagonal, std::int8_t bc_marker) const{
    if (bc_marker)
      return Scalar(1); // set the value to 1 if it is clamped
    else
      return Scalar(1) / diagonal; // invert the diagonal value
  }
};


// compute the residual r = D^{-1} * (b - Ax)
template <typename T>
__global__ void preconditioned_residual_kernel(
  std::size_t n,
  const T* b,
  const T* Ax,
  const T* diagonal_inverse,
  T* residual)
{
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < n)
  {
    residual[i] = diagonal_inverse[i] * (b[i] - Ax[i]);
  }
}


// compute the preconditioned residual r = D^{-1} * (b - Ax)
template <typename Vector>
void compute_preconditioned_residual(
  const Vector& b,
  const Vector& Ax,
  const Vector& diagonal_inverse,
  Vector& residual)
{
  using T = typename Vector::value_type;
  const std::size_t owned_size = b.index_map()->size_local() * b.bs();

  const std::size_t block_size = 256;
  const std::size_t num_blocks = static_cast<std::size_t>(owned_size + block_size - 1) / block_size;

  preconditioned_residual_kernel<<<num_blocks, block_size>>>(
    owned_size,
    thrust::raw_pointer_cast(b.array().data()),
    thrust::raw_pointer_cast(Ax.array().data()),
    thrust::raw_pointer_cast(diagonal_inverse.array().data()),
    thrust::raw_pointer_cast(residual.array().data())
  );
}


// Jacobi smoother
// one weighted jacobi step: x = x + omega * D^{-1} * (b - Ax)
template <typename Vector, typename Operator>
void jacobi_smooth(
  const Operator& A,
  Vector& x,
  const Vector& b,
  const Vector& diagonal_inverse,
  Vector& Ax,
  Vector& residual,
  int num_steps,
  typename Vector::value_type omega)
{
  using Scalar = typename Vector::value_type;
  const std::size_t owned_size = x.bs() * x.index_map()->size_local();

  for (int step = 0; step < num_steps; ++step){
    A(Ax, x); // compute Ax

    // compute residual = b - Ax
    compute_preconditioned_residual(b, Ax, diagonal_inverse, residual);

    // add result to x with damping factor omega
    thrust::transform(
      thrust::device,
      residual.array().begin(),
      residual.array().begin() + owned_size,
      x.array().begin(),
      x.array().begin(),
      elasticity::axpyOperation<Scalar>{omega}
    );
  }
};


template <typename Scalar>
struct chebyshev_blend
{
  Scalar omega;

  __host__ __device__ 
  Scalar operator()(Scalar previous, Scalar current) const
  {
    return (Scalar(1) - omega) * previous + omega * current;
  }
};


template <typename T>
__global__ void chebyshev_update_kernel(
  std::size_t n,
  const T* previous,
  const T* current,
  const T* residual,
  T* next,
  T omega,
  T scale)
{
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < n)
  {
    next[i] = (T(1) - omega) * previous[i] + omega * current[i] + omega * scale * residual[i];
  }
}


template <typename Vector>
void chebyshev_update(
  const Vector& residual,
  const Vector& previous,
  const Vector& current,
  Vector& next,
  typename Vector::value_type omega,
  typename Vector::value_type scale)
{
  using T = typename Vector::value_type;
  const std::size_t owned_size = residual.bs() * residual.index_map()->size_local();

  const std::size_t block_size = 256;
  const std::size_t num_blocks = static_cast<std::size_t>(owned_size + block_size - 1) / block_size;

  chebyshev_update_kernel<<<num_blocks, block_size>>>(
    owned_size,
    thrust::raw_pointer_cast(previous.array().data()),
    thrust::raw_pointer_cast(current.array().data()),
    thrust::raw_pointer_cast(residual.array().data()),
    thrust::raw_pointer_cast(next.array().data()),
    omega,
    scale
  );
}


// Chebyshev smoother
template <typename Vector, typename Operator>
void chebyshev_smooth(
  const Operator& A,
  Vector& x,
  const Vector& b,
  const Vector& diagonal_inverse,
  Vector& Ax,
  Vector& residual,
  Vector& previous,
  Vector& next,
  int degree,
  typename Vector::value_type lambda_min,
  typename Vector::value_type lambda_max)
{
  using T = typename Vector::value_type;
  const std::int32_t owned_size = x.index_map()->size_local() * x.bs();

  if (degree < 1)
    return;

  // Chebyshev coefficients
  const T scale = T(2) / (lambda_max + lambda_min);
  const T alpha = T(1) - scale * lambda_min;
  const T mu = T(1) / alpha;
  const T omega_product = T(2) / alpha;

  T c_previous = T(1);
  T c_current = mu;

  // save x_0
  thrust::copy(thrust::device, x.array().begin(), x.array().begin() + owned_size, previous.array().begin());

  // r = b - Ax
  A(Ax, x);

  compute_preconditioned_residual(b, Ax, diagonal_inverse, next);

  // x_1 = x_0 + scale * z
  thrust::transform(thrust::device, next.array().begin(), next.array().begin() + owned_size, previous.array().begin(), x.array().begin(), elasticity::axpyOperation<T>{scale});

  // remaining Chebyshev stages
  for (int k = 1; k < degree; ++k)
  {
    // r = b - Ax_k
    A(Ax, x);

    compute_preconditioned_residual(b, Ax, diagonal_inverse, residual);

    const T c_next = T(2) * mu * c_current - c_previous;
    const T omega = omega_product * c_current / c_next;

    chebyshev_update(residual, previous, x, next, omega, scale);

    // previous <- x
    thrust::copy(thrust::device, x.array().begin(), x.array().begin() + owned_size, previous.array().begin());

    // x <- next
    thrust::copy(thrust::device, next.array().begin(), next.array().begin() + owned_size, x.array().begin());

    c_previous = c_current;
    c_current = c_next;
  }
}


// lamda_max helper function
template <typename Vector, typename Operator, typename Marker>
typename Vector::value_type estimate_lambda_max(
  const Operator& A,
  const Vector& diagonal_inverse,
  const Marker& bc_marker,
  Vector& v,
  Vector& temp,
  Vector& Av,
  Vector& y,
  int iterations = 10)
{
  using T = typename Vector::value_type;

  auto indicies = thrust::make_counting_iterator<std::size_t>(0);

  thrust::transform(thrust::device, indicies, indicies + v.array().size(), bc_marker.begin(), v.array().begin(), 
    [] __host__ __device__ (std::size_t i, std::int8_t marker) { return marker ? T(0) : std::sin(T(i+1)); });

  auto dot = [](const Vector& a, const Vector& b) {
    const std::int32_t size = a.bs() * a.index_map()->size_local();

    const T local = thrust::inner_product(thrust::device, a.array().begin(), a.array().begin() + size, b.array().begin(), T(0));

    T global;

    MPI_Allreduce(&local, &global, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM, a.index_map()->comm());

    return global;
  };

  T norm = std::sqrt(dot(v, v));

  thrust::transform(thrust::device, v.array().begin(), v.array().end(), v.array().begin(), [norm] __host__ __device__ (T value) { return value / norm; });

  T lambda = T(0);

  for (int k = 0; k < iterations; ++k)
  {
    // temp = D^{-1/2} * v
    thrust::transform(thrust::device, diagonal_inverse.array().begin(), diagonal_inverse.array().end(), v.array().begin(), temp.array().begin(), [] __host__ __device__ (T d_inv, T v_val) { return std::sqrt(d_inv) * v_val; });

    // Av = A(temp);
    A(Av, temp);

    // y = D^{-1/2} * Av
    thrust::transform(thrust::device, diagonal_inverse.array().begin(), diagonal_inverse.array().end(), Av.array().begin(), y.array().begin(), [] __host__ __device__ (T d_inv, T Av_val) { return std::sqrt(d_inv) * Av_val; });

    lambda = dot(v, y);
    norm = std::sqrt(dot(y, y));

    thrust::transform(thrust::device, y.array().begin(), y.array().end(), v.array().begin(), [norm] __host__ __device__ (T value) { return value / norm; });
  }

  return lambda;
}

// residual norm helper function
template <typename Vector, typename Operator>
typename Vector::value_type residual_norm(
  const Operator& A,
  Vector& x,
  const Vector& b,
  Vector& Ax,
  Vector& residual)
{
  using Scalar = typename Vector::value_type;

  A(Ax, x); // compute Ax

  // compute residual = b - Ax
  thrust::transform(
    thrust::device,
    b.array().begin(),
    b.array().end(),
    Ax.array().begin(),
    residual.array().begin(),
    thrust::minus<Scalar>()
  );

  // compute norm of residual
  const std::int32_t local_size = residual.bs() * residual.index_map()->size_local();

  const Scalar local_norm_squared = thrust::inner_product(
    thrust::device,
    residual.array().begin(),
    residual.array().begin() + local_size,
    residual.array().begin(),
    Scalar(0)
  );

  Scalar norm_squared;
  MPI_Allreduce(&local_norm_squared, &norm_squared, 1, dolfinx::MPI::mpi_t<Scalar>, MPI_SUM, residual.index_map()->comm());

  return std::sqrt(norm_squared);
};