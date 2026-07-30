#pragma once

#include <cmath>
#include <cstdint>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>

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

  for (int step = 0; step < num_steps; ++step){
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

    // multiply residual by inverse diagonal
    thrust::transform(
      thrust::device,
      residual.array().begin(),
      residual.array().end(),
      diagonal_inverse.array().begin(),
      residual.array().begin(),
      thrust::multiplies<Scalar>()
    );

    // add result to x with damping factor omega
    thrust::transform(
      thrust::device,
      residual.array().begin(),
      residual.array().end(),
      x.array().begin(),
      x.array().begin(),
      elasticity::axpyOperation<Scalar>{omega}
    );
  }
};


// residual norm helper function
template <typename Vector, typename Operator>
typename Vector::value_type residual_norm(
  const Operator& A,
  const Vector& x,
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
  const Scalar norm_squared = thrust::inner_product(
    thrust::device,
    residual.array().begin(),
    residual.array().end(),
    residual.array().begin(),
    Scalar(0)
  );

  return std::sqrt(norm_squared);
};