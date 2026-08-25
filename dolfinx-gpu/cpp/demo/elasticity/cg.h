// conjugate gradient solver
#pragma once

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/common/MPI.h>

#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <memory>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "timings.h"

namespace elasticity
{
    // compute the linear combination of two vectors i.e. result = alpha * x + y
    template <typename T>
    struct axpyOperation{
        T alpha;

        __host__ __device__
        T operator()(T x_i, T y_i) const{
            return alpha * x_i + y_i;
        }
    };

    template <typename Vector>
    class CGSolver{
        public:
            using T = typename Vector::value_type;

            // constructor that creates three temporary vectors required in CG
            // _map describes how many finite element nodes are owned and how the vector is laid out
            // _bs is the number of components per node (3 for 3D elasticity)
            CGSolver(std::shared_ptr<const dolfinx::common::IndexMap> _map, int _bs)
            : _r(std::make_unique<Vector>(_map, _bs)), // residual vector
              _p(std::make_unique<Vector>(_map, _bs)), // search direction vector
              _y(std::make_unique<Vector>(_map, _bs)) // matrix-vector product vector
            {
            }

            // solver settings
            // max number of iterations
            void set_max_iterations(int max_iter){
                if (max_iter <= 0)
                    throw std::runtime_error("Maximum number of iterations must be positive");

                _max_iter = max_iter;
            }


            // relative tolerance for convergence
            void set_tolerance(T tol){
                if (tol <= T(0))
                    throw std::runtime_error("Tolerance must be positive");

                _rtol = tol;
            }


            template <typename Operator, typename Preconditioner>
            int solve(Operator& A, Vector& x, const Vector& b, Preconditioner& M){
                if (x.array().size() != b.array().size())
                    throw std::runtime_error("Vectors x and b must be the same size");

                time_gpu(solve_timings.outer_matvec, [&](){
                    // compute initial residual r_0 = b - A*x_0
                    A(*_y, x); // _Ap = A*x
                    axpy(*_r, T(-1), *_y, b); // _r = b - A*x
                });

                T residual_squared0;
                time_gpu(solve_timings.cg_dots, [&](){
                    residual_squared0 = dot(*_r, *_r);
                });

                // compute the tolerance based on the initial residual norm
                const T rtol2 = _rtol * _rtol;

                if (residual_squared0 < T(1e-24)) // if the initial guess is already the solution
                    return 0;

                M(*_y, *_r); // z0 = M^{-1} * r0

                time_gpu(solve_timings.cg_vector_ops, [&](){
                    // copy z0 to p0
                    copy(*_p, *_y); // p0 = z0
                });

                T rho;
                time_gpu(solve_timings.cg_dots, [&](){
                    rho = dot(*_r, *_y); // rho0 = r0^T * z0
                });

                if (rho <= T(0)){
                    throw std::runtime_error("Preconditioner is not positive definite");
                }
                    
                // main CG iteration loop
                int k = 0;
                while (k < _max_iter){
                    time_gpu(solve_timings.outer_matvec, [&](){
                        // compute matrix-vector product Ap = A*p
                        A(*_y, *_p); // _y = A*p
                    });

                    // check that p^TAp is positive i.e. A is positive definite
                    T pAp;
                    time_gpu(solve_timings.cg_dots, [&](){
                        pAp = dot(*_p, *_y);
                    });

                    if (!(pAp > T(0))){
                        throw std::runtime_error("Matrix A is not positive definite");
                    }

                    // compute alpha = (r^T * r) / (p^T * Ap)
                    const T alpha = rho / pAp;

                    time_gpu(solve_timings.cg_vector_ops, [&](){
                        // update solution x = x + alpha * p
                        axpy(x, alpha, *_p, x); // x = alpha * p + x

                        // update residual r = r - alpha * Ap
                        axpy(*_r, T(-alpha), *_y, *_r); // r = -alpha * Ap + r
                    });

                    T residual_squared;
                    time_gpu(solve_timings.cg_dots, [&](){
                        residual_squared = dot(*_r, *_r);
                    });

                    std::cout
                        << "Outer CG iteration " << k + 1
                        << ", relative residual = "
                        << std::sqrt(residual_squared / residual_squared0)
                        << "\n";

                    ++k; // increment iteration counter

                    // check for convergence
                    if (residual_squared / residual_squared0 < rtol2)
                        break;

                    M(*_y, *_r); // z = M^{-1} * r

                    // compute new residual norm
                    T rho_new;
                    time_gpu(solve_timings.cg_dots, [&](){
                        rho_new = dot(*_r, *_y); // rho_new = r^T * z
                    });

                    if (rho_new <= T(0)){
                        throw std::runtime_error("Preconditioner is not positive definite");
                    }

                    // compute beta = (r_new^T * r_new) / (r^T * r)
                    const T beta = rho_new / rho;

                    time_gpu(solve_timings.cg_vector_ops, [&](){
                        // update search direction p = z + beta * p
                        axpy(*_p, beta, *_p, *_y); // p = beta * p + z
                    });

                    // update residual norm for next iteration
                    rho = rho_new;
                }
            return k;
            }


        private:
            // solver parameters
            int _max_iter = 1000;
            T _rtol = T(1e-8);

            // number of owned entries on this MPI rank
            static std::int32_t local_size(const Vector& x)
            {
                return x.bs() * x.index_map()->size_local();
            }

            // compute the dot product of two vectors
            static T dot(const Vector& x, const Vector& y){
                const std::int32_t size = x.bs() * x.index_map()->size_local();

                if (size != y.bs() * y.index_map()->size_local())
                    throw std::runtime_error("Vectors must be the same size for dot product");
                
                const T local = thrust::inner_product(thrust::device, x.array().begin(), x.array().begin() + size, y.array().begin(), T(0));

                T result;

                MPI_Allreduce(&local, &result, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM, x.index_map()->comm());

                return result;
            }

            // copy the contents of source vector to destination vector
            static void copy(Vector& destination, const Vector& source){
                auto& dest_array = destination.array();
                const auto& source_array = source.array();

                if (dest_array.size() != source_array.size())
                    throw std::runtime_error("Vectors must be the same size for copy");
                
                const std::int32_t size = local_size(destination);
                thrust::copy(thrust::device, source_array.begin(), source_array.begin() + size, dest_array.begin());
            }

            // compute the linear combination of two vectors i.e. result = alpha * x + y
            static void axpy(Vector& result, T alpha, const Vector& x, const Vector& y){
                auto& result_values = result.array();
                const auto& x_values = x.array();
                const auto& y_values = y.array();

                if (result_values.size() != x_values.size() || result_values.size() != y_values.size()){
                    throw std::runtime_error("Vectors must be the same size for axpy");
                }

                const std::int32_t size = local_size(result);
                thrust::transform(thrust::device, x_values.begin(), x_values.begin() + size, y_values.begin(), result_values.begin(), axpyOperation<T>{alpha});
            }


            // working vectors
            std::unique_ptr<Vector> _r; // residual vector
            std::unique_ptr<Vector> _p; // search direction vector
            std::unique_ptr<Vector> _y; // matrix-vector product vector
    };
} // namespace elasticity