// conjugate gradient solver
#pragma once

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/Vector.h>

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
              _y(std::make_unique<Vector>(_map, _bs)), // matrix-vector product vector
              _diag_inv(std::make_unique<Vector>(_map, _bs)) // inverse of the diagonal of A
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

            void set_diag_inverse(const Vector& diag_inv){
                if (diag_inv.array().size() != _diag_inv->array().size())
                    throw std::runtime_error("Diagonal inverse vector must be the same size as the solver's diagonal inverse vector");
                
                copy(*_diag_inv, diag_inv);
            }

            template <typename Operator>
            int solve(Operator& A, Vector& x, const Vector& b, bool jacobi = true){
                if (x.array().size() != b.array().size())
                    throw std::runtime_error("Vectors x and b must be the same size");

                // compute initial residual r_0 = b - A*x_0
                A(*_y, x); // _Ap = A*x
                axpy(*_r, T(-1), *_y, b); // _r = b - A*x

                const T residual_squared0 = dot(*_r, *_r);

                if (jacobi){
                    // apply Jacobi preconditioner
                    pointwise_mult(*_p, *_r, *_diag_inv); // p_0 = D^{-1} * r_0
                }
                else{
                    // set initial search direction
                    copy(*_p, *_r); // p_0 = r_0
                }

                // compute initial residual norm i.e. residual squared
                const T rnorm0 = dot(*_r, *_p); // rnorm = r^T * D^{-1} * r
                T rnorm = rnorm0; // current residual norm

                // compute the tolerance based on the initial residual norm
                const T rtol2 = _rtol * _rtol;

                if (residual_squared0 < T(1e-24)) // if the initial guess is already the solution
                    return 0;
                    
                // main CG iteration loop
                int k = 0;
                while (k < _max_iter){
                    // compute matrix-vector product Ap = A*p
                    A(*_y, *_p); // _y = A*p

                    // check that p^TAp is positive i.e. A is positive definite
                    const T pAp = dot(*_p, *_y);
                    if (!(pAp > T(0))){
                        throw std::runtime_error("Matrix A is not positive definite");
                    }

                    // compute alpha = (r^T * r) / (p^T * Ap)
                    const T alpha = rnorm / pAp;

                    // update solution x = x + alpha * p
                    axpy(x, alpha, *_p, x); // x = alpha * p + x

                    // update residual r = r - alpha * Ap
                    axpy(*_r, T(-alpha), *_y, *_r); // r = -alpha * Ap + r

                    if (jacobi){
                        // y = D^{-1} * r
                        pointwise_mult(*_y, *_r, *_diag_inv);
                    }
                    else{
                        // y = r
                        copy(*_y, *_r);
                    }

                    // compute new residual norm
                    const T rnorm_new = dot(*_r, *_y); // rnorm_new = r^T * D^{-1} * r

                    ++k; // increment iteration counter

                    // check for convergence
                    if (rnorm_new / rnorm0 < rtol2)
                        break;

                    // compute beta = (r_new^T * r_new) / (r^T * r)
                    const T beta = rnorm_new / rnorm;

                    // update search direction 
                    // with jacobi: p = D^{-1} * r + beta * p
                    // without jacobi: p = r + beta * p
                    axpy(*_p, beta, *_p, *_y); // p = beta * p + r

                    // update residual norm for next iteration
                    rnorm = rnorm_new;
                    std::cout << "residual norm = " << std::setprecision(17) << rnorm << "\n";
                }
            return k;
            }


        private:
            // solver parameters
            int _max_iter = 1000;
            T _rtol = T(1e-8);

            // compute the dot product of two vectors
            static T dot(const Vector& x, const Vector& y){
                const auto& x_array = x.array();
                const auto& y_array = y.array();

                if (x_array.size() != y_array.size())
                    throw std::runtime_error("Vectors must be the same size for dot product");
                
                return thrust::inner_product(thrust::device, x_array.begin(), x_array.end(), y_array.begin(), T(0));
            }

            // copy the contents of source vector to destination vector
            static void copy(Vector& destination, const Vector& source){
                auto& dest_array = destination.array();
                const auto& source_array = source.array();

                if (dest_array.size() != source_array.size())
                    throw std::runtime_error("Vectors must be the same size for copy");
                
                thrust::copy(thrust::device, source_array.begin(), source_array.end(), dest_array.begin());
            }

            // compute the linear combination of two vectors i.e. result = alpha * x + y
            static void axpy(Vector& result, T alpha, const Vector& x, const Vector& y){
                auto& result_values = result.array();
                const auto& x_values = x.array();
                const auto& y_values = y.array();

                if (result_values.size() != x_values.size() || result_values.size() != y_values.size()){
                    throw std::runtime_error("Vectors must be the same size for axpy");
                }

                thrust::transform(thrust::device, x_values.begin(), x_values.end(), y_values.begin(), result_values.begin(), axpyOperation<T>{alpha});
            }

            // compute the pointwise multiplication of two vectors i.e. result = x * y
            static void pointwise_mult(Vector& result, const Vector& x, const Vector& y){
                auto& result_values = result.array();
                const auto& x_values = x.array();
                const auto& y_values = y.array();

                if (result_values.size() != x_values.size() || result_values.size() != y_values.size()){
                    throw std::runtime_error("Vectors must be the same size for pointwise multiplication");
                }

                thrust::transform(thrust::device, x_values.begin(), x_values.end(), y_values.begin(), result_values.begin(), thrust::multiplies<T>());
            }

            // working vectors
            std::unique_ptr<Vector> _r; // residual vector
            std::unique_ptr<Vector> _p; // search direction vector
            std::unique_ptr<Vector> _y; // matrix-vector product vector
            std::unique_ptr<Vector> _diag_inv; // inverse of the diagonal of A
    };
} // namespace elasticity