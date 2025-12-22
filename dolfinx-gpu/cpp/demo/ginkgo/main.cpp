// # Poisson equation
//
// This demo illustrates how to:
//
// * Solve a linear partial differential equation
// * Create and apply Dirichlet boundary conditions
// * Define Expressions
// * Define a FunctionSpace
//
// ## Equation and problem definition
//
// The Poisson equation is the canonical elliptic partial differential
// equation.  For a domain $\Omega \subset \mathbb{R}^n$ with boundary
// $\partial \Omega = \Gamma_{D} \cup \Gamma_{N}$, the Poisson equation
// with particular boundary conditions reads:
//
// \begin{align*}
//    - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
//      u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
//      \nabla u \cdot n &= g \quad {\rm on} \ \Gamma_{N}. \\
// \end{align*}
//
// Here, $f$ and $g$ are input data and $n$ denotes the outward directed
// boundary normal. The most standard variational form of Poisson
// equation reads: find $u \in V$ such that
//
// $$
//    a(u, v) = L(v) \quad \forall \ v \in V,
// $$
// where $V$ is a suitable function space and
//
// \begin{align*}
//    a(u, v) &= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x, \\
//    L(v)    &= \int_{\Omega} f v \, {\rm d} x
//    + \int_{\Gamma_{N}} g v \, {\rm d} s.
// \end{align*}
//
// The expression $a(u, v)$ is the bilinear form and $L(v)$ is the
// linear form. It is assumed that all functions in $V$ satisfy the
// Dirichlet boundary conditions ($u = 0 \ {\rm on} \ \Gamma_{D}$).
//
// In this demo, we shall consider the following definitions of the
// input functions, the domain, and the boundaries:
//
// * $\Omega = [0,1] \times [0,1]$ (a unit square)
// * $\Gamma_{D} = \{(0, y) \cup (1, y) \subset \partial \Omega\}$
// (Dirichlet boundary)
// * $\Gamma_{N} = \{(x, 0) \cup (x, 1) \subset \partial \Omega\}$
// (Neumann boundary)
// * $g = \sin(5x)$ (normal derivative)
// * $f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)$ (source term)
//
//
// ## Implementation
//
// The implementation is split in two files: a file containing the
// definition of the variational forms expressed in UFL and a C++ file
// containing the actual solver.
//
// Running this demo requires the files: {download}`demo_poisson/main.cpp`,
// {download}`demo_poisson/poisson.py` and
// {download}`demo_poisson/CMakeLists.txt`.
//
// ### UFL code
//
// The UFL code is implemented in {download}`demo_poisson/poisson.py`.
// ````{admonition} UFL code implemented in Python
// :class: dropdown
// ![ufl-code]
// ````
//
// ### C++ program
//
// The main solver is implemented in the
// {download}`demo_poisson/main.cpp` file.
//
// At the top we include the DOLFINx header file and the generated
// header file "Poisson.h" containing the variational forms for the
// Poisson equation.  For convenience we also include the DOLFINx
// namespace.

#include "poisson.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <utility>
#include <vector>

#include <ginkgo/ginkgo.hpp>
#include <thrust/device_vector.h>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

// Then follows the definition of the coefficient functions (for $f$ and
// $g$), which are derived from the {cpp:class}`Expression` class in
// DOLFINx

// Inside the `main` function, we begin by defining a mesh of the
// domain. As the unit square is a very standard domain, we can use a
// built-in mesh provided by the {cpp:class}`UnitSquareMesh` factory. In
// order to create a mesh consisting of 32 x 32 squares with each square
// divided into two triangles, and the finite element space (specified
// in the form file) defined relative to this mesh, we do as follows:

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {10, 10, 10},
        mesh::CellType::tetrahedron, part));

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(element)));

    //  Next, we define the variational formulation by initializing the
    //  bilinear and linear forms ($a$, $L$) using the previously
    //  defined {cpp:class}`FunctionSpace` `V`.  Then we can create the
    //  source and boundary flux term ($f$, $g$) and attach these to the
    //  linear form.

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(0.01);
    auto u = std::make_shared<fem::Function<T>>(V);

    // Define variational forms
    fem::Form<T> a = fem::create_form<T>(*form_poisson_a, {V, V}, {},
                                         {{"kappa", kappa}}, {}, {});
    fem::Form<T> L
        = fem::create_form<T>(*form_poisson_L, {V}, {{"f", u}}, {}, {}, {});

    //  Now, the Dirichlet boundary condition ($u = 0$) can be created
    //  using the class {cpp:class}`DirichletBC`. A
    //  {cpp:class}`DirichletBC` takes two arguments: the value of the
    //  boundary condition, and the part of the boundary on which the
    //  condition applies. In our example, the value of the boundary
    //  condition (0) can represented using a {cpp:class}`Function`,
    //  and the Dirichlet boundary is defined by the indices of degrees
    //  of freedom to which the boundary condition applies. The
    //  definition of the Dirichlet boundary condition then looks as
    //  follows:

    // Define boundary condition

    std::vector facets = mesh::locate_entities_boundary(
        *mesh, 2,
        [](auto x)
        {
          using U = typename decltype(x)::value_type;
          constexpr U eps = 1.0e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto x0 = x(0, p);
            if (std::abs(x0) < eps or std::abs(x0 - 1.0) < eps)
              marker[p] = true;
          }
          return marker;
        });
    std::vector bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets);
    fem::DirichletBC<T> bc(0, bdofs, V);

    u->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            f.push_back(10 * std::exp(-(dx + dy) / 0.02));
          }

          return {f, {f.size()}};
        });

    std::cout << "u.norm [init] = " << dolfinx::la::norm(*u->x()) << "\n";

    //  Now, we have specified the variational forms and can consider
    //  the solution of the variational problem. First, we need to
    //  define a {cpp:class}`Function` `u` to store the solution. (Upon
    //  initialization, it is simply set to the zero function.) Next, we
    //  can call the `solve` function with the arguments `a == L`, `u`
    //  and `bc` as follows:

    la::SparsityPattern sp = fem::create_sparsity_pattern(a);
    sp.finalize();
    la::MatrixCSR<T> A(sp);
    la::Vector<T> b(L.function_spaces()[0]->dofmap()->index_map,
                    L.function_spaces()[0]->dofmap()->index_map_bs());

    fem::assemble_matrix(A.mat_add_values(), a, {bc});
    A.scatter_rev();
    fem::set_diagonal<T>(A.mat_set_values(), *V, {bc});

    la::MatrixCSR<T, thrust::device_vector<T>,
                  thrust::device_vector<std::int32_t>,
                  thrust::device_vector<std::int32_t>>
        A_device(A);

    std::ranges::fill(b.array(), 0);
    fem::assemble_vector(b.array(), L);
    fem::apply_lifting(b.array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    bc.set(b.array(), std::nullopt);
    std::cout << "b.norm = " << dolfinx::la::norm(b) << "\n";

    // Copy RHS to device
    la::Vector<T, thrust::device_vector<T>> b_device(b);
    la::Vector<T, thrust::device_vector<T>> u_device(*(u->x()));

    // Solve here A.u = b

    auto executor = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    int nnz = A_device.cols().size();
    int nrows1 = A_device.row_ptr().size();

    std::int64_t nrows = b.index_map()->size_local();
    using vec = gko::matrix::Dense<>;
    using val_array = gko::array<T>;

    auto b_gko = vec::create(
        executor, gko::dim<2>(nrows, 1),
        val_array::view(executor, nrows, b_device.array().data().get()), 1);

    auto u_gko = vec::create(
        executor, gko::dim<2>(nrows, 1),
        val_array::view(executor, nrows, u_device.array().data().get()), 1);

    using mtx = gko::matrix::Csr<>;
    auto mat = mtx::create(executor, gko::dim<2>(nrows), nnz);
    mtx::value_type* values = mat->get_values();
    mtx::index_type* row_ptr = mat->get_row_ptrs();
    mtx::index_type* col_idx = mat->get_col_idxs();

    thrust::copy(A_device.values().begin(), A_device.values().end(), values);
    thrust::copy(A_device.cols().begin(), A_device.cols().end(), col_idx);
    thrust::copy(A_device.row_ptr().begin(), A_device.row_ptr().end(), row_ptr);

    std::cout << "mat contains " << mat->get_num_stored_elements() << "\n";

    std::cout << "u.norm [0] = " << dolfinx::la::norm(*u->x()) << "\n";
    mat->apply(b_gko, u_gko);

    spdlog::info("Pointer to u_gko at {}", (std::size_t)(u_gko->get_values()));

    std::cout << "u.norm [1] = " << dolfinx::la::norm(*u->x()) << "\n";

    dolfinx::common::Timer tsolve1("Set up Ginkgo");

    using cg = gko::solver::Cg<T>;
    using bj = gko::preconditioner::Jacobi<T, std::int32_t>;
    const gko::remove_complex<T> reduction_factor = 1e-7;
    auto solver
        = cg::build()
              .with_criteria(
                  gko::stop::Iteration::build().with_max_iters(100),
                  gko::stop::ResidualNorm<T>::build().with_reduction_factor(
                      reduction_factor))
              .with_preconditioner(bj::build())
              .on(executor)
              ->generate(
                  clone(executor, mat)); // copy the matrix to the executor

    tsolve1.stop();
    tsolve1.flush();

    io::XDMFFile file(MPI_COMM_WORLD, "u.xdmf", "w");
    file.write_mesh(*mesh);

    for (int i = 0; i < 20; ++i)
    {

      dolfinx::common::Timer tsolve2("Tsolve2");
      solver->apply(b_gko, u_gko);

      // Copy solution back to CPU
      thrust::copy(u_device.array().begin(), u_device.array().end(),
                   u->x()->array().begin());
      std::cout << "u.norm [after] = " << dolfinx::la::norm(*u->x()) << "\n";

      std::ranges::fill(b.array(), 0);
      fem::assemble_vector(b.array(), L);
      fem::apply_lifting(b.array(), {a}, {{bc}}, {}, T(1));
      b.scatter_rev(std::plus<T>());
      bc.set(b.array(), std::nullopt);
      std::cout << "b.norm = " << dolfinx::la::norm(b) << "\n";

      // Copy RHS back to device
      thrust::copy(b.array().begin(), b.array().end(),
                   b_device.array().begin());

      //   // Update ghost values before output
      //   // u->x()->scatter_fwd();

      file.write_function<T>(*u, static_cast<double>(i));
    }

    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}
