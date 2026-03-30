// Copyright (C) 2026 Chris Richardson
// FEniCS Project
// SPDX: MIT

#include "poisson.h"
#include <basix/finite-element.h>
#include <boost/program_options.hpp>
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
namespace po = boost::program_options;
using T = double;
using U = typename dolfinx::scalar_value_t<T>;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  // Define command line options
  po::options_description desc("Options");
  desc.add_options()("help,h", "Print usage message")(
      "direct", po::value<bool>()->default_value(false),
      "Compute platform (cpu or gpu)");

  // Parse command line options
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run(),
            vm);

  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << "DOLFINx Ginkgo demo\n-----------------\n";
    std::cout << desc << std::endl;
    return 0;
  }
  bool direct_solver = vm["direct"].as<bool>();

  if (direct_solver)
    std::cout << "Direct solver (LU)\n";
  else
    std::cout << "Iterative solver (CG)\n";

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {20, 20, 20},
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

#if defined(USE_HIP)
    auto executor = gko::HipExecutor::create(0, gko::OmpExecutor::create());
#elif defined(USE_CUDA)
    auto executor = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
#endif

    int nnz = A_device.cols().size();

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

    dolfinx::common::Timer tsolve1("[Set up Ginkgo]");

    std::unique_ptr<gko::LinOp> solver;

    if (direct_solver)
    {
      auto solver_factory
          = gko::experimental::solver::Direct<double, int>::build()
                .with_factorization(
                    gko::experimental::factorization::Lu<double, int>::build()
                        .on(executor))
                .on(executor);

      solver = solver_factory->generate(gko::share(std::move(mat)));
    }
    else
    {
      using cg = gko::solver::Cg<T>;
      using bj = gko::preconditioner::Jacobi<T, std::int32_t>;
      const gko::remove_complex<T> reduction_factor = 1e-7;
      solver
          = cg::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(100),
                    gko::stop::ResidualNorm<T>::build().with_reduction_factor(
                        reduction_factor))
                .with_preconditioner(bj::build())
                .on(executor)
                ->generate(gko::share(std::move(mat)));
    }

    tsolve1.stop();
    tsolve1.flush();

    io::XDMFFile file(MPI_COMM_WORLD, "u.xdmf", "w");
    file.write_mesh(*mesh);

    for (int i = 0; i < 20; ++i)
    {
      {
        dolfinx::common::Timer tsolve2("[Call solver]");
        solver->apply(b_gko, u_gko);
        executor->synchronize(); // force GPU completion before stopping timer
      }
      {
        dolfinx::common::Timer tsolve3("[Update and save]");

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

        file.write_function<T>(*u, static_cast<double>(i));
      }
    }

    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}
