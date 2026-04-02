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

#include <ginkgo/core/distributed/partition.hpp>

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
      "solver", po::value<std::string>()->default_value("AMG"),
      "Solver type (AMG, LU or BJ)")(
      "n", po::value<std::int32_t>()->default_value(20), "Mesh size");

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
  std::string solver_type = vm["solver"].as<std::string>();
  std::int32_t n = vm["n"].as<std::int32_t>();

  std::cout << "Mesh = " << n << " cube.\n";
  std::cout << "Solver = " << solver_type << "\n";

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {n, n, n},
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

    la::SparsityPattern sp = fem::create_sparsity_pattern(a);
    sp.finalize();
    la::MatrixCSR<T> A(sp);
    la::Vector<T> b(L.function_spaces()[0]->dofmap()->index_map,
                    L.function_spaces()[0]->dofmap()->index_map_bs());

    fem::assemble_matrix(A.mat_add_values(), a, {bc});
    A.scatter_rev();
    fem::set_diagonal<T>(A.mat_set_values(), *V, {bc});

    std::ranges::fill(b.array(), 0);
    fem::assemble_vector(b.array(), L);
    fem::apply_lifting(b.array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    bc.set(b.array(), std::nullopt);
    std::cout << "b.norm = " << dolfinx::la::norm(b) << "\n";

    // Solve here A.u = b
    const auto comm = gko::experimental::mpi::communicator(mesh->comm());
    int rank = comm.rank();
    int nranks = comm.size();
    std::cout << "Rank = " << rank << "/" << nranks << "\n";

#if defined(USE_HIP)
    auto executor = gko::HipExecutor::create(
        rank % gko::HipExecutor::get_num_devices(), gko::OmpExecutor::create());
#elif defined(USE_CUDA)
    auto executor
        = gko::CudaExecutor::create(rank % gko::CudaExecutor::get_num_devices(),
                                    gko::OmpExecutor::create());
#endif

    namespace dist = gko::experimental::distributed;

    auto index_map = b.index_map();
    std::int64_t local_size = index_map->size_local();
    auto [global_start, global_end] = index_map->local_range();

    auto partition = gko::share(
        dist::build_partition_from_local_size<std::int32_t, std::int64_t>(
            executor, comm, local_size));

    gko::size_type nrglobal = index_map->size_global();
    gko::matrix_data<T, std::int64_t> local_data;
    local_data.size = {nrglobal, nrglobal};

    // Convert all column indices to global
    const auto& cols_local = A.cols();
    const auto& row_ptr = A.row_ptr();
    const auto& vals = A.values();

    std::vector<std::int64_t> cols_global(cols_local.size());
    A.index_map(1)->local_to_global(cols_local, cols_global);

    for (std::int64_t i = 0; i < local_size; ++i)
    {
      for (auto j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
      {
        local_data.nonzeros.emplace_back(global_start + i, cols_global[j],
                                         vals[j]);
      }
    }
    local_data.sort_row_major();

    auto mat = gko::share(
        dist::Matrix<T, std::int32_t, std::int64_t>::create(executor, comm));
    mat->read_distributed(local_data, partition);

    std::cout << "Distributed matrix: " << mat->get_size()[0] << " x "
              << mat->get_size()[1] << "\n";

    auto b_gko = dist::Vector<T>::create(
        executor, comm, gko::dim<2>(index_map->size_global(), 1),
        gko::dim<2>(index_map->size_local(), 1));

    auto u_gko = dist::Vector<T>::create(
        executor, comm, gko::dim<2>(index_map->size_global(), 1),
        gko::dim<2>(index_map->size_local(), 1));

    u_gko->fill(0.0);

    thrust::copy(b.array().begin(), b.array().begin() + local_size,
                 b_gko->get_local_values());

    std::cout << "u.norm [0] = " << dolfinx::la::norm(*u->x()) << "\n";

    dolfinx::common::Timer tsolve1("[Set up Ginkgo]");
    std::unique_ptr<gko::LinOp> solver;

    if (solver_type == "LU")
    {
      auto solver_factory
          = gko::experimental::solver::Direct<double, int>::build()
                .with_factorization(
                    gko::experimental::factorization::Lu<double, int>::build()
                        .on(executor))
                .on(executor);

      solver = solver_factory->generate(gko::share(std::move(mat)));
    }
    else if (solver_type == "AMG")
    {
      using cg = gko::solver::Cg<T>;
      using mg = gko::solver::Multigrid;
      using pgm = gko::multigrid::Pgm<T, std::int32_t>;
      using ir = gko::solver::Ir<T>;
      using bj = gko::preconditioner::Jacobi<T, std::int32_t>;
      using schwarz
          = dist::preconditioner::Schwarz<T, std::int32_t, std::int64_t>;

      auto smoother = gko::share(
          ir::build()
              .with_solver(bj::build().with_max_block_size(1u).on(executor))
              .with_criteria(
                  gko::stop::Iteration::build().with_max_iters(1u).on(executor))
              .on(executor));

      auto amg_factory = gko::share(
          mg::build()
              .with_max_levels(10u)
              .with_min_coarse_rows(32u)
              .with_pre_smoother(smoother)
              .with_post_smoother(smoother)
              .with_mg_level(gko::share(
                  pgm::build().with_deterministic(true).on(executor)))
              .with_criteria(
                  gko::stop::Iteration::build().with_max_iters(1u).on(executor))
              .on(executor));

      solver = cg::build()
                   .with_criteria(
                       gko::stop::Iteration::build().with_max_iters(1000u).on(
                           executor),
                       gko::stop::ResidualNorm<T>::build()
                           .with_reduction_factor(T{1e-7})
                           .on(executor))
                   .with_preconditioner(schwarz::build()
                                            .with_local_solver(amg_factory)
                                            .on(executor))
                   .on(executor)
                   ->generate(gko::share(std::move(mat)));
    }
    else if (solver_type == "BJ")
    {
      using cg = gko::solver::Cg<T>;
      using bj = gko::preconditioner::Jacobi<T, std::int32_t>;
      using schwarz
          = dist::preconditioner::Schwarz<T, std::int32_t, std::int64_t>;

      solver = cg::build()
                   .with_criteria(
                       gko::stop::Iteration::build().with_max_iters(1000u).on(
                           executor),
                       gko::stop::ResidualNorm<T>::build()
                           .with_reduction_factor(T{1e-7})
                           .on(executor))
                   .with_preconditioner(
                       schwarz::build()
                           .with_local_solver(
                               bj::build().with_max_block_size(1u).on(executor))
                           .on(executor))
                   .on(executor)
                   ->generate(mat);
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
        thrust::copy(u_gko->get_local_values(),
                     u_gko->get_local_values() + local_size,
                     u->x()->array().begin());
        std::cout << "u.norm [after] = " << dolfinx::la::norm(*u->x()) << "\n";

        u->x()->scatter_fwd();

        std::ranges::fill(b.array(), 0);
        fem::assemble_vector(b.array(), L);
        fem::apply_lifting(b.array(), {a}, {{bc}}, {}, T(1));
        b.scatter_rev(std::plus<T>());
        bc.set(b.array(), std::nullopt);
        std::cout << "b.norm = " << dolfinx::la::norm(b) << "\n";

        // Copy RHS back to device
        thrust::copy(b.array().begin(), b.array().end(),
                     b_gko->get_local_values());

        file.write_function<T>(*u, static_cast<double>(i));
      }
    }

    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}
