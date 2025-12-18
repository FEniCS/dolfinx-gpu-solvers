#include "poisson.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <utility>
#include <vector>

#include <dolfinx/fem/petsc.h>
#include <petscmat.h>

#include <thrust/device_vector.h>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  dolfinx::init_logging(argc, argv);

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}},
                                  {320, 320}, mesh::CellType::triangle, part));

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::triangle, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(element)));
    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T>>(V);
    auto g = std::make_shared<fem::Function<T>>(V);

    // Define variational forms
    fem::Form<T> a = fem::create_form<T>(*form_poisson_a, {V, V}, {},
                                         {{"kappa", kappa}}, {}, {});
    fem::Form<T> L = fem::create_form<T>(*form_poisson_L, {V},
                                         {{"f", f}, {"g", g}}, {}, {}, {});

    std::vector facets = mesh::locate_entities_boundary(
        *mesh, 1,
        [](auto x)
        {
          using U = typename decltype(x)::value_type;
          constexpr U eps = 1.0e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto x0 = x(0, p);
            if (std::abs(x0) < eps or std::abs(x0 - 2) < eps)
              marker[p] = true;
          }
          return marker;
        });
    std::vector bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 1, facets);
    fem::DirichletBC<T> bc(0, bdofs, V);

    f->interpolate(
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
    g->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f.push_back(std::sin(5 * x(0, p)));
          return {f, {f.size()}};
        });

    auto u = std::make_shared<fem::Function<T>>(V);
    la::SparsityPattern sp = fem::create_sparsity_pattern(a);
    sp.finalize();

#ifdef __CUDACC__
    Mat device_mat = la::petsc::create_matrix(mesh->comm(), sp, "aijcusparse");
#endif
#ifdef __HIPCC__
    Mat device_mat = la::petsc::create_matrix(mesh->comm(), sp, "aijhipsparse");
#endif

    MatZeroEntries(device_mat);

    auto set_fn = la::petsc::Matrix::set_block_fn(device_mat, ADD_VALUES);

    la::Vector<T> b(L.function_spaces()[0]->dofmap()->index_map,
                    L.function_spaces()[0]->dofmap()->index_map_bs());

    dolfinx::common::Timer t0("Assemble matrix");
    fem::assemble_matrix(set_fn, a, {bc});
    spdlog::info("Assemble matrix...");
    MatAssemblyBegin(device_mat, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(device_mat, MAT_FLUSH_ASSEMBLY);
    auto insert = la::petsc::Matrix::set_fn(device_mat, INSERT_VALUES);
    fem::set_diagonal<T>(insert, *V, {bc});
    MatAssemblyBegin(device_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(device_mat, MAT_FINAL_ASSEMBLY);
    t0.stop();
    t0.flush();

    std::ranges::fill(b.array(), 0);
    fem::assemble_vector(b.array(), L);
    fem::apply_lifting(b.array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    bc.set(b.array(), std::nullopt);

    la::Vector<T, thrust::device_vector<T>> b_device(b);
    la::Vector<T, thrust::device_vector<T>> u_device(*(u->x()));

    // Solve here A.u = b

    auto index_map = L.function_spaces()[0]->dofmap()->index_map;
    const PetscInt local_size = index_map->size_local();
    const PetscInt global_size = index_map->size_global();
    Vec b_petsc;
    Vec u_petsc;

#ifdef __HIPCC__
    int ierr = VecCreateMPIHIPWithArray(
        mesh->comm(), PetscInt(1), local_size, global_size,
        b_device.array().data().get(), &b_petsc);
    ierr = VecCreateMPIHIPWithArray(mesh->comm(), PetscInt(1), local_size,
                                    global_size, u_device.array().data().get(),
                                    &u_petsc);
#endif

#ifdef __CUDACC__
    int ierr = VecCreateMPICUDAWithArray(
        mesh->comm(), PetscInt(1), local_size, global_size,
        b_device.array().data().get(), &b_petsc);
    ierr = VecCreateMPICUDAWithArray(mesh->comm(), PetscInt(1), local_size,
                                     global_size, u_device.array().data().get(),
                                     &u_petsc);
#endif

    spdlog::info("ierr={}", ierr);

    spdlog::info("Create Petsc KSP");
    // Create PETSc KSP object
    KSP solver;
    PC prec;
    KSPCreate(mesh->comm(), &solver);
    spdlog::info("Set KSP Type");
    KSPSetType(solver, KSPCG);
    spdlog::info("Set Operators");
    KSPSetOperators(solver, device_mat, device_mat);
    spdlog::info("Set PC Type");
    KSPGetPC(solver, &prec);
    PCSetType(prec, PCHYPRE);
    //    spdlog::info( "Set AMG Type";
    //    PCGAMGSetType(prec, PCGAMGAGG);
    KSPSetFromOptions(solver);
    spdlog::info("KSP Setup");
    KSPSetUp(solver);

    dolfinx::common::Timer tsolve("Solve");
    KSPSolve(solver, b_petsc, u_petsc);
    tsolve.stop();
    tsolve.flush();
    KSPView(solver, PETSC_VIEWER_STDOUT_WORLD);

    KSPConvergedReason reason;
    KSPGetConvergedReason(solver, &reason);

    PetscInt num_iterations = 0;
    ierr = KSPGetIterationNumber(solver, &num_iterations);
    if (ierr != 0)
      spdlog::error("KSPGetIterationNumber Error: {}", ierr);

    int rank = dolfinx::MPI::rank(mesh->comm());
    if (rank == 0)
    {
      std::cout << "Converged reason: " << reason << "\n";
      std::cout << "Num iterations: " << num_iterations << "\n";
    }

    // Copy back to host
    thrust::copy(u_device.array().begin(), u_device.array().end(),
                 u->x()->array().begin());

    std::cout << "U norm = " << dolfinx::la::norm(*(u->x()));

    // Save solution in VTK format
    io::VTKFile file(mesh->comm(), "u.pvd", "w");
    file.write<T>({*u}, 0);
  }

  dolfinx::list_timings(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}
