
#include "poisson.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <utility>
#include <vector>

#include <cudss.h>
#include <thrust/device_vector.h>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {50, 50, 50},
        mesh::CellType::tetrahedron, part));

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V =
        std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
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

    // Define boundary condition

    std::vector facets = mesh::locate_entities_boundary(*mesh, 2, [](auto x) {
      using U = typename decltype(x)::value_type;
      constexpr U eps = 1.0e-8;
      std::vector<std::int8_t> marker(x.extent(1), false);
      for (std::size_t p = 0; p < x.extent(1); ++p) {
        auto x0 = x(0, p);
        if (std::abs(x0) < eps or std::abs(x0 - 2) < eps)
          marker[p] = true;
      }
      return marker;
    });
    std::vector bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets);
    fem::DirichletBC<T> bc(0, bdofs, V);

    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>> {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p) {
            auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            f.push_back(10 * std::exp(-(dx + dy) / 0.02));
          }

          return {f, {f.size()}};
        });

    g->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>> {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f.push_back(std::sin(5 * x(0, p)));
          return {f, {f.size()}};
        });

    auto u = std::make_shared<fem::Function<T>>(V);
    la::SparsityPattern sp = fem::create_sparsity_pattern(a);
    sp.finalize();
    la::MatrixCSR<T> A(sp);
    la::Vector<T> b(L.function_spaces()[0]->dofmap()->index_map,
                    L.function_spaces()[0]->dofmap()->index_map_bs());

    fem::assemble_matrix(A.mat_add_values(), a, {bc});
    A.scatter_rev();
    fem::set_diagonal<T>(A.mat_set_values(), *V, {bc});

    // Copy matrix to GPU device
    la::MatrixCSR<T, thrust::device_vector<T>,
                  thrust::device_vector<std::int32_t>,
                  thrust::device_vector<std::int32_t>>
        A_device(A);

    std::ranges::fill(b.array(), 0);
    fem::assemble_vector(b.array(), L);
    fem::apply_lifting(b.array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    bc.set(b.array(), std::nullopt);

    // Copy vectors to GPU device
    la::Vector<T, thrust::device_vector<T>> b_device(b);
    la::Vector<T, thrust::device_vector<T>> u_device(*(u->x()));

    // Solve here A.u = b

    // cuDSS data structures and handle initialization
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t data;
    cudssMatrix_t A_dss;
    cudssMatrix_t b_dss;
    cudssMatrix_t u_dss;

    cudssCreate(&handle);
    cudssConfigCreate(&config);
    cudssDataCreate(handle, &data);
    cudssMatrixCreateCsr(
        &A_dss, A_device.index_map(0)->size_local(),
        A_device.index_map(1)->size_local(), A_device.cols().size(),
        (void *)(A_device.row_ptr().data().get()), NULL,
        (void *)(A_device.cols().data().get()),
        (void *)(A_device.values().data().get()), CUDA_R_32I, CUDA_R_64F,
        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);

    std::int64_t nrows = b.index_map()->size_local();
    cudssMatrixCreateDn(&b_dss, nrows, 1, nrows,
                        (void *)b_device.array().data().get(), CUDA_R_64F,
                        CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateDn(&u_dss, nrows, 1, nrows,
                        (void *)u_device.array().data().get(), CUDA_R_64F,
                        CUDSS_LAYOUT_COL_MAJOR);

    //    cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
    //    cudssConfigSet(config, CUDSS_REORDERING_ALGORITHM, &reorder_alg,
    //                   sizeof(cudssAlgType_t));

    dolfinx::common::Timer tsolve1("Solve CUDSS - analysis");
    //---------------------------------------------------------------------------------
    // Reordering & symbolic factorization
    cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A_dss, u_dss,
                 b_dss);
    tsolve1.stop();
    tsolve1.flush();

    dolfinx::common::Timer tsolve2("Solve CUDSS - factorisation");
    //---------------------------------------------------------------------------------
    // Numerical factorization
    cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A_dss, u_dss,
                 b_dss);
    tsolve2.stop();
    tsolve2.flush();

    //---------------------------------------------------------------------------------
    // Solving the system
    for (int i = 0; i < 100; ++i) {
      dolfinx::common::Timer tsolve3("Solve CUDSS - solve");
      cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A_dss, u_dss,
                   b_dss);
    }

    // Update ghost values before output
    // u->x()->scatter_fwd();

    //  The function `u` will be modified during the call to solve. A
    //  {cpp:class}`Function` can be saved to a file. Here, we output
    //  the solution to a `VTK` file (specified using the suffix `.pvd`)
    //  for visualisation in an external program such as Paraview.

    dolfinx::common::Timer tsolve4("Solve CUDSS - copy back to CPU");
    // Copy back to host
    thrust::copy(u_device.array().begin(), u_device.array().end(),
                 u->x()->array().begin());
    tsolve4.stop();
    tsolve4.flush();

    // Save solution in VTK format
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write<T>({*u}, 0);

    cudssConfigDestroy(config);
    cudssDataDestroy(handle, data);
    cudssMatrixDestroy(A_dss);
    cudssMatrixDestroy(u_dss);
    cudssMatrixDestroy(b_dss);
    cudssDestroy(handle);

    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}
