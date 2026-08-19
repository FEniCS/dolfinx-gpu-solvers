#pragma once

#include "elasticity_level.h"
#include "jacobi.h"
#include "coarse_elasticity.h"
#include "util.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <iterator>
#include <span>
#include <cmath>

#include <dolfinx/la/utils.h>
#include <basix/interpolation.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/petsc.h>

#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#if defined(__HIP_PLATFORM_AMD__)
    #include <hip/hip_runtime.h>
#else
    #include <cuda/atomic>
    #include <cuda_runtime.h>
#endif


enum class SmootherType
{
  Jacobi,
  Chebyshev
};

constexpr SmootherType smoother_type = SmootherType::Jacobi;


namespace p_transfer
{
    // compute prolongation of cells from coarse to fine mesh
    template <typename T, int coarse_dofs, int fine_dofs>

    __global__ void prolong_cells(
        std::size_t num_cells,
        const T* __restrict__ interpolation, // points to interpolation matrix from P coarse to P fine
        const std::int32_t* __restrict__ coarse_dofmap, 
        const std::int32_t* __restrict__ fine_dofmap,
        const T* __restrict__ coarse_values,
        T* __restrict__ fine_values)
    {
        // one flattened task for each thread
        // cell x local fine node x component
        const std::size_t task = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        // compute the cell index from the task index
        constexpr int value_per_cell = fine_dofs * 3; // number of values per cell
        const std::size_t total_tasks = num_cells * value_per_cell; // total number of tasks

        if (task >= total_tasks) return; // check if the thread index is valid

        // index mapping for multiple cells per blockS
        const std::size_t cell = task / value_per_cell; // find which cell this task belongs to 
        const int local_task = static_cast<int>(task % value_per_cell); // position of task within this cell
        const int component = local_task % 3; // 3 consectutive tasks correspond to x, y, z
        const int fine_i = local_task / 3; // which local fine basis function this task corresponds to

        T value = T(0); // local accumulator essentially, adds contibutions from each coarse basis function to this variable

        for (int coarse_j = 0; coarse_j < coarse_dofs; ++coarse_j){ // loop over all basis functions on the cell
            const std::int32_t coarse_node = coarse_dofmap[cell * coarse_dofs + coarse_j]; // global coarse node
            const std::int32_t coarse_dof = coarse_node * 3 + component; // convert coarse node into vector dof

            const T P_ij = interpolation[fine_i * coarse_dofs + coarse_j]; // read one interpolation matrix entry
            value += P_ij * coarse_values[coarse_dof]; // add coarse contribution to the fine value
        }
        
        const std::int32_t fine_node = fine_dofmap[cell * fine_dofs + fine_i]; // global fine node
        const std::int32_t fine_dof = fine_node * 3 + component; // convert fine node into vector dof

        #if defined(__HIP_PLATFORM_AMD__)
            fine_values[fine_dof] = value; // write the computed fine value to the output vector
            // at some point, fix HIP write race maybe with cell ownership?
        #else
            // atomic store used to remove write race condition
            cuda::atomic_ref<T, cuda::thread_scope_system> ref(fine_values[fine_dof]);
            ref.store(value, cuda::memory_order_relaxed); // write the computed fine value to the output vector
        #endif
    }


    // compute restriction of nodes from fine to coarse mesh
    template <typename T, int coarse_dofs, int fine_dofs>

    __global__ void restrict_nodes(
        std::size_t num_fine_nodes,
        const T* __restrict__ interpolation, // points to interpolation matrix from P coarse to P fine
        const std::int32_t* __restrict__ coarse_dofmap,
        const std::int32_t* __restrict__ owner_cell,
        const std::int32_t* __restrict__ owner_local,
        const std::int8_t* __restrict__ coarse_bc_marker,
        const T* __restrict__ fine_values,
        T* __restrict__ coarse_values)
    {
        // one task for each global fine node and component
        const std::size_t task = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t total_tasks = num_fine_nodes * 3; // total number of tasks

        if (task >= total_tasks) return; // check if the thread index is valid

        const std::size_t fine_node = task / 3; // which global fine node this task corresponds to
        const int component = task % 3; // component this task corresponds to i.e. x, y, or z

        const std::int32_t cell = owner_cell[fine_node]; // which cell owns this fine node
        const std::int32_t fine_i = owner_local[fine_node]; // find node's local position within the cell

        const std::int32_t fine_dof = fine_node * 3 + component; // convert fine node into vector dof
        const T fine_value = fine_values[fine_dof]; // read the fine value for this task

        for (int coarse_j = 0; coarse_j < coarse_dofs; ++coarse_j){ // loop over all basis functions on the cell
            const std::int32_t coarse_node = coarse_dofmap[cell * coarse_dofs + coarse_j]; // global coarse node
            const std::int32_t coarse_dof = coarse_node * 3 + component; // convert coarse node into vector dof

            const T P_ij = interpolation[fine_i * coarse_dofs + coarse_j]; // read one interpolation matrix entry
            
            if (!coarse_bc_marker[coarse_dof]){ // only add contribution if coarse node is not clamped
              atomicAdd(&coarse_values[coarse_dof], P_ij * fine_value); // add contribution to the coarse value
            }
        }
    }
} // namespace p_transfer


namespace p_multigrid_detail
{
  MatNullSpace build_near_nullspace(const dolfinx::fem::FunctionSpace<double>& V)
  {
    // Index map for displacement nodes
    auto map = V.dofmap()->index_map;
    const int bs = V.dofmap()->index_map_bs();

    // six rigid-body modes:
    // 3 translations + 3 rotations
    std::vector<dolfinx::la::Vector<T>> basis(6, dolfinx::la::Vector<T>(map, bs));
    
    for (auto& v : basis)
      std::ranges::fill(v.array(), T(0));

    const std::size_t num_nodes = map->size_local() + map->num_ghosts();

    // translation modes:
    // mode 0: translation in x-direction (1, 0, 0)
    // mode 1: translation in y-direction (0, 1, 0)
    // mode 2: translation in z-direction (0, 0, 1)
    for (int k = 0; k < 3; ++k)
    {
      auto& values = basis[k].array();

      for (std::int32_t i = 0; i < num_nodes; ++i)
        values[bs * i + k] = T(1);
    }

    // rotation modes:
    auto& x3 = basis[3].array(); // rotation about x-axis
    auto& x4 = basis[4].array(); // rotation about y-axis
    auto& x5 = basis[5].array(); // rotation about z-axis

    const std::vector<double> x = V.tabulate_dof_coordinates(false);
    const std::int32_t* dofs = V.dofmap()->map().data_handle();

    std::cout << "Dof coordinate entries = " << x.size() << "\n";
    std::cout << "Dofmap entries = " << V.dofmap()->map().size() << "\n";
    std::cout << "Basis vector entries = " << basis[0].array().size() << "\n";

    const std::size_t num_coordinate_blocks = x.size() / 3;
    const std::size_t num_basis_blocks = basis[0].array().size() / bs;

    for (size_t i = 0; i < V.dofmap()->map().size(); ++i)
    {
      const std::int32_t dof = dofs[i];

      if (dof < 0 || dof >= static_cast<std::int32_t>(num_coordinate_blocks)
                  || dof >= static_cast<std::int32_t>(num_basis_blocks))
        throw std::runtime_error("Dof index out of bounds");

      std::span<const double, 3> xd(x.data() + 3 * dof, 3);

      // rotation about z-axis: (-y, x, 0)
      x3[bs * dof + 0] = -xd[1];
      x3[bs * dof + 1] = xd[0];

      // rotation about y-axis: (z, 0, -x)
      x4[bs * dof + 0] = xd[2];
      x4[bs * dof + 2] = -xd[0];

      // rotation about x-axis: (0, -z, y)
      x5[bs * dof + 1] = -xd[2];
      x5[bs * dof + 2] = xd[1];
    }

    // orthonormalise the six modes
    dolfinx::la::orthonormalize(std::vector<std::reference_wrapper<dolfinx::la::Vector<T>>>(basis.begin(), basis.end()));

    if (!dolfinx::la::is_orthonormal(std::vector<std::reference_wrapper<const dolfinx::la::Vector<T>>>(basis.begin(), basis.end())))
      throw std::runtime_error("Space not orthonormal");

    // build PETSc nullspace object
    const std::int32_t length = bs * map->size_local();

    std::vector<std::span<const T>> basis_local;
    basis_local.reserve(6);

    std::transform(basis.cbegin(), basis.cend(), std::back_inserter(basis_local),
                   [length](const auto& v) {
                     return std::span<const T>(v.array().data(), length);
                   });
    
    MPI_Comm comm = V.mesh()->comm();
    
    std::vector<Vec> petsc_basis = dolfinx::la::petsc::create_vectors(comm, basis_local);
    MatNullSpace ns = dolfinx::la::petsc::create_nullspace(comm, petsc_basis);

    for (auto v : petsc_basis)
      VecDestroy(&v);

    return ns;
  }

} // namespace p_multigrid_detail


template <int P>
const ufcx_form* get_coarse_ufcx_form()
{
  if constexpr (P == 1)
    return form_coarse_elasticity_a1;
  else if constexpr (P == 2)
    return form_coarse_elasticity_a2;
  else if constexpr (P == 3)
    return form_coarse_elasticity_a3;
  else if constexpr (P == 4)
    return form_coarse_elasticity_a4;
  else if constexpr (P == 5)
    return form_coarse_elasticity_a5;
  else
    throw std::runtime_error("No coarse elasticity form available for polynomial degree " + std::to_string(P));
}


// forward declaration of p-multigrid hierarchy that lets you describe exact sequence of polynomial levels
template <int... Orders>
class PMultigridHierarchy;

// recursive template for multigrid hierarchy
template <int FineP, int NextCoarserP, int... RemainingOrders>
class PMultigridHierarchy<FineP, NextCoarserP, RemainingOrders...>{
  public:
    ElasticityLevel<FineP> level; // current polynomial level
    PMultigridHierarchy<NextCoarserP, RemainingOrders...> coarser; // hierachy beginning at the next lower polynomial level

    int pre_smooth_steps = 3; // number of Jacobi smoothing steps before restriction
    int post_smooth_steps = 3; // number of Jacobi smoothing steps after prolongation

    thrust::device_vector<T> interpolation; // interpolation matrix from level NextCoarserP to level FineP

    using DeviceIndexVector = thrust::device_vector<std::int32_t>;
    using DeviceVector = dolfinx::la::Vector<T, thrust::device_vector<T>>;  

    // information needed to perform transfers involving level P
    std::size_t num_fine_nodes = 0;
    DeviceIndexVector owner_cell;
    DeviceIndexVector owner_local;

    // Jacobi data on fineP
    DeviceVector fine_Ax;
    DeviceVector fine_residual;
    DeviceVector diagonal_inverse;

    // Chebyshev
    T lambda_max = T(0);
    DeviceVector chebyshev_previous;
    DeviceVector chebyshev_next;

    // vectors used on NextCoarserP
    DeviceVector coarse_rhs;
    DeviceVector coarse_correction;

    // prolongated correction on FineP
    DeviceVector fine_correction;

    template <typename MeshPtr, typename CellList, typename BoundaryLocator>
    PMultigridHierarchy(
      const MeshPtr& mesh_ptr,
      const CellList& cell_list,
      const BoundaryLocator& boundary_locator,
      const T lambda,
      const T mu)
      : level(mesh_ptr, cell_list, boundary_locator, lambda, mu),
        coarser(mesh_ptr, cell_list, boundary_locator, lambda, mu),
        diagonal_inverse(level.V->dofmap()->index_map, 3),
        fine_Ax(level.V->dofmap()->index_map, 3),
        fine_residual(level.V->dofmap()->index_map, 3),
        chebyshev_previous(level.V->dofmap()->index_map, 3),
        chebyshev_next(level.V->dofmap()->index_map, 3),
        coarse_rhs(coarser.level.V->dofmap()->index_map, 3),
        coarse_correction(coarser.level.V->dofmap()->index_map, 3),
        fine_correction(level.V->dofmap()->index_map, 3)
    {
        // build interpolation matrix from level CoarseP to level FineP
        auto [interpolation_host, interpolation_shape] = basix::compute_interpolation_operator(
            coarser.level.elem, level.elem
        );

        interpolation.assign(interpolation_host.begin(), interpolation_host.end()); 
        build_fine_node_owners();

        // assemble inverse of diagonal
        level.assemble_diagonal(diagonal_inverse);
        thrust::transform(
          thrust::device, diagonal_inverse.array().begin(), 
          diagonal_inverse.array().end(),
          level.bc_marker.begin(),
          diagonal_inverse.array().begin(),
          invert_jacobi_diagonal<T>()
        );

        if constexpr (smoother_type == SmootherType::Chebyshev)
        {
            lambda_max = estimate_lambda_max(level, diagonal_inverse, level.bc_marker, chebyshev_previous, chebyshev_next, fine_Ax, fine_residual, 10);
            std::cout << "Estimated lambda_max = " << lambda_max << "\n";
        }
    }


    // wrapper for the prolongation operator from level CoarseP to level FineP
    void prolong(const DeviceVector& coarse_values, DeviceVector& fine_values) const
    {
        constexpr int coarse_dofs = detail::elasticity_traits<NextCoarserP>::ndofs;
        constexpr int fine_dofs = detail::elasticity_traits<FineP>::ndofs;

        const std::size_t num_cells = level.gpu_dofmap.extent(0); // number of cells in the mesh
        const std::size_t total_tasks = num_cells * fine_dofs * 3; // number of tasks for each cell, local fine node, and component

        const int block_size = 256;
        const int grid_size = static_cast<int>((total_tasks + block_size - 1) / block_size);

        p_transfer::prolong_cells<T, coarse_dofs, fine_dofs><<<grid_size, block_size>>>(
            num_cells,
            interpolation.data().get(),
            coarser.level.gpu_dofmap.map().data().get(),
            level.gpu_dofmap.map().data().get(),
            coarse_values.array().data().get(),
            fine_values.array().data().get()
        );
    }

    
    // wrapper for the restriction operator from level FineP to level CoarseP
    void restrict(const DeviceVector& fine_values, DeviceVector& coarse_values) const
    {
        constexpr int coarse_dofs = detail::elasticity_traits<NextCoarserP>::ndofs;
        constexpr int fine_dofs = detail::elasticity_traits<FineP>::ndofs;

        thrust::fill(thrust::device, coarse_values.array().begin(), coarse_values.array().end(), T(0));
        
        const int block_size = 256;
        const std::size_t total_tasks = num_fine_nodes * 3; // number of tasks for each fine node and component
        const int grid_size = static_cast<int>((total_tasks + block_size - 1) / block_size);

        p_transfer::restrict_nodes<T, coarse_dofs, fine_dofs><<<grid_size, block_size>>>(
            num_fine_nodes,
            interpolation.data().get(),
            coarser.level.gpu_dofmap.map().data().get(),
            owner_cell.data().get(),
            owner_local.data().get(),
            coarser.level.bc_marker.data().get(),
            fine_values.array().data().get(),
            coarse_values.array().data().get()
        );
    }

    void set_smoothing_steps(int order, int pre, int post)
    {
      if (pre < 0 || post < 0)
        throw std::runtime_error("Number of smoothing steps must be non-negative");

      if (order == FineP)
      {
        pre_smooth_steps = pre;
        post_smooth_steps = post;
        return;
      }

      coarser.set_smoothing_steps(order, pre, post);
    }


    template <typename Vector>
    void v_cycle(Vector& solution, const Vector& rhs)
    {
      // pre-smoothing on level FineP
      smooth(solution, rhs, pre_smooth_steps);

      // compute residual_FineP = rhs_FineP - A_FineP * solution_FineP
      level(fine_Ax, solution);
      thrust::transform(thrust::device, rhs.array().begin(), rhs.array().end(), fine_Ax.array().begin(), fine_residual.array().begin(), thrust::minus<T>());

      // restrict residual_FineP to level CoarseP
      // apply restriction operator to residual_FineP to get rhs_coarse:
      // rhs_coarse = R * residual_FineP where R = P^T
      restrict(fine_residual, coarse_rhs);

      // set level CoarseP correction to 0
      thrust::fill(thrust::device, coarse_correction.array().begin(), coarse_correction.array().end(), T(0));

      // recursively solve the correction problem
      coarser.v_cycle(coarse_correction, coarse_rhs);

      // prolongate correction_coarse from level CoarseP to level FineP
      thrust::fill(thrust::device, fine_correction.array().begin(), fine_correction.array().end(), T(0));
      prolong(coarse_correction, fine_correction);


      // testing prolongation
      level(fine_Ax, fine_correction);
      thrust::transform(
        thrust::device,
        fine_residual.array().begin(),
        fine_residual.array().end(),
        fine_Ax.array().begin(),
        fine_residual.array().begin(),
        thrust::minus<T>()
      );

      restrict(fine_residual, coarse_rhs);

      device_synchronize();

      // ||r2_after||
      T fine_norm_sq = thrust::inner_product(
          thrust::device,
          fine_residual.array().begin(),
          fine_residual.array().end(),
          fine_residual.array().begin(),
          T(0));

      // ||R r2_after||
      T restricted_norm_sq = thrust::inner_product(
          thrust::device,
          coarse_rhs.array().begin(),
          coarse_rhs.array().end(),
          coarse_rhs.array().begin(),
          T(0));

      std::cout << "Fine residual after P1 correction = " << std::sqrt(fine_norm_sq) << '\n';

      std::cout << "Restricted residual after P1 correction = " << std::sqrt(restricted_norm_sq) << '\n';

      // add correction to solution_FineP
      thrust::transform(thrust::device, solution.array().begin(), solution.array().end(), fine_correction.array().begin(), solution.array().begin(), thrust::plus<T>());

      // post-smoothing on level FineP
      smooth(solution, rhs, post_smooth_steps);
    }

    private:
      // compute the owner cell and local index for each fine node
      void build_fine_node_owners()
      {
        const auto fine_index_map = level.V->dofmap()->index_map;

        num_fine_nodes = static_cast<std::size_t>(fine_index_map->size_local() 
            + static_cast<std::size_t>(fine_index_map->num_ghosts()));

        const auto fine_dofmap = level.V->dofmap()->map(); // flattened dofmap for level P

        std::vector<std::int32_t> owner_cell_host(num_fine_nodes, std::int32_t(-1));
        std::vector<std::int32_t> owner_local_host(num_fine_nodes, std::int32_t(-1));

        // visit every cell and fine node
        for (std::size_t cell = 0; cell < fine_dofmap.extent(0); ++cell)
        {
          for (std::size_t fine_i = 0; fine_i < fine_dofmap.extent(1); ++fine_i)
          {
            const std::int32_t fine_node = fine_dofmap(cell, fine_i);
            // if this fine node has not yet been assigned an owner cell, assign it now
            if (owner_cell_host[fine_node] == std::int32_t(-1)){
                owner_cell_host[fine_node] = static_cast<std::int32_t>(cell);
                owner_local_host[fine_node] = static_cast<std::int32_t>(fine_i);
            }
          }
        }

        // check that every fine node received an owner
        for (std::size_t fine_node = 0; fine_node < num_fine_nodes; ++fine_node){
          const std::int32_t cell = owner_cell_host[fine_node];
          const std::int32_t local = owner_local_host[fine_node];
          if (cell == std::int32_t(-1) || local == std::int32_t(-1)){
            throw std::runtime_error("Fine node " + std::to_string(fine_node) + " has no owner cell");
          }
          if (fine_dofmap(cell, local) != static_cast<std::int32_t>(fine_node)){
            throw std::runtime_error("Fine node " + std::to_string(fine_node) + " is not owned by cell " + std::to_string(cell) + " at local index " + std::to_string(local));
          }
        }

        // copy to device
        owner_cell.assign(owner_cell_host.begin(), owner_cell_host.end());
        owner_local.assign(owner_local_host.begin(), owner_local_host.end());
      }

      void smooth(
        DeviceVector& solution,
        const DeviceVector& rhs,
        int num_steps
      )
      {
        if constexpr (smoother_type == SmootherType::Jacobi)
        {
          constexpr T omega = T(0.25); // damping factor
          jacobi_smooth(level, solution, rhs, diagonal_inverse, fine_Ax, fine_residual, num_steps, omega);
        }
        else if constexpr (smoother_type == SmootherType::Chebyshev)
        {
          const T cheby_min = T(0.1) * lambda_max; // minimum eigenvalue for Chebyshev smoother
          const T cheby_max = T(1.1) * lambda_max; // maximum eigenvalue for Chebyshev smoother
          chebyshev_smooth(level, solution, rhs, diagonal_inverse, fine_Ax, fine_residual, chebyshev_previous, chebyshev_next, num_steps, cheby_min, cheby_max);
        }
      }
};

// template specialisation for the lowest order level
// stop the recursion when lowest order = current order
template <int LastCoarserP>
class PMultigridHierarchy<LastCoarserP>{
  public:
    using DeviceVector = typename ElasticityLevel<LastCoarserP>::DeviceVector;

    ElasticityLevel<LastCoarserP> level;
    
  private:

    dolfinx::fem::DirichletBC<T> coarse_bc; // homogeneous displacement boundary condition
    dolfinx::fem::Form<T> coarse_form; // dolfinx representation of assembled elasticity form

    Mat coarse_A = nullptr; // PETSc matrix for coarse solve
    Vec coarse_rhs_petsc = nullptr; // PETSc vector for coarse solve rhs
    Vec coarse_solution_petsc = nullptr; // PETSc vector for coarse solve solution
    KSP coarse_solver = nullptr; // PETSc KSP solver for coarse solve

    // physical coordinates of the coarse mesh nodes
    std::vector<T> coarse_coordinates;

  public:
    template <typename MeshPtr, typename CellList, typename BoundaryLocator>
    PMultigridHierarchy(
      const MeshPtr& mesh_ptr,
      const CellList& cell_list,
      const BoundaryLocator& boundary_locator,
      const T lambda,
      const T mu)
      : level(mesh_ptr, cell_list, boundary_locator, lambda, mu),
        coarse_bc(std::array<T, 3>{T(0), T(0), T(0)}, level.bc_nodes, level.V),
        coarse_form(dolfinx::fem::create_form<T>(*get_coarse_ufcx_form<LastCoarserP>(), {level.V, level.V}, {}, {}, {}, {}, level.V->mesh())),
        coarse_coordinates(level.V->tabulate_dof_coordinates(false))
    {
      assemble_coarse_matrix();
      attach_near_nullspace();
      setup_gamg();
      create_petsc_vectors();

    }

    ~PMultigridHierarchy()
    {
      KSPDestroy(&coarse_solver);
      MatDestroy(&coarse_A);
      VecDestroy(&coarse_rhs_petsc);
      VecDestroy(&coarse_solution_petsc);
    }

    void set_smoothing_steps(int order, int pre, int post)
    {
      if (order == LastCoarserP)
      {
        throw std::runtime_error("Cannot set smoothing steps for the coarsest level");
      }

      throw std::runtime_error("Polynomial order " + std::to_string(order) + " not found in the hierarchy");
    }
    
    template <typename Vector>
    void v_cycle(Vector& solution, const Vector& rhs)
    {
      // solve the coarsest level problem directly using PETSc GAMG

      // Finish CUDA/HIP restriction before PETSc reads rhs
       device_synchronize();

      #if defined(__HIP_PLATFORM_AMD__)
        VecHIPPlaceArray(coarse_rhs_petsc, rhs.array().data().get());
        VecHIPPlaceArray(coarse_solution_petsc, solution.array().data().get());
      #else
        VecCUDAPlaceArray(coarse_rhs_petsc, rhs.array().data().get());
        VecCUDAPlaceArray(coarse_solution_petsc, solution.array().data().get());
      #endif


      PetscReal rhs_norm;
      VecNorm(coarse_rhs_petsc, NORM_2, &rhs_norm);
      std::cout << "Coarse level rhs norm: " << rhs_norm << "\n";

      KSPSolve(coarse_solver, coarse_rhs_petsc, coarse_solution_petsc);

      PetscReal correction_norm;
      VecNorm(coarse_solution_petsc, NORM_2, &correction_norm);
      std::cout << "Coarse level correction norm: " << correction_norm << "\n";

      Vec coarse_residual;
      VecDuplicate(coarse_rhs_petsc, &coarse_residual);
      MatMult(coarse_A, coarse_solution_petsc, coarse_residual);
      VecAXPY(coarse_residual, -1.0, coarse_rhs_petsc);
      PetscReal coarse_residual_norm;
      VecNorm(coarse_residual, NORM_2, &coarse_residual_norm);
      std::cout << "P1 true residual norm: " << coarse_residual_norm << "\n";
      std::cout << "P1 relative residual norm: " << coarse_residual_norm / rhs_norm << "\n";
      VecDestroy(&coarse_residual);


      #if defined(__HIP_PLATFORM_AMD__)
        VecHIPResetArray(coarse_rhs_petsc);
        VecHIPResetArray(coarse_solution_petsc);
      #else
        VecCUDAResetArray(coarse_rhs_petsc);
        VecCUDAResetArray(coarse_solution_petsc);
      #endif
        // Ensure PETSc coarse correction is ready before prolongation
        device_synchronize();
    } 


  private:

    void create_petsc_vectors()
    {
      MPI_Comm comm = level.V->mesh()->comm();

      const auto index_map = level.V->dofmap()->index_map;

      const PetscInt bs = level.V->dofmap()->index_map_bs();
      const PetscInt local_size = bs * static_cast<PetscInt>(index_map->size_local());
      const PetscInt global_size = bs * static_cast<PetscInt>(index_map->size_global());

    #if defined(__HIP_PLATFORM_AMD__)
      VecCreateMPIHIPWithArray(
          comm,
          bs,
          local_size,
          global_size,
          nullptr,
          &coarse_rhs_petsc);

      VecCreateMPIHIPWithArray(
          comm,
          bs,
          local_size,
          global_size,
          nullptr,
          &coarse_solution_petsc);
    #else
      VecCreateMPICUDAWithArray(
          comm,
          bs,
          local_size,
          global_size,
          nullptr,
          &coarse_rhs_petsc);

      VecCreateMPICUDAWithArray(
          comm,
          bs,
          local_size,
          global_size,
          nullptr,
          &coarse_solution_petsc);
    #endif
    }


    void assemble_coarse_matrix()
    {
      // build sparsity pattern
      auto pattern = dolfinx::fem::create_sparsity_pattern(coarse_form);
      pattern.finalize();

      #if defined(__HIP_PLATFORM_AMD__)
          coarse_A = dolfinx::la::petsc::create_matrix(level.V->mesh()->comm(), pattern, "aij");
      #else
          coarse_A = dolfinx::la::petsc::create_matrix(level.V->mesh()->comm(), pattern, "aij");
      #endif

      MatZeroEntries(coarse_A);

      // assemble FE matrix
      dolfinx::fem::assemble_matrix(dolfinx::la::petsc::Matrix::set_block_fn(coarse_A, ADD_VALUES), coarse_form, {coarse_bc});

      // flush before setting BC diagonal
      MatAssemblyBegin(coarse_A, MAT_FLUSH_ASSEMBLY);
      MatAssemblyEnd(coarse_A, MAT_FLUSH_ASSEMBLY);

      // set Dirichlet diagonal entries to 1
      dolfinx::fem::set_diagonal<T>(dolfinx::la::petsc::Matrix::set_fn(coarse_A, INSERT_VALUES), *level.V, {coarse_bc});

      MatAssemblyBegin(coarse_A, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(coarse_A, MAT_FINAL_ASSEMBLY);

      // 3 displacement components per node
      MatSetBlockSize(coarse_A, level.V->dofmap()->index_map_bs());
      }


      void setup_gamg()
      {
        // set up the GAMG solver
        MPI_Comm comm = level.V->mesh()->comm();
        KSPCreate(comm, &coarse_solver);

        KSPSetOperators(coarse_solver, coarse_A, coarse_A); // matrix for coarse solve
        KSPSetOptionsPrefix(coarse_solver, "coarse_"); // set prefix for command line options
        KSPSetType(coarse_solver, KSPPREONLY); // apply GAMG as a preconditioner to CG iteration
        // KSPSetTolerances(coarse_solver, 1e-5, PETSC_DEFAULT, PETSC_DEFAULT, 10); // set tolerances for coarse solve
        KSPSetInitialGuessNonzero(coarse_solver, PETSC_FALSE);
        KSPSetNormType(coarse_solver, KSP_NORM_UNPRECONDITIONED);

        PC pc;
        KSPGetPC(coarse_solver, &pc);
        PCSetType(pc, PCGAMG);
        PCGAMGSetCoarseEqLim(pc, 1000);
        KSPSetFromOptions(coarse_solver); // apply any command line options
        KSPSetUp(coarse_solver); // actually construct the AMG hierarchy 
      }


      void attach_near_nullspace()
      {
        MatNullSpace ns = p_multigrid_detail::build_near_nullspace(*level.V);
        MatSetNearNullSpace(coarse_A, ns);
        MatNullSpaceDestroy(&ns);
      }
  };