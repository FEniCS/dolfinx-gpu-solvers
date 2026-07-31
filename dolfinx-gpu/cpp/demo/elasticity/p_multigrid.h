#pragma once

#include "elasticity_level.h"
#include "jacobi.h"
#include "cg.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <basix/interpolation.h>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>

#if defined(__HIP_PLATFORM_AMD__)
    #include <hip/hip_runtime.h>
#else
    #include <cuda/atomic>
    #include <cuda_runtime.h>
#endif


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
            atomicAdd(&coarse_values[coarse_dof], P_ij * fine_value); // add contribution to the coarse value
        }
    }
} // namespace p_transfer


// forward declaration of p-multigrid hierarchy that lets you describe exact sequence of polynomial levels
template <int... Orders>
class PMultigridHierarchy;

// recursive template for multigrid hierarchy
template <int FineP, int NextCoarserP, int... RemainingOrders>
class PMultigridHierarchy<FineP, NextCoarserP, RemainingOrders...>{
  public:
    ElasticityLevel<FineP> level; // current polynomial level
    PMultigridHierarchy<NextCoarserP, RemainingOrders...> coarser; // hierachy beginning at the next lower polynomial level
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

    // vectors used on NextCoarserP
    DeviceVector coarse_rhs;
    DeviceVector coarse_correction;

    // prolongated correction on FineP
    DeviceVector fine_correction;

    template <typename MeshPtr, typename CellList, typename BoundaryLocator>
    PMultigridHierarchy(
      const MeshPtr& mesh_ptr,
      const CellList& cell_list,
      const BoundaryLocator& boundary_locator)
      : level(mesh_ptr, cell_list, boundary_locator),
        coarser(mesh_ptr, cell_list, boundary_locator),
        diagonal_inverse(level.V->dofmap()->index_map, 3),
        fine_Ax(level.V->dofmap()->index_map, 3),
        fine_residual(level.V->dofmap()->index_map, 3),
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
            fine_values.array().data().get(),
            coarse_values.array().data().get()
        );
    }


    template <typename Vector>
    void v_cycle(Vector& solution, const Vector& rhs)
    {
      constexpr int pre_smooth_steps = 3;
      constexpr int post_smooth_steps = 3;
      constexpr T omega = T(2.0 / 3.0); // damping factor

      // pre-smoothing on level FineP
      jacobi_smooth(level, solution, rhs, diagonal_inverse, fine_Ax, fine_residual, pre_smooth_steps, omega);

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

      // add correction to solution_FineP
      thrust::transform(thrust::device, solution.array().begin(), solution.array().end(), fine_correction.array().begin(), solution.array().begin(), thrust::plus<T>());

      // post-smoothing on level FineP
      jacobi_smooth(level, solution, rhs, diagonal_inverse, fine_Ax, fine_residual, post_smooth_steps, omega);
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
};

// template specialisation for the lowest order level
// stop the recursion when lowest order = current order
template <int LastCoarserP>
class PMultigridHierarchy<LastCoarserP>{
  public:
    using DeviceVector = typename ElasticityLevel<LastCoarserP>::DeviceVector;

    ElasticityLevel<LastCoarserP> level;
    DeviceVector diagonal_inverse; // inverse jacobi diagonal for coarse cg solver
    elasticity::CGSolver<DeviceVector> coarse_solver; // coarse cg solver

    template <typename MeshPtr, typename CellList, typename BoundaryLocator>
    PMultigridHierarchy(
      const MeshPtr& mesh_ptr,
      const CellList& cell_list,
      const BoundaryLocator& boundary_locator)
      : level(mesh_ptr, cell_list, boundary_locator),
        diagonal_inverse(level.V->dofmap()->index_map, 3),
        coarse_solver(level.V->dofmap()->index_map, 3)
    {
      // assemble diag(A) on coarsest level
      level.assemble_diagonal(diagonal_inverse);

      // compute the inverse of the diagonal
      thrust::transform(thrust::device, diagonal_inverse.array().begin(), diagonal_inverse.array().end(),
                        diagonal_inverse.array().begin(), thrust::placeholders::_1 = 1.0 / thrust::placeholders::_1);

      // set the coarse solver's diagonal inverse
      coarse_solver.set_diag_inverse(diagonal_inverse);

      coarse_solver.set_max_iterations(1000);
      coarse_solver.set_tolerance(1e-8);
    }

    template <typename Vector>
    void v_cycle(Vector& solution, const Vector& rhs)
    {
      // solve the coarsest level problem directly //
      // for now, could just do CG
      // PETSc PCGAMG

      coarse_solver(level, solution, rhs, true);
    }
  };