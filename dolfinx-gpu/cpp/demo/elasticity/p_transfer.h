#pragma once

#include <cstdint>
#include <cstddef>

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
        // atomic store used to remove write race condition
        cuda::atomic_ref<T, cuda::thread_scope_system> ref(fine_values[fine_dof]);
        ref.store(value, cuda::memory_order_relaxed); // write the computed fine value to the output vector
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
