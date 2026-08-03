#pragma once

#include <array>
#include <cstddef>
#include <cstdint>


namespace detail
{
    template <typename T, int nq, int ndofs>
    __global__ void body_force_kernel(
        T* __restrict__ b, // output vector
        const T* __restrict__ phi_data, // basis function values at quadrature points
        const T* __restrict__ wdetJ, // quadrature weights times determinant of Jacobian
        const std::int32_t* __restrict__ cell_dofs, // mapping from cell to global dofs
        const std::int32_t* __restrict__ cells, // list of cells to process
        const std::int8_t* __restrict__ bc_marker, // boundary condition marker for each dof
        const std::size_t num_cells,
        T force_x, T force_y, T force_z
    )
    {
        // calculate the global thread index
        const std::size_t task = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; 
        
        // number of tasks per cell
        constexpr int values_per_cell = 3 * ndofs; // 3 values per dof (x, y, z)
        const std::size_t total_tasks = num_cells * values_per_cell;

        if (task >= total_tasks) return; // exit if out of bounds

        const std::size_t cell_idx = task / values_per_cell; // which cell
        const std::int32_t cell_id = static_cast<std::size_t>(cells[cell_idx]); // get the actual cell index
        const std::size_t local_task = (task % values_per_cell); // position inside cell
        
        // reverse the flattened ordering essentially
        const std::size_t component = task % 3; // which component (x, y, z)
        const int local_node = local_task / 3; // local basis function

        // select relevant force component
        T force;
        if (component == 0)
            force = force_x;
        else if (component == 1)
            force = force_y;
        else
            force = force_z;

        // initialise cell contribution
        T cell_value = T(0); // i.e. local accumulator to hold the contribution for this cell, local node and component

        for (int q = 0; q < nq; ++q) // loop over quadrature points
        {
            const T phi_i = phi_data[q * ndofs + local_node]; // basis function value at quadrature point q for local node
            const T wj = wdetJ[cell_id * nq + q]; // weight times determinant of Jacobian for this cell and quadrature point
            cell_value += phi_i * wj * force; // accumulate contribution
        }

        const std::int32_t global_node = cell_dofs[cell_id * ndofs + local_node]; // get global dof index
        const std::size_t global_dof = 3 * global_node + component; // global index in the output 
        
        // apply boundary condition and assemble
        if (!bc_marker[global_dof]) // if not a boundary condition node
        {
            atomicAdd(&b[global_dof], cell_value); // atomic add to global vector
        }
    }
} // namespace detail


// wrapper to launch kernel
template <int P, typename T, typename DeviceVector, typename ScalarContainer, typename IndexContainer, typename BCMarkerContainer>
void launch_body_force_kernel(
    DeviceVector& b, // output vector
    const ScalarContainer& phi_data, // basis function values at quadrature points
    const ScalarContainer& wdetJ, // quadrature weights times determinant of Jacobian
    const IndexContainer& cell_dofs, // mapping from cell to global dofs
    const IndexContainer& cells, // list of cells to process
    const BCMarkerContainer& bc_marker, // boundary condition marker for each dof
    T force_x, T force_y, T force_z
)
{
    constexpr int nq = detail::elasticity_traits<P>::nq; // number of quadrature points for P-th order elements
    constexpr int ndofs = detail::elasticity_traits<P>::ndofs; // number of local degrees of freedom

    thrust::fill(thrust::device, b.array().begin(), b.array().end(), T(0)); // initialise output vector to zero

    const std::size_t total_tasks = cells.size() * 3 * ndofs; // total number of tasks

    constexpr int block_size = 256; // number of threads per block
    const int grid_size = (total_tasks + block_size - 1) / block_size; // number of blocks needed

    detail::body_force_kernel<T, nq, ndofs><<<grid_size, block_size>>>(
        b.array().data().get(), 
        phi_data.data().get(), 
        wdetJ.data().get(), 
        cell_dofs.data().get(), 
        cells.data().get(), 
        bc_marker.data().get(), 
        cells.size(),
        force_x, force_y, force_z
    );
}