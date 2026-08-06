#pragma once

#include <array>
#include <cstddef>
#include <cstdint>


namespace detail
{
    // for constant force, x y z componnts are diliberatly ignored
    template <typename T>
    struct ConstantBodyForce
    {
        static constexpr bool uses_coordinates = false;

        T force_x, force_y, force_z;

        __host__ __device__
        T operator()(int component, T, T, T) const
        {
            if (component == 0)
                return force_x;
            else if (component == 1)
                return force_y;
            else
                return force_z;
        }
    };
    
    template <typename T>
    struct ManufacturedBodyForce
    {
        static constexpr bool uses_coordinates = true;

        T lambda, mu;

        __host__ __device__
        T operator()(int component, T x, T y, T z) const
        {
            const T X = x * (T(1) - x);
            const T Y = y * (T(1) - y);
            const T Z = z * (T(1) - z);

            const T dX = T(1) - T(2) * x;
            const T dY = T(1) - T(2) * y;
            const T dZ = T(1) - T(2) * z;

            if (component == 0)
                return -(lambda + mu) * dX * Y * dZ;
            else if (component == 1)
                return -(lambda + mu) * X * dY * dZ;
            else
                return T(2) * mu * (Y * Z + X * Z) + T(2) * (lambda + T(2) * mu) * X * Y;
        }
    };


    // generic assembly kernel for force vector
    template <typename T, int nq, int ndofs, typename ForceEvaluator>
    __global__ void body_force_kernel(
        T* __restrict__ b, // output vector
        const T* __restrict__ phi_data, // basis function values at quadrature points
        const T* __restrict__ wdetJ, // quadrature weights times determinant of Jacobian
        const T* __restrict__ dof_coordinates, // physical coordinates of the dofs
        const std::int32_t* __restrict__ cell_dofs, // mapping from cell to global dofs
        const std::int32_t* __restrict__ cells, // list of cells to process
        const std::int8_t* __restrict__ bc_marker, // boundary condition marker for each dof
        const std::size_t num_cells,
        ForceEvaluator force
    )
    {
        // calculate the global thread index
        const std::size_t task = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; 
        
        // number of tasks per cell
        constexpr int values_per_cell = 3 * ndofs; // 3 values per dof (x, y, z)
        const std::size_t total_tasks = num_cells * values_per_cell;

        if (task >= total_tasks) return; // exit if out of bounds

        const std::size_t cell_idx = task / values_per_cell; // which cell
        const std::size_t cell_id = static_cast<std::size_t>(cells[cell_idx]); // get the actual cell index
        const std::size_t local_task = (task % values_per_cell); // position inside cell
        
        // reverse the flattened ordering essentially
        const int component = static_cast<int>(local_task % 3); // which component (x, y, z)
        const int local_node = local_task / 3; // local basis function

        // initialise cell contribution
        T cell_value = T(0); // i.e. local accumulator to hold the contribution for this cell, local node and component

        for (int q = 0; q < nq; ++q) // loop over quadrature points
        {
            const T phi_i = phi_data[q * ndofs + local_node]; // basis function value at quadrature point q for local node
            const T wj = wdetJ[cell_id * nq + q]; // weight times determinant of Jacobian for this cell and quadrature point
            
            T xq = T(0);
            T yq = T(0);
            T zq = T(0); 

            // interpolate node coordinates to quadrature point
            for (int i = 0; i < ndofs; ++i){
                const std::int32_t global_node = cell_dofs[cell_id * ndofs + i]; // get global node index
                const T phi = phi_data[q * ndofs + i]; // basis function value for this node at quadrature point q

                xq += phi * dof_coordinates[3 * global_node + 0]; // x coordinate
                yq += phi * dof_coordinates[3 * global_node + 1]; // y coordinate
                zq += phi * dof_coordinates[3 * global_node + 2]; // z coordinate
            }
            const T force_value = force(component, xq, yq, zq); // evaluate the force at this quadrature point
            
            cell_value += phi_i * wj * force_value; // accumulate contribution
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
template <int P, typename DeviceVector, typename ScalarContainer, typename IndexContainer, typename BCMarkerContainer, typename ForceEvaluator>
void launch_body_force_kernel(
    DeviceVector& b, // output vector
    const ScalarContainer& phi_data, // basis function values at quadrature points
    const ScalarContainer& wdetJ, // quadrature weights times determinant of Jacobian
    const ScalarContainer& dof_coordinates, // physical coordinates of the dofs
    const IndexContainer& cell_dofs, // mapping from cell to global dofs
    const IndexContainer& cells, // list of cells to process
    const BCMarkerContainer& bc_marker, // boundary condition marker for each dof
    ForceEvaluator force // functor to evaluate the force at a given point
)
{
    using T = typename DeviceVector::value_type; // deduce the scalar type from the device vector
    
    constexpr int nq = detail::elasticity_traits<P>::nq; // number of quadrature points for P-th order elements
    constexpr int ndofs = detail::elasticity_traits<P>::ndofs; // number of local degrees of freedom

    thrust::fill(thrust::device, b.array().begin(), b.array().end(), T(0)); // initialise output vector to zero

    const std::size_t total_tasks = cells.size() * 3 * ndofs; // total number of tasks

    constexpr int block_size = 256; // number of threads per block
    const int grid_size = (total_tasks + block_size - 1) / block_size; // number of blocks needed

    detail::body_force_kernel<T, nq, ndofs, ForceEvaluator><<<grid_size, block_size>>>(
        b.array().data().get(), 
        phi_data.data().get(), 
        wdetJ.data().get(), 
        dof_coordinates.data().get(), 
        cell_dofs.data().get(), 
        cells.data().get(), 
        bc_marker.data().get(), 
        cells.size(),
        force
    );
}