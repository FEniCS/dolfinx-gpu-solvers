#pragma once

#include <cstdint>

namespace p_transfer
{
    // compute prolongation of one cell from coarse to fine mesh
    template <typename T, int coarse_dofs, int fine_dofs>

    __global__ void prolong_one_cell(
        int cell,
        const T* __restrict__ interpolation, // points to interpolation matrix from P coarse to P fine
        const std::int32_t* __restrict__ coarse_dofmap, 
        const std::int32_t* __restrict__ fine_dofmap,
        const T* __restrict__ coarse_values,
        T* __restrict__ fine_values)
    {
        const int fine_i = threadIdx.x; // x-direction thread index chooses local fine basis function
        const int component = threadIdx.y; // y-direction thread index chooses displacement component (0,1,2)

        if (fine_i >= fine_dofs || component >= 3) return; // check if the thread index is valid

        T value = T(0); // local accumulator essentially, adds contibutions from each coarse basis function to this variable

        for (int coarse_j = 0; coarse_j < coarse_dofs; ++coarse_j){ // loop over all basis functions on the cell
            const std::int32_t coarse_node = coarse_dofmap[cell * coarse_dofs + coarse_j]; // global coarse node
            const std::int32_t coarse_dof = coarse_node * 3 + component; // convert coarse node into vector dof

            const T P_ij = interpolation[fine_i * coarse_dofs + coarse_j]; // read one interpolation matrix entry
            value += P_ij * coarse_values[coarse_dof]; // add coarse contribution to the fine value
        }
        
        const std::int32_t fine_node = fine_dofmap[cell * fine_dofs + fine_i]; // global fine node
        const std::int32_t fine_dof = fine_node * 3 + component; // convert fine node into vector dof
        fine_values[fine_dof] = value; // write the computed fine value to the output vector
    }
} // namespace p_transfer
