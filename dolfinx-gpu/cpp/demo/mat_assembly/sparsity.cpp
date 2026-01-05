
#include "sparsity.h"
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <vector>

namespace
{
void __global__ insert_pattern(std::int32_t* row_keys, std::int32_t* col_vals,
                               const std::int32_t* cell_dofs)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ndofs = blockDim.x;
  int cell = blockIdx.x;

  int in_offset = cell * ndofs;
  int out_offset = cell * ndofs * ndofs;

  row_keys[out_offset + tx * ndofs + ty] = cell_dofs[in_offset + tx];
  col_vals[out_offset + tx * ndofs + ty] = cell_dofs[in_offset + ty];
}
} // namespace

/// @brief Create a sparsity pattern from a DofMap
/// @param dm DofMap
/// @return column indices and offset row pointers for CSR matrix
GPUSparsityPattern
create_sparsity(const GPUDofMap<thrust::device_vector<std::int32_t>>& dm)
{
  int ncells = dm.extent(0);
  int ndofs = dm.extent(1);
  const thrust::device_vector<std::int32_t>& cell_dofs_dev = dm.map();

  // Fill with entries from cell_dofs
  thrust::device_vector<std::int32_t> row_keys(ndofs * cell_dofs_dev.size());
  thrust::device_vector<std::int32_t> col_vals(ndofs * cell_dofs_dev.size());
  dim3 grid_size(ncells);
  dim3 block_size(ndofs, ndofs);
  insert_pattern<<<grid_size, block_size>>>(
      row_keys.data().get(), col_vals.data().get(), cell_dofs_dev.data().get());

  // Sort entries into order
  auto zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(row_keys.begin(), col_vals.begin()));
  auto zip_end = thrust::make_zip_iterator(
      thrust::make_tuple(row_keys.end(), col_vals.end()));
  thrust::sort(zip_begin, zip_end);

  // Remove duplicates
  thrust::device_vector<std::int32_t> reduced_rows(row_keys.size());
  thrust::device_vector<std::int32_t> reduced_cols(col_vals.size());
  thrust::device_vector<std::int32_t> counts(row_keys.size(), 1);
  auto reduce_zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(reduced_rows.begin(), reduced_cols.begin()));
  // Just use thrust::reduce_by_key as "unique" (ignores counts)
  auto new_end = thrust::reduce_by_key(zip_begin, zip_end, counts.begin(),
                                       reduce_zip_begin, counts.begin());
  auto row_end = thrust::get<0>(new_end.first.get_iterator_tuple());
  reduced_rows.erase(row_end, reduced_rows.end());
  auto col_end = thrust::get<1>(new_end.first.get_iterator_tuple());
  reduced_cols.erase(col_end, reduced_cols.end());

  // Count entries per row and create offset array
  thrust::device_vector<std::int32_t> row_ptr(reduced_rows.size());
  thrust::device_vector<std::int32_t> dummy(reduced_rows.size());
  thrust::fill(counts.begin(), counts.end(), 1);
  // Use thrust::reduce_by_key to count (ignores unique output)
  auto rp_end
      = thrust::reduce_by_key(reduced_rows.begin(), reduced_rows.end(),
                              counts.begin(), dummy.begin(), row_ptr.begin());
  row_ptr.erase(rp_end.second, row_ptr.end());
  row_ptr.push_back(0);
  // Create offsets from counts
  thrust::exclusive_scan(row_ptr.begin(), row_ptr.end(), row_ptr.begin());

  std::shared_ptr<const dolfinx::common::IndexMap> im0 = dm.index_map();
  std::shared_ptr<const dolfinx::common::IndexMap> im1 = dm.index_map();
  return GPUSparsityPattern(reduced_cols, row_ptr, {im0, im1});
}
