// Copyright (C) 2025 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <memory>
#include <span>
#include <thrust/device_vector.h>

/// @brief On-device CSR sparsity pattern for a sparse matrix.
///
/// Stores the column indices and row pointers of a CSR matrix on the
/// GPU device, along with the row and column index maps that describe
/// the parallel distribution of the matrix. This class is designed to
/// be compatible with `dolfinx::la::MatrixCSR`, providing the same
/// interface for construction.
///
/// Typical usage is to construct a `GPUDofMap` from a
/// `dolfinx::fem::DofMap`, then call `create_sparsity` to build the
/// pattern, and finally pass the pattern to `dolfinx::la::MatrixCSR`
/// to allocate a device-resident sparse matrix.
class GPUSparsityPattern
{
public:
  /// @brief Construct a sparsity pattern from on-device CSR arrays.
  ///
  /// @param[in] cols Column indices of the non-zero entries, stored on
  /// device. Length is equal to the number of non-zeros.
  /// @param[in] row_ptr Row pointer array in CSR format, stored on
  /// device. Length is `num_rows + 1`, where `row_ptr[i]` is the index
  /// into `cols` of the first non-zero in row `i`.
  /// @param[in] index_maps Row (index 0) and column (index 1) index
  /// maps describing the parallel distribution of the matrix.
  GPUSparsityPattern(
      thrust::device_vector<std::int32_t> cols,
      thrust::device_vector<std::int32_t> row_ptr,
      std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2>
          index_maps)
      : cols(cols), row_ptr(row_ptr), bs({1, 1}), index_maps(index_maps)
  {
  }

  /// @brief Return the column indices and row pointers as spans.
  ///
  /// Returns raw host-accessible spans into the device data. Note that
  /// dereferencing these spans from host code requires unified memory or
  /// an explicit device-to-host copy.
  ///
  /// @return A pair of spans: (column indices, row pointers).
  std::pair<std::span<const std::int32_t>, std::span<const std::int32_t>>
  graph() const
  {
    return {
        std::span<const std::int32_t>(cols.data().get(), cols.size()),
        std::span<const std::int32_t>(row_ptr.data().get(), row_ptr.size())};
  }

  /// @brief Return the number of non-zero entries.
  /// @return Number of non-zeros.
  std::int32_t num_nonzeros() const { return cols.size(); }

  /// @brief Return the block size for a given dimension.
  ///
  /// Currently always returns 1 (scalar block size).
  ///
  /// @param[in] j Dimension index (0 for rows, 1 for columns).
  /// @return Block size in dimension `j`.
  int block_size(int j) const { return bs.at(j); }

  /// @brief Return the index map for a given dimension.
  ///
  /// @param[in] j Dimension index (0 for rows, 1 for columns).
  /// @return Index map for dimension `j`.
  std::shared_ptr<const dolfinx::common::IndexMap> index_map(int j) const
  {
    return index_maps.at(j);
  }

  /// @brief Return offsets into the column index array marking the
  /// boundary between diagonal and off-diagonal blocks.
  ///
  /// @note Currently unimplemented; always returns an empty span.
  /// @return Off-diagonal offsets.
  std::span<const std::int32_t> off_diagonal_offsets() const
  {
    return _off_diagonal_offsets;
  }

private:
  thrust::device_vector<std::int32_t> cols;
  thrust::device_vector<std::int32_t> row_ptr;
  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> index_maps;
  std::array<int, 2> bs;
  std::vector<std::int32_t> _off_diagonal_offsets;
};

/// @brief On-device copy of a DOLFINx degree-of-freedom map.
///
/// Copies the dofmap data from a `dolfinx::fem::DofMap` onto the GPU
/// device using the provided container type (typically
/// `thrust::device_vector<std::int32_t>`). The 2D shape and associated
/// index map are retained for use in sparsity pattern construction and
/// assembly.
///
/// @tparam ContainerI Device container type for integer data, e.g.
/// `thrust::device_vector<std::int32_t>`.
template <typename ContainerI>
class GPUDofMap
{
public:
  /// @brief Construct a device dofmap from a host DofMap.
  ///
  /// Copies the flattened dofmap array to the device. The shape and
  /// index map are stored but not copied to the device.
  ///
  /// @param[in] dofmap The host-side degree-of-freedom map to copy.
  GPUDofMap(const dolfinx::fem::DofMap& dofmap)
      : _dofmap(dofmap.map().data_handle(),
                dofmap.map().data_handle() + dofmap.map().size()),
        _shape({dofmap.map().extent(0), dofmap.map().extent(1)}),
        _im(dofmap.index_map)
  {
  }

  /// @brief Return the on-device dofmap data array.
  ///
  /// The array is stored in row-major order with shape
  /// `(num_cells, num_dofs_per_cell)`.
  ///
  /// @return Reference to the device container holding dof indices.
  const ContainerI& map() const { return _dofmap; }

  /// @brief Return the size of the dofmap in a given dimension.
  ///
  /// @param[in] j Dimension index (0 for number of cells, 1 for number
  /// of dofs per cell).
  /// @return Size in dimension `j`.
  std::size_t extent(int j) const { return _shape.at(j); }

  /// @brief Return the index map associated with the dofmap.
  /// @return Index map for the owned and ghost degrees-of-freedom.
  std::shared_ptr<const dolfinx::common::IndexMap> index_map() const
  {
    return _im;
  }

private:
  ContainerI _dofmap;
  std::array<std::size_t, 2> _shape;
  std::shared_ptr<const dolfinx::common::IndexMap> _im;
};

/// @brief Create a CSR sparsity pattern on the device from a dofmap.
///
/// Computes the sparsity pattern of the matrix that would result from
/// assembling a finite element form over the mesh cells described by
/// `dm`. The column indices and row pointers are computed on the host
/// and then copied to the device.
///
/// @param[in] dm On-device dofmap (must use
/// `thrust::device_vector<std::int32_t>`).
/// @return A `GPUSparsityPattern` containing the on-device CSR arrays
/// and index maps for the assembled matrix.
GPUSparsityPattern
create_sparsity(const GPUDofMap<thrust::device_vector<std::int32_t>>& dm);
