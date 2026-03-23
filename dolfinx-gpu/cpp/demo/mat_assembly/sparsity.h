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

/// On-device SparsityPattern
class GPUSparsityPattern
{
public:
  /// @brief Construct a sparsity pattern
  /// @param cols Flattened list of column indices
  /// @param row_ptr Pointers to the start of each row in `cols`
  /// @param index_maps Row and column index maps
  GPUSparsityPattern(
      thrust::device_vector<std::int32_t> cols,
      thrust::device_vector<std::int32_t> row_ptr,
      std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2>
          index_maps)
      : cols(cols), row_ptr(row_ptr), bs({1, 1}), index_maps(index_maps)
  {
  }

  /// @brief Get the columns for each row
  /// @returns column indices and row offsets as two lists
  std::pair<std::span<const std::int32_t>, std::span<const std::int32_t>>
  graph() const
  {
    return {
        std::span<const std::int32_t>(cols.data().get(), cols.size()),
        std::span<const std::int32_t>(row_ptr.data().get(), row_ptr.size())};
  }

  /// @brief Number of non-zeros in sparsity pattern
  std::int32_t num_nonzeros() const { return cols.size(); }

  /// @brief Block size for each axis
  /// @param j Axis
  int block_size(int j) const { return bs.at(j); }

  /// @brief IndexMap for each axis
  /// @param j Axis
  std::shared_ptr<const dolfinx::common::IndexMap> index_map(int j) const
  {
    return index_maps.at(j);
  }

  /// @brief Column IndexMap post-finalization
  /// @note This is not yet implemented (needed for parallel)
  dolfinx::common::IndexMap column_index_map() const
  {
    // FIXME
    return dolfinx::common::IndexMap(index_maps[1]->comm(), 1);
  }

  /// @brief Off-diagonal offsets
  /// @note This is not yet implemented (needed for parallel)
  std::span<const std::int32_t> off_diagonal_offsets() const
  {
    // FIXME
    return _off_diagonal_offsets;
  }

private:
  thrust::device_vector<std::int32_t> cols;
  thrust::device_vector<std::int32_t> row_ptr;
  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> index_maps;
  std::array<int, 2> bs;
  std::vector<std::int32_t> _off_diagonal_offsets;
};

// Copy Dofmap onto device
template <typename ContainerI>
class GPUDofMap
{
public:
  /// @brief Construct on-device from existing DofMap
  GPUDofMap(const dolfinx::fem::DofMap& dofmap)
      : _dofmap(dofmap.map().data_handle(),
                dofmap.map().data_handle() + dofmap.map().size()),
        _shape({dofmap.map().extent(0), dofmap.map().extent(1)}),
        _im(dofmap.index_map)
  {
  }

  /// @brief The DofMap as a flattened list
  const ContainerI& map() const { return _dofmap; }

  /// @brief The size of the DofMap in each axis
  /// @param j Axis
  std::size_t extent(int j) const { return _shape.at(j); }

  /// @brief IndexMap for the DofMap
  std::shared_ptr<const dolfinx::common::IndexMap> index_map() const
  {
    return _im;
  }

private:
  ContainerI _dofmap;
  std::array<std::size_t, 2> _shape;
  std::shared_ptr<const dolfinx::common::IndexMap> _im;
};

/// @brief Create a sparsity pattern from a dofmap
/// @param dm DofMap
/// @return GPUSparsityPattern
GPUSparsityPattern
create_sparsity(const GPUDofMap<thrust::device_vector<std::int32_t>>& dm);
