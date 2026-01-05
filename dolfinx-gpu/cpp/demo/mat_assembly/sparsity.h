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

class GPUSparsityPattern
{
public:
  GPUSparsityPattern(
      thrust::device_vector<std::int32_t> cols,
      thrust::device_vector<std::int32_t> row_ptr,
      std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2>
          index_maps)
      : cols(cols), row_ptr(row_ptr), bs({1, 1}), index_maps(index_maps)
  {
  }

  std::pair<std::span<const std::int32_t>, std::span<const std::int32_t>>
  graph() const
  {
    return {
        std::span<const std::int32_t>(cols.data().get(), cols.size()),
        std::span<const std::int32_t>(row_ptr.data().get(), row_ptr.size())};
  }

  std::int32_t num_nonzeros() const { return cols.size(); }

  int block_size(int j) const { return bs.at(j); }

  std::shared_ptr<const dolfinx::common::IndexMap> index_map(int j) const
  {
    return index_maps.at(j);
  }

  dolfinx::common::IndexMap column_index_map() const
  {
    // FIXME
    return dolfinx::common::IndexMap(index_maps[1]->comm(), 1);
  }

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

// Copy Dofmap onto device
template <typename ContainerI>
class GPUDofMap
{
public:
  GPUDofMap(const dolfinx::fem::DofMap& dofmap)
      : _dofmap(dofmap.map().data_handle(),
                dofmap.map().data_handle() + dofmap.map().size()),
        _shape({dofmap.map().extent(0), dofmap.map().extent(1)}),
        _im(dofmap.index_map)
  {
  }

  const ContainerI& map() const { return _dofmap; }

  std::size_t extent(int j) const { return _shape.at(j); }

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
/// @return column indices and offset row pointers for CSR matrix
GPUSparsityPattern
create_sparsity(const GPUDofMap<thrust::device_vector<std::int32_t>>& dm);
