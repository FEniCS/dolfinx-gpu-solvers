
#include <dolfinx/mesh/Mesh.h>
#include <thrust/device_vector.h>

/// Compute outward facet normals scaled by the facet Jacobian (i.e. facet
/// area in physical space) for every local facet of a tetrahedral mesh.
///
/// For each facet, the scaled normal is the cross product of the two physical
/// edge tangent vectors on that facet.  Given the cell Jacobian columns
///   g1 = v1-v0,  g2 = v2-v0,  g3 = v3-v0
/// the four outward normals follow the DOLFINx local facet ordering (facet i
/// is opposite vertex i of the reference tetrahedron):
///   facet 0 (v1,v2,v3):  n = (g2-g1) x (g3-g1)
///   facet 1 (v0,v2,v3):  n = g3 x g2
///   facet 2 (v0,v1,v3):  n = g1 x g3
///   facet 3 (v0,v1,v2):  n = g2 x g1
/// These satisfy the identity n0+n1+n2+n3 = 0 (closed surface).
///
/// For shared internal facets, the normal is computed by the lower-indexed
/// owning cell and points outward from that cell.
///
/// @param mesh  The tetrahedral mesh (geometry coordinates must be available).
/// @return      Device vector of length num_facets*3 containing the packed
///              (nx, ny, nz) scaled outward normal for each global facet index.
template <typename T>
std::tuple<thrust::device_vector<T>, thrust::device_vector<T>,
           thrust::device_vector<T>>
compute_facet_normals(dolfinx::mesh::Mesh<T>& mesh);
