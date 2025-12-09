import numpy as np
import numpy.typing as npt

from dolfinx.cpp.common import IndexMap
from dolfinx_gpu import gpucpp as _cpp


def _map_type(name: str, dtype: npt.DTypeLike = np.float64):
    if np.issubdtype(dtype, np.float32):
        typename = "float32"
    elif np.issubdtype(dtype, np.float64):
        typename = "float64"
    elif np.issubdtype(dtype, np.complex64):
        typename = "complex64"
    elif np.issubdtype(dtype, np.complex128):
        typename = "complex128"
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")
    vtype = getattr(_cpp, f"GPU_{name}_{typename}")
    return vtype


class GPUVector:
    def __init__(self, x):
        self._cpp_object = _map_type("Vector", x._cpp_object.dtype)(x._cpp_object)

    @property
    def index_map(self) -> IndexMap:
        """Index map that describes size and parallel distribution."""
        return self._cpp_object.index_map

    @property
    def block_size(self) -> int:
        """Block size for the vector."""
        return self._cpp_object.bs

    @property
    def array(self):
        """Local representation of the vector."""
        return self._cpp_object.array


class GPUMatrixCSR:
    def __init__(self, A):
        self._cpp_object = _map_type("MatrixCSR", A._cpp_object.dtype)(A._cpp_object)

    @property
    def index_map(self) -> IndexMap:
        """Index map that describes size and parallel distribution."""
        return self._cpp_object.index_map

    @property
    def block_size(self) -> int:
        """Block size."""
        return self._cpp_object.bs

    @property
    def data(self):
        """Values of matrix entries."""
        return self._cpp_object.data

    @property
    def indices(self):
        """Column indices."""
        return self._cpp_object.indices

    @property
    def indptr(self):
        """Row pointers."""
        return self._cpp_object.indptr


class GPUSolver:
    def __init__(self, A, b, u):
        self._cpp_object = _map_type("Solver", A._cpp_object.dtype)(A._cpp_object, b._cpp_object, u._cpp_object)

    def analyze(self):
        self._cpp_object.analyze()

    def factorize(self):
        self._cpp_object.factorize()

    def solve(self):
        self._cpp_object.solve()


class GPUSPMV:
    def __init__(self, A, b, u):
        self._cpp_object = _map_type("SPMV", A._cpp_object.dtype)(A._cpp_object, b._cpp_object, u._cpp_object)

    def apply(self):
        self._cpp_object.apply()
