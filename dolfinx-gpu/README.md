# dolfinx-gpu

In this folder:

* a working C++ example, which demonstrates the use of `cuDSS` and `cuSPARSE` - only available for NVIDIA

* Python wrappers, using the same C++ headers, which can be used alongside `dolfinx` and has an interface to `cupy`

The Python wrappers will compile for both CUDA and HIP, but cuDSS (sparse solver) is only available in CUDA.