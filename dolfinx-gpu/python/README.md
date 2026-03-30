DOLFINx GPU interfaces
----------------------

Install alongside dolfinx.
Important: nanobind version must match the version used in dolfinx.
Prerequisites: CUDA, cuSPARSE, cuDSS (NVIDIA), HIP, HIPsparse (AMD)

First choose a `CUDA_ARCH`: e.g. `export CUDA_ARCH=90a`,
or for AMD, choose `HIP_ARCH` e.g. `export HIP_ARCH=gfx1100` followed by:

`pip install --no-build-isolation .`
