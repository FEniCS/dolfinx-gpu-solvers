# Demo using PETSc

In this demo, a PETSc matrix is assembled on the CPU, and copied to the GPU
and solved using hypre.

## Build

* Set either `HIP_ARCH` or `CUDA_ARCH` and use `cmake`.
* PETSc and hypre need to be configured with cuda or rocm support.

### Note: currently not working on AMD gfx1100 architecture
