# Solver for the heat equation using cuDSS from NVIDIA

This code solves the time-dependent heat equation on a 100x100x100
cube, using NVIDIA's cuDSS sparse direct solver, and cuSPARSE to apply
a mass matrix.

## Prerequisites

* NVIDIA CUDA 12.9+
* NVIDIA cuSPARSE
* NVIDIA cuDSS library
* dolfinx
* ffcx

## Building

Create a `build` directory, then:

```
cd build
cmake ..
make
```
