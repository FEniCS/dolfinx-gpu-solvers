

#pragma once

#ifdef __HIPCC__
#define VecCreateMPIDeviceWithArray VecCreateMPIHIPWithArray
#define aijdevicesparse "aijhipsparse"
#endif
#ifdef __CUDACC__
#define VecCreateMPIDeviceWithArray VecCreateMPICUDAWithArray
#define aijdevicesparse "aijcusparse"
#endif
