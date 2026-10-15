#ifndef COLLISION_CUH
#define COLLISION_CUH

// CUDA INCLUDE
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "../../include/var.h"

__device__ static void inline moment_collision(dfloat rho,
                                               dfloat ux, dfloat uy, dfloat uz,
                                               dfloat *mxx, dfloat *mxy, dfloat *mxz, dfloat *myy, dfloat *myz, dfloat *mzz)
{
    // equation 90
    *mxx = T_OMEGA * (*mxx) + OMEGAd2 * ux * ux;
    *myy = T_OMEGA * (*myy) + OMEGAd2 * uy * uy;
    *mzz = T_OMEGA * (*mzz) + OMEGAd2 * uz * uz;

    *mxy = T_OMEGA * (*mxy) + OMEGA * ux * uy;
    *mxz = T_OMEGA * (*mxz) + OMEGA * ux * uz;
    *myz = T_OMEGA * (*myz) + OMEGA * uy * uz;
}

#endif // !RECONSTRUCTION_CUH
