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
    const dfloat omegaVar = OMEGA;
    const dfloat t_omegaVar = static_cast<dfloat>(1.0) - omegaVar;
    const dfloat omegaVar_d2 = omegaVar / static_cast<dfloat>(2.0);

    // equation 90
    *mxx = (t_omegaVar * (*mxx) + omegaVar_d2 * ux * ux);
    *myy = (t_omegaVar * (*myy) + omegaVar_d2 * uy * uy);
    *mzz = (t_omegaVar * (*mzz) + omegaVar_d2 * uz * uz);

    *mxy = (t_omegaVar * (*mxy) + omegaVar * ux * uy);
    *mxz = (t_omegaVar * (*mxz) + omegaVar * ux * uz);
    *myz = (t_omegaVar * (*myz) + omegaVar * uy * uz);
}

#endif // !RECONSTRUCTION_CUH