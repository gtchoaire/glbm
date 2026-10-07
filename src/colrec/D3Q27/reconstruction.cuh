#ifndef RECONSTRUCTION_CUH
#define RECONSTRUCTION_CUH

// CUDA INCLUDE
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "../../../include/var.h"
#include "properties.cuh"

__device__ inline void pop_reconstruction(dfloat rho,
                                          dfloat ux, dfloat uy, dfloat uz,
                                          dfloat mxx, dfloat mxy, dfloat mxz, dfloat myy, dfloat myz, dfloat mzz,
                                          dfloat *pop)
{
    dfloat multiplyTerm = rho * W0;
    const dfloat pics2 = static_cast<dfloat>(1.0) - cs2 * (mxx + myy + mzz);

    pop[0] = multiplyTerm * (pics2);

    multiplyTerm = rho * W1;
    pop[1] = multiplyTerm * (pics2 + ux + mxx);
    pop[2] = multiplyTerm * (pics2 - ux + mxx);
    pop[3] = multiplyTerm * (pics2 + uy + myy);
    pop[4] = multiplyTerm * (pics2 - uy + myy);
    pop[5] = multiplyTerm * (pics2 + uz + mzz);
    pop[6] = multiplyTerm * (pics2 - uz + mzz);

    multiplyTerm = rho * W2;
    pop[7] = multiplyTerm * (pics2 + ux + uy + mxx + myy + mxy);
    pop[8] = multiplyTerm * (pics2 - ux - uy + mxx + myy + mxy);
    pop[9] = multiplyTerm * (pics2 + ux + uz + mxx + mzz + mxz);
    pop[10] = multiplyTerm * (pics2 - ux - uz + mxx + mzz + mxz);
    pop[11] = multiplyTerm * (pics2 + uy + uz + myy + mzz + myz);
    pop[12] = multiplyTerm * (pics2 - uy - uz + myy + mzz + myz);
    pop[13] = multiplyTerm * (pics2 + ux - uy + mxx + myy - mxy);
    pop[14] = multiplyTerm * (pics2 - ux + uy + mxx + myy - mxy);
    pop[15] = multiplyTerm * (pics2 + ux - uz + mxx + mzz - mxz);
    pop[16] = multiplyTerm * (pics2 - ux + uz + mxx + mzz - mxz);
    pop[17] = multiplyTerm * (pics2 + uy - uz + myy + mzz - myz);
    pop[18] = multiplyTerm * (pics2 - uy + uz + myy + mzz - myz);

    multiplyTerm = rho * W3;
    pop[19] = multiplyTerm * (pics2 + ux + uy + uz + mxx + myy + mzz + (mxy + mxz + myz));
    pop[20] = multiplyTerm * (pics2 - ux - uy - uz + mxx + myy + mzz + (mxy + mxz + myz));
    pop[21] = multiplyTerm * (pics2 + ux + uy - uz + mxx + myy + mzz + (mxy - mxz - myz));
    pop[22] = multiplyTerm * (pics2 - ux - uy + uz + mxx + myy + mzz + (mxy - mxz - myz));
    pop[23] = multiplyTerm * (pics2 + ux - uy + uz + mxx + myy + mzz - (mxy - mxz + myz));
    pop[24] = multiplyTerm * (pics2 - ux + uy - uz + mxx + myy + mzz - (mxy - mxz + myz));
    pop[25] = multiplyTerm * (pics2 - ux + uy + uz + mxx + myy + mzz - (mxy + mxz - myz));
    pop[26] = multiplyTerm * (pics2 + ux - uy - uz + mxx + myy + mzz - (mxy + mxz - myz));
}

__device__ inline void pop_reconstruction_collision(dfloat rho,
                                                    dfloat ux, dfloat uy, dfloat uz,
                                                    dfloat mxx, dfloat mxy, dfloat mxz, dfloat myy, dfloat myz, dfloat mzz,
                                                    dfloat *pop)
{

    const dfloat omegaVar = OMEGA;
    const dfloat t_omegaVar = static_cast<dfloat>(1.0) - omegaVar;
    const dfloat omegaVar_d2 = omegaVar / static_cast<dfloat>(2.0);

    dfloat mxx_star = (t_omegaVar * (mxx) + omegaVar_d2 * ux * ux);
    dfloat myy_star = (t_omegaVar * (myy) + omegaVar_d2 * uy * uy);
    dfloat mzz_star = (t_omegaVar * (mzz) + omegaVar_d2 * uz * uz);

    dfloat mxy_star = (t_omegaVar * (mxy) + omegaVar * ux * uy);
    dfloat mxz_star = (t_omegaVar * (mxz) + omegaVar * ux * uz);
    dfloat myz_star = (t_omegaVar * (myz) + omegaVar * uy * uz);

    dfloat multiplyTerm = rho * W0;
    const dfloat pics2 = static_cast<dfloat>(1.0) - cs2 * (mxx_star + myy_star + mzz_star);

    pop[0] = multiplyTerm * (pics2);

    multiplyTerm = rho * W1;
    pop[1] = multiplyTerm * (pics2 + ux + mxx_star);
    pop[2] = multiplyTerm * (pics2 - ux + mxx_star);
    pop[3] = multiplyTerm * (pics2 + uy + myy_star);
    pop[4] = multiplyTerm * (pics2 - uy + myy_star);
    pop[5] = multiplyTerm * (pics2 + uz + mzz_star);
    pop[6] = multiplyTerm * (pics2 - uz + mzz_star);

    multiplyTerm = rho * W2;
    pop[7] = multiplyTerm * (pics2 + ux + uy + mxx_star + myy_star + mxy_star);
    pop[8] = multiplyTerm * (pics2 - ux - uy + mxx_star + myy_star + mxy_star);
    pop[9] = multiplyTerm * (pics2 + ux + uz + mxx_star + mzz_star + mxz_star);
    pop[10] = multiplyTerm * (pics2 - ux - uz + mxx_star + mzz_star + mxz_star);
    pop[11] = multiplyTerm * (pics2 + uy + uz + myy_star + mzz_star + myz_star);
    pop[12] = multiplyTerm * (pics2 - uy - uz + myy_star + mzz_star + myz_star);
    pop[13] = multiplyTerm * (pics2 + ux - uy + mxx_star + myy_star - mxy_star);
    pop[14] = multiplyTerm * (pics2 - ux + uy + mxx_star + myy_star - mxy_star);
    pop[15] = multiplyTerm * (pics2 + ux - uz + mxx_star + mzz_star - mxz_star);
    pop[16] = multiplyTerm * (pics2 - ux + uz + mxx_star + mzz_star - mxz_star);
    pop[17] = multiplyTerm * (pics2 + uy - uz + myy_star + mzz_star - myz_star);
    pop[18] = multiplyTerm * (pics2 - uy + uz + myy_star + mzz_star - myz_star);

    multiplyTerm = rho * W3;
    pop[19] = multiplyTerm * (pics2 + ux + uy + uz + mxx_star + myy_star + mzz_star + (mxy_star + mxz_star + myz_star));
    pop[20] = multiplyTerm * (pics2 - ux - uy - uz + mxx_star + myy_star + mzz_star + (mxy_star + mxz_star + myz_star));
    pop[21] = multiplyTerm * (pics2 + ux + uy - uz + mxx_star + myy_star + mzz_star + (mxy_star - mxz_star - myz_star));
    pop[22] = multiplyTerm * (pics2 - ux - uy + uz + mxx_star + myy_star + mzz_star + (mxy_star - mxz_star - myz_star));
    pop[23] = multiplyTerm * (pics2 + ux - uy + uz + mxx_star + myy_star + mzz_star - (mxy_star - mxz_star + myz_star));
    pop[24] = multiplyTerm * (pics2 - ux + uy - uz + mxx_star + myy_star + mzz_star - (mxy_star - mxz_star + myz_star));
    pop[25] = multiplyTerm * (pics2 - ux + uy + uz + mxx_star + myy_star + mzz_star - (mxy_star + mxz_star - myz_star));
    pop[26] = multiplyTerm * (pics2 + ux - uy - uz + mxx_star + myy_star + mzz_star - (mxy_star + mxz_star - myz_star));
}

#endif
