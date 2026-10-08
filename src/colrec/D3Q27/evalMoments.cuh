#ifndef EVAL_MOMENTS_CUH
#define EVAL_MOMENTS_CUH

#include "properties.cuh"

__device__ inline void eval_moments(
    dfloat *const rhoVar, dfloat *const ux, dfloat *const uy, dfloat *const uz,
    dfloat *const mxx, dfloat *const mxy, dfloat *const mxz,
    dfloat *const myy, dfloat *const myz, dfloat *const mzz,
    const dfloat *const pop)
{
    // The stored populations are shifted by rho_0*w_i. Their sum is drho.
    const dfloat sumPop = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] +
        pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] +
        pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] +
        pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] +
        pop[24] + pop[25] + pop[26];
    *rhoVar = RHO_0 + sumPop;
    const dfloat invRho = 1 / (*rhoVar);

    *ux = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26]) - (pop[2] + pop[8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25])) * invRho;
    *uy = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[4] + pop[8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26])) * invRho;
    *uz = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26])) * invRho;

    *mxx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26] - cs2 * sumPop) * invRho;
    *mxy = ((pop[7] + pop[8] + pop[19] + pop[20] + pop[21] + pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26])) * invRho;
    *mxz = ((pop[9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26])) * invRho;
    *myy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26] - cs2 * sumPop) * invRho;
    *myz = ((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24])) * invRho;
    *mzz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26] - cs2 * sumPop) * invRho;
}

#endif
