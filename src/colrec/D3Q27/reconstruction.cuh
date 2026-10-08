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
    // Shifted populations: f'_i = f_i - rho_0 w_i.
    // Writing the expression directly in terms of drho avoids subtracting
    // two values of order w_i after rounding them to single precision.
    const dfloat drho = rho - RHO_0;
    const dfloat trace_term = -cs2 * (mxx + myy + mzz);
    const dfloat rhoW0 = rho * W0;
    const dfloat rhoW1 = rho * W1;
    const dfloat rhoW2 = rho * W2;
    const dfloat rhoW3 = rho * W3;

    pop[0] = W0 * drho + rhoW0 * trace_term;

    pop[1] = W1 * drho + rhoW1 * (trace_term + ux + mxx);
    pop[2] = W1 * drho + rhoW1 * (trace_term - ux + mxx);
    pop[3] = W1 * drho + rhoW1 * (trace_term + uy + myy);
    pop[4] = W1 * drho + rhoW1 * (trace_term - uy + myy);
    pop[5] = W1 * drho + rhoW1 * (trace_term + uz + mzz);
    pop[6] = W1 * drho + rhoW1 * (trace_term - uz + mzz);

    pop[7] = W2 * drho + rhoW2 * (trace_term + ux + uy + mxx + myy + mxy);
    pop[8] = W2 * drho + rhoW2 * (trace_term - ux - uy + mxx + myy + mxy);
    pop[9] = W2 * drho + rhoW2 * (trace_term + ux + uz + mxx + mzz + mxz);
    pop[10] = W2 * drho + rhoW2 * (trace_term - ux - uz + mxx + mzz + mxz);
    pop[11] = W2 * drho + rhoW2 * (trace_term + uy + uz + myy + mzz + myz);
    pop[12] = W2 * drho + rhoW2 * (trace_term - uy - uz + myy + mzz + myz);
    pop[13] = W2 * drho + rhoW2 * (trace_term + ux - uy + mxx + myy - mxy);
    pop[14] = W2 * drho + rhoW2 * (trace_term - ux + uy + mxx + myy - mxy);
    pop[15] = W2 * drho + rhoW2 * (trace_term + ux - uz + mxx + mzz - mxz);
    pop[16] = W2 * drho + rhoW2 * (trace_term - ux + uz + mxx + mzz - mxz);
    pop[17] = W2 * drho + rhoW2 * (trace_term + uy - uz + myy + mzz - myz);
    pop[18] = W2 * drho + rhoW2 * (trace_term - uy + uz + myy + mzz - myz);

    pop[19] = W3 * drho + rhoW3 * (trace_term + ux + uy + uz + mxx + myy + mzz + (mxy + mxz + myz));
    pop[20] = W3 * drho + rhoW3 * (trace_term - ux - uy - uz + mxx + myy + mzz + (mxy + mxz + myz));
    pop[21] = W3 * drho + rhoW3 * (trace_term + ux + uy - uz + mxx + myy + mzz + (mxy - mxz - myz));
    pop[22] = W3 * drho + rhoW3 * (trace_term - ux - uy + uz + mxx + myy + mzz + (mxy - mxz - myz));
    pop[23] = W3 * drho + rhoW3 * (trace_term + ux - uy + uz + mxx + myy + mzz - (mxy - mxz + myz));
    pop[24] = W3 * drho + rhoW3 * (trace_term - ux + uy - uz + mxx + myy + mzz - (mxy - mxz + myz));
    pop[25] = W3 * drho + rhoW3 * (trace_term - ux + uy + uz + mxx + myy + mzz - (mxy + mxz - myz));
    pop[26] = W3 * drho + rhoW3 * (trace_term + ux - uy - uz + mxx + myy + mzz - (mxy + mxz - myz));
}

__device__ __forceinline__ dfloat reconstruct_population(
    const int i,
    const dfloat rho,
    const dfloat ux, const dfloat uy, const dfloat uz,
    const dfloat mxx, const dfloat mxy, const dfloat mxz,
    const dfloat myy, const dfloat myz, const dfloat mzz)
{
    const dfloat drho = rho - RHO_0;
    const dfloat trace_term = -cs2 * (mxx + myy + mzz);
    const dfloat rhoW0 = rho * W0;
    const dfloat rhoW1 = rho * W1;
    const dfloat rhoW2 = rho * W2;
    const dfloat rhoW3 = rho * W3;

    switch (i)
    {
    case 0: return W0 * drho + rhoW0 * trace_term;
    case 1: return W1 * drho + rhoW1 * (trace_term + ux + mxx);
    case 2: return W1 * drho + rhoW1 * (trace_term - ux + mxx);
    case 3: return W1 * drho + rhoW1 * (trace_term + uy + myy);
    case 4: return W1 * drho + rhoW1 * (trace_term - uy + myy);
    case 5: return W1 * drho + rhoW1 * (trace_term + uz + mzz);
    case 6: return W1 * drho + rhoW1 * (trace_term - uz + mzz);
    case 7: return W2 * drho + rhoW2 * (trace_term + ux + uy + mxx + myy + mxy);
    case 8: return W2 * drho + rhoW2 * (trace_term - ux - uy + mxx + myy + mxy);
    case 9: return W2 * drho + rhoW2 * (trace_term + ux + uz + mxx + mzz + mxz);
    case 10: return W2 * drho + rhoW2 * (trace_term - ux - uz + mxx + mzz + mxz);
    case 11: return W2 * drho + rhoW2 * (trace_term + uy + uz + myy + mzz + myz);
    case 12: return W2 * drho + rhoW2 * (trace_term - uy - uz + myy + mzz + myz);
    case 13: return W2 * drho + rhoW2 * (trace_term + ux - uy + mxx + myy - mxy);
    case 14: return W2 * drho + rhoW2 * (trace_term - ux + uy + mxx + myy - mxy);
    case 15: return W2 * drho + rhoW2 * (trace_term + ux - uz + mxx + mzz - mxz);
    case 16: return W2 * drho + rhoW2 * (trace_term - ux + uz + mxx + mzz - mxz);
    case 17: return W2 * drho + rhoW2 * (trace_term + uy - uz + myy + mzz - myz);
    case 18: return W2 * drho + rhoW2 * (trace_term - uy + uz + myy + mzz - myz);
    case 19: return W3 * drho + rhoW3 * (trace_term + ux + uy + uz + mxx + myy + mzz + (mxy + mxz + myz));
    case 20: return W3 * drho + rhoW3 * (trace_term - ux - uy - uz + mxx + myy + mzz + (mxy + mxz + myz));
    case 21: return W3 * drho + rhoW3 * (trace_term + ux + uy - uz + mxx + myy + mzz + (mxy - mxz - myz));
    case 22: return W3 * drho + rhoW3 * (trace_term - ux - uy + uz + mxx + myy + mzz + (mxy - mxz - myz));
    case 23: return W3 * drho + rhoW3 * (trace_term + ux - uy + uz + mxx + myy + mzz - (mxy - mxz + myz));
    case 24: return W3 * drho + rhoW3 * (trace_term - ux + uy - uz + mxx + myy + mzz - (mxy - mxz + myz));
    case 25: return W3 * drho + rhoW3 * (trace_term - ux + uy + uz + mxx + myy + mzz - (mxy + mxz - myz));
    default: return W3 * drho + rhoW3 * (trace_term + ux - uy - uz + mxx + myy + mzz - (mxy + mxz - myz));
    }
}

#endif
