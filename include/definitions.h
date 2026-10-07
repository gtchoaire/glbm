#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "model.h"
#include "index.h"

/* --------------------------- CONSTANTS --------------------------- */

constexpr dfloat OMEGA = 1.0 / TAU;            // (tau)^-1
constexpr dfloat OMEGAd2 = OMEGA / 2.0;        // OMEGA/2
constexpr dfloat OMEGAd9 = OMEGA / 9.0;        // OMEGA/9
constexpr dfloat T_OMEGA = 1.0 - OMEGA;        // 1-OMEGA
constexpr dfloat TT_OMEGA = 1.0 - 0.5 * OMEGA; // 1.0 - OMEGA/2
constexpr dfloat OMEGA_P1 = 1.0 + OMEGA;       // 1+ OMEGA
constexpr dfloat TT_OMEGA_T3 = TT_OMEGA * 3.0; // 3*(1-0.5*OMEGA)

#define SQRT_2 (1.41421356237309504880168872420969807856967187537)
#define SQRT_10 (3.162277660168379331998893544432718533719555139325)

constexpr dfloat ONESIXTH = 1.0 / 6.0;
constexpr dfloat ONETHIRD = 1.0 / 3.0;

constexpr const int BLOCK_NX = 8;
constexpr const int BLOCK_NY = 8;
constexpr const int BLOCK_NZ = 4;

#define BLOCK_LBM_SIZE (BLOCK_NX * BLOCK_NY * BLOCK_NZ)

constexpr const size_t NUM_BLOCK_X = NX / BLOCK_NX;
constexpr const size_t NUM_BLOCK_Y = NY / BLOCK_NY;
constexpr const size_t NUM_BLOCK_Z = NZ / BLOCK_NZ;

constexpr const size_t NUM_BLOCK = NUM_BLOCK_X * NUM_BLOCK_Y * NUM_BLOCK_Z;

constexpr const size_t NUMBER_LBM_NODES = NUM_BLOCK * BLOCK_LBM_SIZE;
constexpr const size_t MEM_SIZE_BLOCK_LBM = sizeof(dfloat) * BLOCK_LBM_SIZE * NUMBER_MOMENTS;
// memory size
constexpr const size_t MEM_SIZE_SCALAR = sizeof(dfloat) * NUMBER_LBM_NODES;
constexpr const size_t MEM_SIZE_MOM = sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS;

#endif