#ifndef GLOBAL_FUNCTIONS_H
#define GLOBAL_FUNCTIONS_H

#include <builtin_types.h>
#include "globalStructs.cuh"
#include "var.h"
#include "definitions.h"

// f_i^eq = rho w_i (1 + 3(u*ci) + 9/2(u*ci)^2 - 3/2(u*u))
__host__ __device__
    dfloat __forceinline__
    f_eq(const dfloat rhow, const dfloat uc3, const dfloat p1_muu)
{
    return (rhow * (p1_muu + uc3 * (1.0 + uc3 * 0.5)));
}

__host__ __device__
    size_t __forceinline__
    idxMom(
        const int tx,
        const int ty,
        const int tz,
        const int mom,
        const int bx,
        const int by,
        const int bz)
{
    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ * (mom + NUMBER_MOMENTS * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz))))));
}

__host__ __device__
    size_t __forceinline__
    idxPopBlock(const unsigned int tx, const unsigned int ty, const unsigned int tz, const unsigned int pop)
{

    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ * (pop)));
}

__host__ __device__
    size_t __forceinline__
    idxScalarBlock(
        const int tx,
        const int ty,
        const int tz,
        const int bx,
        const int by,
        const int bz)
{
    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz)))));
}

__host__ __device__
    size_t __forceinline__
    idxBlock()
{
    return threadIdx.x +
           blockDim.x *
               (threadIdx.y +
                blockDim.y * (threadIdx.z +
                              blockDim.z * (blockIdx.x +
                                            gridDim.x * (blockIdx.y +
                                                         gridDim.y * blockIdx.z))));
}

__host__ __device__
    size_t __forceinline__
    idxScalarGlobal(unsigned int x, unsigned int y, unsigned int z)
{
    return x + NX * (y + NY * (z));
}

#endif