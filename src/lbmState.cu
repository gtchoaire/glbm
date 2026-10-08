#include "../include/lbmState.cuh"

#include <iostream>
#include <new>

#include "../include/definitions.h"
#include "../include/globalFunctions.cuh"
#include PROPERTIES

namespace
{
bool checkCuda(const cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
        return true;

    std::cerr << "CUDA error while " << operation << ": "
              << cudaGetErrorString(status) << '\n';
    return false;
}

bool allocateField(dfloat *&host, dfloat *&device)
{
    host = new (std::nothrow) dfloat[NUMBER_LBM_NODES];
    if (host == nullptr)
        return false;

    return checkCuda(
        cudaMalloc(reinterpret_cast<void **>(&device), MEM_SIZE_SCALAR),
        "allocating an LBM field");
}

void freeField(dfloat *&host, dfloat *&device)
{
    delete[] host;
    host = nullptr;

    if (device != nullptr)
        cudaFree(device);
    device = nullptr;
}

bool copyFieldToHost(dfloat *host, const dfloat *device)
{
    return checkCuda(
        cudaMemcpy(host, device, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost),
        "copying an LBM field to the host");
}

__global__ void gpuInitializeEquilibrium(LBMState state)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
        return;

    const size_t index = idxScalarBlock(
        threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z);

    const dfloat rho = RHO_0;
    const dfloat ux = static_cast<dfloat>(0);
    const dfloat uy = static_cast<dfloat>(0);
    const dfloat uz = static_cast<dfloat>(0);

    state.d_rho[index] = rho - RHO_0;
    state.d_ux[index] = F_M_I_SCALE * ux;
    state.d_uy[index] = F_M_I_SCALE * uy;
    state.d_uz[index] = F_M_I_SCALE * uz;

    dfloat pop[Q];
#pragma unroll
    for (int i = 0; i < Q; ++i)
        pop[i] = f_eq(
            w[i] * rho,
            static_cast<dfloat>(3) *
                (ux * cx[i] + uy * cy[i] + uz * cz[i]),
            static_cast<dfloat>(1) - static_cast<dfloat>(1.5) *
                (ux * ux + uy * uy + uz * uz));

    const dfloat inverse_rho = static_cast<dfloat>(1) / rho;
    const dfloat mxx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] +
        pop[10] + pop[13] + pop[14] + pop[15] + pop[16] + pop[19] +
        pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] +
        pop[26]) * inverse_rho - cs2;
    const dfloat mxy = ((pop[7] + pop[8] + pop[19] + pop[20] + pop[21] +
        pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] +
        pop[26])) * inverse_rho;
    const dfloat mxz = ((pop[9] + pop[10] + pop[19] + pop[20] + pop[23] +
        pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] +
        pop[26])) * inverse_rho;
    const dfloat myy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] +
        pop[12] + pop[13] + pop[14] + pop[17] + pop[18] + pop[19] +
        pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] +
        pop[26]) * inverse_rho - cs2;
    const dfloat myz = ((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] +
        pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] +
        pop[24])) * inverse_rho;
    const dfloat mzz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] +
        pop[12] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] +
        pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] +
        pop[26]) * inverse_rho - cs2;

    state.d_mxx[index] = F_M_II_SCALE * mxx;
    state.d_mxy[index] = F_M_IJ_SCALE * mxy;
    state.d_mxz[index] = F_M_IJ_SCALE * mxz;
    state.d_myy[index] = F_M_II_SCALE * myy;
    state.d_myz[index] = F_M_IJ_SCALE * myz;
    state.d_mzz[index] = F_M_II_SCALE * mzz;
}
} // namespace

bool allocateLBMState(LBMState &state)
{
    const bool success =
        allocateField(state.h_rho, state.d_rho) &&
        allocateField(state.h_ux, state.d_ux) &&
        allocateField(state.h_uy, state.d_uy) &&
        allocateField(state.h_uz, state.d_uz) &&
        allocateField(state.h_mxx, state.d_mxx) &&
        allocateField(state.h_mxy, state.d_mxy) &&
        allocateField(state.h_mxz, state.d_mxz) &&
        allocateField(state.h_myy, state.d_myy) &&
        allocateField(state.h_myz, state.d_myz) &&
        allocateField(state.h_mzz, state.d_mzz);

    if (!success)
    {
        std::cerr << "Failed to allocate an LBM state.\n";
        freeLBMState(state);
    }

    return success;
}

void freeLBMState(LBMState &state)
{
    freeField(state.h_rho, state.d_rho);
    freeField(state.h_ux, state.d_ux);
    freeField(state.h_uy, state.d_uy);
    freeField(state.h_uz, state.d_uz);
    freeField(state.h_mxx, state.d_mxx);
    freeField(state.h_mxy, state.d_mxy);
    freeField(state.h_mxz, state.d_mxz);
    freeField(state.h_myy, state.d_myy);
    freeField(state.h_myz, state.d_myz);
    freeField(state.h_mzz, state.d_mzz);
}

bool copyMacroscopicFieldsToHost(LBMState &state)
{
    return
        copyFieldToHost(state.h_rho, state.d_rho) &&
        copyFieldToHost(state.h_ux, state.d_ux) &&
        copyFieldToHost(state.h_uy, state.d_uy) &&
        copyFieldToHost(state.h_uz, state.d_uz);
}

bool initializeEquilibrium(LBMState &state, const dim3 grid, const dim3 block)
{
    gpuInitializeEquilibrium<<<grid, block>>>(state);
    return checkCuda(cudaGetLastError(), "launching equilibrium initialization") &&
           checkCuda(cudaDeviceSynchronize(), "initializing equilibrium");
}
