#include "../include/lbmState.cuh"


#include <new>

#include "../include/definitions.h"
#include "../include/globalFunctions.cuh"
#include PROPERTIES

namespace
{

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

    state.device(M_RHO_INDEX, index) = rho - RHO_0;
    state.device(M_UX_INDEX, index) = F_M_I_SCALE * ux;
    state.device(M_UY_INDEX, index) = F_M_I_SCALE * uy;
    state.device(M_UZ_INDEX, index) = F_M_I_SCALE * uz;

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

    state.device(M_MXX_INDEX, index) = F_M_II_SCALE * mxx;
    state.device(M_MXY_INDEX, index) = F_M_IJ_SCALE * mxy;
    state.device(M_MXZ_INDEX, index) = F_M_IJ_SCALE * mxz;
    state.device(M_MYY_INDEX, index) = F_M_II_SCALE * myy;
    state.device(M_MYZ_INDEX, index) = F_M_IJ_SCALE * myz;
    state.device(M_MZZ_INDEX, index) = F_M_II_SCALE * mzz;
}
} // namespace

void allocateLBMState(LBMState &state)
{
    state.h_moments =
        new (std::nothrow) dfloat[NUMBER_LBM_NODES * NUMBER_MOMENTS];
    if (state.h_moments == nullptr)
    {
        std::printf("Failed to allocate the host LBM moments.\n");
        std::exit(EXIT_FAILURE);
    }

    checkCuda(cudaMalloc(
        reinterpret_cast<void **>(&state.d_moments), MEM_SIZE_MOM));
}

void freeLBMState(LBMState &state)
{
    delete[] state.h_moments;
    state.h_moments = nullptr;

    if (state.d_moments != nullptr)
        cudaFree(state.d_moments);
    state.d_moments = nullptr;
}

void copyMacroscopicFieldsToHost(LBMState &state)
{
    checkCuda(cudaMemcpy(
        state.h_moments, state.d_moments,
        MEM_SIZE_MOM, cudaMemcpyDeviceToHost));
}

void initializeEquilibrium(LBMState &state, const dim3 grid, const dim3 block)
{
    gpuInitializeEquilibrium<<<grid, block>>>(state);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}
