
#include "../include/temporalAverages.cuh"
#include "../include/globalFunctions.cuh"
#include <stdexcept>

#include PROPERTIES

namespace
{

    __global__ void gpuAccumulateTemporalAverages(
        LBMState state,
        dfloat *means,
        const dfloat inverse_samples)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        const int z = threadIdx.z + blockDim.z * blockIdx.z;

        if (x >= NX || y >= NY || z >= NZ)
            return;

        const size_t index = idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

        const dfloat rho = RHO_0 + state.device(M_RHO_INDEX, index);
        const dfloat ux = state.device(M_UX_INDEX, index) / F_M_I_SCALE;
        const dfloat uy = state.device(M_UY_INDEX, index) / F_M_I_SCALE;
        const dfloat uz = state.device(M_UZ_INDEX, index) / F_M_I_SCALE;

        const dfloat mxx = state.device(M_MXX_INDEX, index) / F_M_II_SCALE;
        const dfloat mxy = state.device(M_MXY_INDEX, index) / F_M_IJ_SCALE;
        const dfloat mxz = state.device(M_MXZ_INDEX, index) / F_M_IJ_SCALE;
        const dfloat myy = state.device(M_MYY_INDEX, index) / F_M_II_SCALE;
        const dfloat myz = state.device(M_MYZ_INDEX, index) / F_M_IJ_SCALE;
        const dfloat mzz = state.device(M_MZZ_INDEX, index) / F_M_II_SCALE;

        const dfloat factor = static_cast<dfloat>(0.5) * as2 * OMEGA / T_OMEGA;

        const dfloat Sxx = factor * (ux * ux - mxx);
        const dfloat Sxy = factor * (ux * uy - mxy);
        const dfloat Sxz = factor * (ux * uz - mxz);
        const dfloat Syy = factor * (uy * uy - myy);
        const dfloat Syz = factor * (uy * uz - myz);
        const dfloat Szz = factor * (uz * uz - mzz);

        const dfloat values[NUM_AVERAGE_FIELDS] = {rho, ux, uy, uz, Sxx, Sxy, Sxz, Syy, Syz, Szz};

#pragma unroll
        for (int field = 0; field < NUM_AVERAGE_FIELDS; ++field)
        {
            const size_t offset = static_cast<size_t>(field) * NUMBER_LBM_NODES + index;
            means[offset] += (values[field] - means[offset]) * inverse_samples;
        }
    }
}

void allocateTemporalAverages(TemporalAverages &averages)
{
    const size_t bytes = NUM_AVERAGE_FIELDS * NUMBER_LBM_NODES * sizeof(dfloat);
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&averages.d_means), bytes));
    checkCuda(cudaMemset(averages.d_means, 0, bytes));
    averages.samples = 0;
}

void copyTemporalAveragesToHost(const TemporalAverages &averages, std::vector<dfloat> &host_means)
{
    const size_t total_values = static_cast<size_t>(NUM_AVERAGE_FIELDS) * NUMBER_LBM_NODES;
    host_means.resize(total_values);
    const size_t total_bytes = total_values * sizeof(dfloat);
    checkCuda(cudaMemcpy(host_means.data(), averages.d_means, total_bytes, cudaMemcpyDeviceToHost));
}

void accumulateTemporalAverages(TemporalAverages &averages, const LBMState &state, dim3 grid, dim3 block, cudaStream_t stream)
{
    const size_t next_sample = averages.samples + 1;
    const dfloat inverse_samples = static_cast<dfloat>(1) / static_cast<dfloat>(next_sample);
    gpuAccumulateTemporalAverages<<<grid, block, 0, stream>>>(state, averages.d_means, inverse_samples);
    checkCuda(cudaGetLastError());
    averages.samples = next_sample;
}

void copyTemporalAveragesProbe(const TemporalAverages &averages, int x, int y, int z, dfloat *values)
{
    if (x < 0 || x >= NX || y < 0 || y >= NY || z < 0 || z >= NZ)
        throw std::runtime_error("Invalid temporal averages probe");

    const size_t index = idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);

    for (int field = 0; field < NUM_AVERAGE_FIELDS; ++field)
    {
        const size_t offset = static_cast<size_t>(field) * NUMBER_LBM_NODES + index;
        checkCuda(cudaMemcpy(&values[field], averages.d_means + offset, sizeof(dfloat), cudaMemcpyDeviceToHost));
    }
}

void freeTemporalAverages(TemporalAverages &averages)
{
    if (averages.d_means != nullptr)
        checkCuda(cudaFree(averages.d_means));

    averages.d_means = nullptr;
    averages.samples = 0;
}

void restoreTemporalAveragesFromHost(TemporalAverages &averages, const std::vector<dfloat> &host_means, const size_t samples)
{
    const size_t total_values = static_cast<size_t>(NUM_AVERAGE_FIELDS) * NUMBER_LBM_NODES;

    if (averages.d_means == nullptr)
        throw std::runtime_error("Temporal averages must be allocated before restoring.");

    if (host_means.size() != total_values)
        throw std::runtime_error("Invalid temporal averages checkpoint size.");

    checkCuda(cudaMemcpy(averages.d_means, host_means.data(), total_values * sizeof(dfloat), cudaMemcpyHostToDevice));
    averages.samples = samples;
}
