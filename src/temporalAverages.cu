
#include "../include/temporalAverages.cuh"
#include "../include/globalFunctions.cuh"
#include <stdexcept>

#include PROPERTIES

namespace
{

    __global__ void gpuAccumulateTemporalAverages(
        LBMState state,
        double *means,
        const double inverse_samples)
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        const int z = threadIdx.z + blockDim.z * blockIdx.z;

        if (x >= NX || y >= NY || z >= NZ)
            return;

        const size_t index = idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

        const double rho = static_cast<double>(RHO_0 + state.device(M_RHO_INDEX, index));
        const double ux = static_cast<double>(state.device(M_UX_INDEX, index)) / static_cast<double>(F_M_I_SCALE);
        const double uy = static_cast<double>(state.device(M_UY_INDEX, index)) / static_cast<double>(F_M_I_SCALE);
        const double uz = static_cast<double>(state.device(M_UZ_INDEX, index)) / static_cast<double>(F_M_I_SCALE);

        const double mxx = static_cast<double>(state.device(M_MXX_INDEX, index)) / static_cast<double>(F_M_II_SCALE);
        const double mxy = static_cast<double>(state.device(M_MXY_INDEX, index)) / static_cast<double>(F_M_IJ_SCALE);
        const double mxz = static_cast<double>(state.device(M_MXZ_INDEX, index)) / static_cast<double>(F_M_IJ_SCALE);
        const double myy = static_cast<double>(state.device(M_MYY_INDEX, index)) / static_cast<double>(F_M_II_SCALE);
        const double myz = static_cast<double>(state.device(M_MYZ_INDEX, index)) / static_cast<double>(F_M_IJ_SCALE);
        const double mzz = static_cast<double>(state.device(M_MZZ_INDEX, index)) / static_cast<double>(F_M_II_SCALE);

        const double factor = 0.5 * static_cast<double>(as2) *
            static_cast<double>(OMEGA) / static_cast<double>(T_OMEGA);

        const dfloat Sxx = factor * (ux * ux - mxx);
        const dfloat Sxy = factor * (ux * uy - mxy);
        const dfloat Sxz = factor * (ux * uz - mxz);
        const dfloat Syy = factor * (uy * uy - myy);
        const dfloat Syz = factor * (uy * uz - myz);
        const dfloat Szz = factor * (uz * uz - mzz);

        const double values[NUM_AVERAGE_FIELDS] = {rho, ux, uy, uz, Sxx, Sxy, Sxz, Syy, Syz, Szz};

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
    const size_t bytes = NUM_AVERAGE_FIELDS * NUMBER_LBM_NODES * sizeof(double);
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&averages.d_means), bytes));
    checkCuda(cudaMemset(averages.d_means, 0, bytes));
    averages.samples = 0;
}

void copyTemporalAveragesToHost(const TemporalAverages &averages, std::vector<double> &host_means)
{
    const size_t total_values = static_cast<size_t>(NUM_AVERAGE_FIELDS) * NUMBER_LBM_NODES;
    host_means.resize(total_values);
    const size_t total_bytes = total_values * sizeof(double);
    checkCuda(cudaMemcpy(host_means.data(), averages.d_means, total_bytes, cudaMemcpyDeviceToHost));
}

void accumulateTemporalAverages(TemporalAverages &averages, const LBMState &state, dim3 grid, dim3 block, cudaStream_t stream)
{
    const size_t next_sample = averages.samples + 1;
    const double inverse_samples = 1.0 / static_cast<double>(next_sample);
    gpuAccumulateTemporalAverages<<<grid, block, 0, stream>>>(state, averages.d_means, inverse_samples);
    checkCuda(cudaGetLastError());
    averages.samples = next_sample;
}

void copyTemporalAveragesProbe(const TemporalAverages &averages, int x, int y, int z, double *values)
{
    if (x < 0 || x >= NX || y < 0 || y >= NY || z < 0 || z >= NZ)
        throw std::runtime_error("Invalid temporal averages probe");

    const size_t index = idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);

    for (int field = 0; field < NUM_AVERAGE_FIELDS; ++field)
    {
        const size_t offset = static_cast<size_t>(field) * NUMBER_LBM_NODES + index;
        checkCuda(cudaMemcpy(&values[field], averages.d_means + offset, sizeof(double), cudaMemcpyDeviceToHost));
    }
}

void freeTemporalAverages(TemporalAverages &averages)
{
    if (averages.d_means != nullptr)
        checkCuda(cudaFree(averages.d_means));

    averages.d_means = nullptr;
    averages.samples = 0;
}

void restoreTemporalAveragesFromHost(TemporalAverages &averages, const std::vector<double> &host_means, const size_t samples)
{
    const size_t total_values = static_cast<size_t>(NUM_AVERAGE_FIELDS) * NUMBER_LBM_NODES;

    if (averages.d_means == nullptr)
        throw std::runtime_error("Temporal averages must be allocated before restoring.");

    if (host_means.size() != total_values)
        throw std::runtime_error("Invalid temporal averages checkpoint size.");

    checkCuda(cudaMemcpy(averages.d_means, host_means.data(), total_values * sizeof(double), cudaMemcpyHostToDevice));
    averages.samples = samples;
}
