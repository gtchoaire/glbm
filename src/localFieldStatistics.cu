
#include "../include/localFieldStatistics.cuh"
#include "../include/globalFunctions.cuh"
#include <stdexcept>

namespace
{

    __global__ void gpuAccumulateLocalFieldStatistics(
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

        double values[NUM_LOCAL_FIELDS];
        evaluateLocalFields(state, index, values);

#pragma unroll
        for (int field = 0; field < NUM_LOCAL_FIELDS; ++field)
        {
            const size_t offset = static_cast<size_t>(field) * NUMBER_LBM_NODES + index;
            means[offset] += (values[field] - means[offset]) * inverse_samples;
        }
    }
}

void allocateLocalFieldStatistics(LocalFieldStatistics &statistics)
{
    const size_t bytes = NUM_LOCAL_FIELDS * NUMBER_LBM_NODES * sizeof(double);
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&statistics.d_means), bytes));
    checkCuda(cudaMemset(statistics.d_means, 0, bytes));
    statistics.samples = 0;
}

void copyLocalFieldStatisticsToHost(const LocalFieldStatistics &statistics, std::vector<double> &host_means)
{
    const size_t total_values = static_cast<size_t>(NUM_LOCAL_FIELDS) * NUMBER_LBM_NODES;
    host_means.resize(total_values);
    const size_t total_bytes = total_values * sizeof(double);
    checkCuda(cudaMemcpy(host_means.data(), statistics.d_means, total_bytes, cudaMemcpyDeviceToHost));
}

void accumulateLocalFieldStatistics(LocalFieldStatistics &statistics, const LBMState &state, dim3 grid, dim3 block, cudaStream_t stream)
{
    const size_t next_sample = statistics.samples + 1;
    const double inverse_samples = 1.0 / static_cast<double>(next_sample);
    gpuAccumulateLocalFieldStatistics<<<grid, block, 0, stream>>>(state, statistics.d_means, inverse_samples);
    checkCuda(cudaGetLastError());
    statistics.samples = next_sample;
}

void freeLocalFieldStatistics(LocalFieldStatistics &statistics)
{
    if (statistics.d_means != nullptr)
        checkCuda(cudaFree(statistics.d_means));

    statistics.d_means = nullptr;
    statistics.samples = 0;
}

void restoreLocalFieldStatisticsFromHost(LocalFieldStatistics &statistics, const std::vector<double> &host_means, const size_t samples)
{
    const size_t total_values = static_cast<size_t>(NUM_LOCAL_FIELDS) * NUMBER_LBM_NODES;

    if (statistics.d_means == nullptr)
        throw std::runtime_error("Local field statistics must be allocated before restoring.");

    if (host_means.size() != total_values)
        throw std::runtime_error("Invalid local field statistics size.");

    checkCuda(cudaMemcpy(statistics.d_means, host_means.data(), total_values * sizeof(double), cudaMemcpyHostToDevice));
    statistics.samples = samples;
}
