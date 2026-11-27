#ifndef GLOBAL_STATISTICS_CUH
#define GLOBAL_STATISTICS_CUH

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include "localFieldStatistics.cuh"
#include "model.h"

constexpr const char *GLOBAL_STATISTICS_HEADER =
    "step,t_star,kinetic_energy_over_U2,dissipation_star,probe_ux,probe_uy,probe_uz";
constexpr double PROBE_Y_OVER_L = (RE < 10000 ? 7.0 : 5.0) / 150.0;

inline void reconcileGlobalStatistics(const std::filesystem::path &statistics_directory, const size_t completed_step)
{
    const std::filesystem::path path = statistics_directory / "global_statistics.csv";
    if (!std::filesystem::exists(path))
        return;

    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Failed to open global statistics: " + path.string());

    std::filesystem::path temporary = path;
    temporary += ".restart.tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output)
        throw std::runtime_error("Failed to create temporary global statistics: " + temporary.string());

    std::string header;
    if (!std::getline(input, header) || header != GLOBAL_STATISTICS_HEADER)
        throw std::runtime_error("Invalid global statistics header: " + path.string());
    output << header << '\n';

    std::unordered_set<size_t> retained_steps;
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty())
            continue;

        const std::size_t separator = line.find(',');
        if (separator == std::string::npos)
            throw std::runtime_error("Invalid global statistics line: " + line);
        std::istringstream parser(line.substr(0, separator));
        size_t step = 0;
        if (!(parser >> step))
            throw std::runtime_error("Invalid global statistics line: " + line);

        if (step <= completed_step && retained_steps.insert(step).second)
            output << line << '\n';
    }
    if (!input.eof())
        throw std::runtime_error("Failed while reading global statistics: " + path.string());

    output.close();
    if (!output)
        throw std::runtime_error("Failed while rewriting global statistics: " + temporary.string());

    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error)
        throw std::runtime_error("Failed to publish reconciled global statistics: " + error.message());
}

__device__ inline double trapezoidalWeight(const int index, const int size)
{
    return index == 0 || index == size - 1 ? 0.5 : 1.0;
}

struct GlobalStatisticsAt
{
    LBMState state;
    int global_z_start;

    __device__ double2 operator()(const std::size_t node) const
    {
        const int x = static_cast<int>(node % NX);
        const int y = static_cast<int>(node / NX % NY);
        const int z = static_cast<int>(node / NX / NY);
        double values[NUM_LOCAL_FIELDS];
        evaluateLocalFields(
            state,
            idxScalarBlock(
                x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
                x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ),
            values);
        const double weight = trapezoidalWeight(x, GLOBAL_NX) *
            trapezoidalWeight(y, GLOBAL_NY) *
            trapezoidalWeight(global_z_start + z, GLOBAL_NZ);
        return {weight * (values[LOCAL_FIELD_UX2] + values[LOCAL_FIELD_UY2] + values[LOCAL_FIELD_UZ2]),
                weight * values[LOCAL_FIELD_S2]};
    }
};

struct GlobalStatisticsSum
{
    __host__ __device__ double2 operator()(const double2 &a, const double2 &b) const
    {
        return {a.x + b.x, a.y + b.y};
    }

    __host__ __device__ double3 operator()(const double3 &a, const double3 &b) const
    {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }
};

struct ProbeVelocityAt
{
    LBMState state;
    int global_z_start;

    __device__ double3 operator()(const int corner) const
    {
        const double position[3] = {
            0.5 * (GLOBAL_NX - 1), PROBE_Y_OVER_L * (GLOBAL_NX - 1), 0.5 * (GLOBAL_NZ - 1)};
        int node[3];
        double weight = 1.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            const int low = static_cast<int>(position[axis]);
            const int upper = corner >> axis & 1;
            node[axis] = low + upper;
            weight *= upper ? position[axis] - low : 1.0 - position[axis] + low;
        }

        const int z = node[2] - global_z_start;
        if (weight == 0.0 || z < 0 || z >= NZ)
            return {0.0, 0.0, 0.0};

        double values[NUM_LOCAL_FIELDS];
        evaluateLocalFields(
            state,
            idxScalarBlock(
                node[0] % BLOCK_NX, node[1] % BLOCK_NY, z % BLOCK_NZ,
                node[0] / BLOCK_NX, node[1] / BLOCK_NY, z / BLOCK_NZ),
            values);
        return {weight * values[LOCAL_FIELD_UX], weight * values[LOCAL_FIELD_UY], weight * values[LOCAL_FIELD_UZ]};
    }
};

inline void writeGlobalStatistics(
    const std::vector<std::unique_ptr<DeviceDomain>> &domains,
    const std::vector<LBMState *> &states,
    const size_t step,
    const std::filesystem::path &statistics_directory)
{
    double2 sum{0.0, 0.0};
    double3 probe{0.0, 0.0, 0.0};
    for (size_t gpu = 0; gpu < states.size(); ++gpu)
    {
        checkCuda(cudaSetDevice(domains[gpu]->device_id));
        sum = GlobalStatisticsSum{}(sum, thrust::transform_reduce(
            thrust::cuda::par,
            thrust::counting_iterator<std::size_t>(0),
            thrust::counting_iterator<std::size_t>(NUMBER_PHYSICAL_LBM_NODES),
            GlobalStatisticsAt{*states[gpu], domains[gpu]->info.global_z_start},
            double2{0.0, 0.0},
            GlobalStatisticsSum{}));
        probe = GlobalStatisticsSum{}(probe, thrust::transform_reduce(
            thrust::cuda::par,
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(8),
            ProbeVelocityAt{*states[gpu], domains[gpu]->info.global_z_start},
            double3{0.0, 0.0, 0.0},
            GlobalStatisticsSum{}));
    }

    constexpr double volume = static_cast<double>(GLOBAL_NX - 1) *
        static_cast<double>(GLOBAL_NY - 1) * static_cast<double>(GLOBAL_NZ - 1);
    constexpr double velocity = static_cast<double>(U_MAX);
    constexpr double length = static_cast<double>(GLOBAL_NX - 1);
    const double kinetic_energy = 0.5 * sum.x / volume / (velocity * velocity);
    const double dissipation = 2.0 * static_cast<double>(VISC) * sum.y / volume *
        length / (velocity * velocity * velocity);

    const std::filesystem::path path = statistics_directory / "global_statistics.csv";
    const bool write_header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
    std::ofstream file(path, std::ios_base::app);
    if (write_header)
        file << GLOBAL_STATISTICS_HEADER << '\n';
    file << step << ',' << std::setprecision(std::numeric_limits<double>::max_digits10)
         << static_cast<double>(step) * velocity / length << ','
         << kinetic_energy << ',' << dissipation << ','
         << probe.x / velocity << ',' << probe.y / velocity << ',' << probe.z / velocity << '\n';
    if (!file)
        throw std::runtime_error("Failed to write global statistics: " + path.string());
}

#endif
