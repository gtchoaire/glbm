#include "../include/simulation.cuh"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <utility>
#include <cuda_runtime.h>

#include "../include/definitions.h"
#include "../include/lbmState.cuh"
#include "../include/mlbm.cuh"
#include "../include/nodeType.h"
#include "../include/output.h"

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

double calculateMlups(
    const size_t nodes, const int steps, const double seconds)
{
    return seconds > 0.0
        ? static_cast<double>(nodes) * static_cast<double>(steps) /
              (1.0e6 * seconds)
        : 0.0;
}

void reportProgress(
    const int completed_steps,
    const double solver_seconds,
    const double wall_seconds)
{
    const double progress =
        100.0 * static_cast<double>(completed_steps) /
        static_cast<double>(N_STEPS);
    const double remaining_seconds = completed_steps > 0
        ? wall_seconds * static_cast<double>(N_STEPS - completed_steps) /
              static_cast<double>(completed_steps)
        : 0.0;
    const auto eta = static_cast<unsigned long long>(remaining_seconds + 0.5);
    const auto eta_hours = eta / 3600;
    const auto eta_minutes = (eta % 3600) / 60;
    const auto eta_seconds = eta % 60;

    std::printf(
        "Step %010d / %010d | %6.2f%% | "
        "MLUPS solver %10.3f | wall %10.3f | ETA %02llu:%02llu:%02llu\n",
        completed_steps,
        N_STEPS,
        progress,
        calculateMlups(NUMBER_LBM_NODES, completed_steps, solver_seconds),
        calculateMlups(NUMBER_LBM_NODES, completed_steps, wall_seconds),
        eta_hours,
        eta_minutes,
        eta_seconds);
}
}

int runSimulation()
{
    int available_gpus = 0;
    if (!checkCuda(cudaGetDeviceCount(&available_gpus), "detecting devices"))
        return 1;
    if (available_gpus < NUM_GPUS)
    {
        std::cerr << "The simulation requires " << NUM_GPUS
                  << " GPU(s), but only " << available_gpus << " were detected.\n";
        return 1;
    }
    if (!checkCuda(cudaSetDevice(0), "selecting GPU 0"))
        return 1;

    const SubDomainInfo subdomain(
        0, 0, GLOBAL_NZ - 1, LOCAL_NZ, GLOBAL_NZ, true, true);
    DeviceDomain domain(0, 0, subdomain);

    if (!allocateLBMState(domain.state) ||
        !allocateLBMState(domain.next_state) ||
        !initializeNodeTypes(domain))
    {
        freeNodeTypes(domain);
        freeLBMState(domain.state);
        freeLBMState(domain.next_state);
        return 1;
    }

    const dim3 block(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    const dim3 grid(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);
    bool success = initializeEquilibrium(domain.state, grid, block);
    if (success)
        success = checkCuda(
            cudaMalloc(
                reinterpret_cast<void **>(
                    &domain.population_halos.remote.initial_wall),
                Q * sizeof(dfloat)),
            "allocating initial wall populations");
    if (success)
    {
        gpuInitializeWallPopulations<<<1, 1>>>(
            domain.state,
            domain.population_halos.remote.initial_wall);
        success = checkCuda(
            cudaGetLastError(),
            "initializing wall populations");
    }
    LBMState *current = &domain.state;
    LBMState *next = &domain.next_state;

    cudaEvent_t solver_start = nullptr;
    cudaEvent_t solver_stop = nullptr;
    if (success)
        success = checkCuda(
            cudaEventCreate(&solver_start), "creating the MLUPS start event");
    if (success)
        success = checkCuda(
            cudaEventCreate(&solver_stop), "creating the MLUPS stop event");

    double solver_seconds = 0.0;
    int completed_steps = 0;
    const auto wall_start = std::chrono::steady_clock::now();
    if (success)
        success = checkCuda(
            cudaEventRecord(solver_start), "starting the MLUPS measurement");

    for (int step = 0; success && step < N_STEPS; ++step)
    {
        gpuMomCollisionStreamAB<<<grid, block>>>(
            *current, *next, domain.d_node_type,
            domain.population_halos.remote,
            step == 0);
        success = checkCuda(cudaGetLastError(), "launching an AB LBM step");
        std::swap(current, next);

        completed_steps = step + 1;
        const bool save_output = SAVE_VTI_OUTPUT &&
            (completed_steps % OUTPUT_INTERVAL == 0 ||
             completed_steps == N_STEPS);
        const bool print_progress =
            completed_steps % PRINT_INTERVAL == 0 ||
            completed_steps == N_STEPS;
        const bool measurement_boundary =
            save_output || print_progress;

        if (success && measurement_boundary)
        {
            float segment_milliseconds = 0.0F;
            success = checkCuda(
                          cudaEventRecord(solver_stop),
                          "stopping the MLUPS measurement") &&
                      checkCuda(
                          cudaEventSynchronize(solver_stop),
                          "synchronizing the MLUPS measurement") &&
                      checkCuda(
                          cudaEventElapsedTime(
                              &segment_milliseconds,
                              solver_start, solver_stop),
                          "reading the MLUPS measurement");
            solver_seconds +=
                static_cast<double>(segment_milliseconds) * 1.0e-3;
        }

        if (success && save_output)
            success = copyMacroscopicFieldsToHost(*current) &&
                      writeVti(*current, completed_steps);

        if (success && print_progress)
        {
            const double wall_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wall_start).count();
            reportProgress(completed_steps, solver_seconds, wall_seconds);
        }

        if (success && measurement_boundary && completed_steps < N_STEPS)
            success = checkCuda(
                cudaEventRecord(solver_start),
                "restarting the MLUPS measurement");
    }

    const auto wall_stop = std::chrono::steady_clock::now();
    if (success)
    {
        const double wall_seconds = std::chrono::duration<double>(
            wall_stop - wall_start).count();
        std::printf(
            "Completed %d steps in %.6f s | "
            "MLUPS solver %.3f | wall %.3f\n",
            completed_steps,
            wall_seconds,
            calculateMlups(NUMBER_LBM_NODES, completed_steps, solver_seconds),
            calculateMlups(NUMBER_LBM_NODES, completed_steps, wall_seconds));
    }

    if (solver_stop != nullptr)
        cudaEventDestroy(solver_stop);
    if (solver_start != nullptr)
        cudaEventDestroy(solver_start);
    if (domain.population_halos.remote.initial_wall != nullptr)
        cudaFree(domain.population_halos.remote.initial_wall);
    freeNodeTypes(domain);
    freeLBMState(domain.state);
    freeLBMState(domain.next_state);
    return success ? 0 : 1;
}
