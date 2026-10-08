#include "../include/simulation.cuh"

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

    for (int step = 0; success && step < N_STEPS; ++step)
    {
        gpuMomCollisionStreamAB<<<grid, block>>>(
            *current, *next, domain.d_node_type,
            domain.population_halos.remote,
            step == 0);
        success = checkCuda(cudaGetLastError(), "launching an AB LBM step");
        std::swap(current, next);

        const int completed_steps = step + 1;
        const bool save_output = completed_steps % OUTPUT_INTERVAL == 0 ||
                                 completed_steps == N_STEPS;
        if (success && save_output)
            success = checkCuda(
                          cudaDeviceSynchronize(),
                          "synchronizing VTI output") &&
                      copyMacroscopicFieldsToHost(*current) &&
                      writeVti(*current, completed_steps);
    }

    if (success)
        success = checkCuda(cudaDeviceSynchronize(), "executing AB LBM steps");
    if (domain.population_halos.remote.initial_wall != nullptr)
        cudaFree(domain.population_halos.remote.initial_wall);
    freeNodeTypes(domain);
    freeLBMState(domain.state);
    freeLBMState(domain.next_state);
    return success ? 0 : 1;
}
