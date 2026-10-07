#include "../include/globalStructs.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

#if NUM_GPUS == 1

__host__ int runSingleGPUAB()
{

    SubdomainInfo subdomain(GPU_INDEX, 0, GLOBAL_NZ - 1, GLOBAL_NZ, GLOBAL_NZ,
#ifdef BC_Z_PERIODIC
                            false,
                            false);
#else
                            true,
                            true);

#endif

    DeviceDomain domain(GPU_INDEX, 0, subdomain);

    const dim3 thread_block(BLOCK_NX, BLOCK_NY, BLOCK_NZ);

    const dim3 grid_block(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    if (!initializeABDomain(domain, grid_block, thread_block))
    {
        return 1;
    }

    unsigned int step = 0U;

    reportCudaMemoryUsage(domain.device_id);

    ABAsyncOutputPipeline output_pipeline;

    while (step < final_step)
    {
        if (!advanceOneABStep(domain, grid_block, thread_block, step == 0U))
        {
            return 1;
        }

        ++step;
    }

    checkCudaErrors(cudaSetDevice(domain.device_id));
    checkCudaErrors(cudaDeviceSynchronize());

    freeABOutputStaging(domain);
    freePopulationHaloBuffers(domain);

    return 0;
}

#endif

int main()
{

    return 0;
}