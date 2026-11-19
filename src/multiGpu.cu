#include "../include/multiGpu.cuh"

#include <cuda_runtime.h>

#include "../include/model.h"
#include "../include/globalFunctions.cuh"
#include "../include/mlbm.cuh"
#include "../include/var.h"
#include PROPERTIES

namespace
{
constexpr size_t HALO_PLANE_VALUES = static_cast<size_t>(QF) * NX * NY;
constexpr size_t HALO_PLANE_BYTES = HALO_PLANE_VALUES * sizeof(dfloat);
constexpr unsigned int HALO_BLOCK_X = 32;
constexpr unsigned int HALO_BLOCK_Y = 8;

dim3 haloGrid()
{
    const dim3 block(HALO_BLOCK_X, HALO_BLOCK_Y);
    return dim3((NX + block.x - 1) / block.x, (NY + block.y - 1) / block.y);
}

bool deviceSupportsMappedHostMemory(const int device_id)
{
    int supported = 0;
    checkCuda(cudaDeviceGetAttribute(&supported, cudaDevAttrCanMapHostMemory, device_id));
    return supported != 0;
}

void allocateHaloSendBuffer(dfloat **device_buffer, dfloat **host_buffer, const bool mapped)
{
    const unsigned int flags = mapped ? cudaHostAllocPortable | cudaHostAllocMapped : cudaHostAllocPortable;
    checkCuda(cudaHostAlloc(reinterpret_cast<void **>(host_buffer), HALO_PLANE_BYTES, flags));
    if (mapped)
        checkCuda(cudaHostGetDevicePointer(reinterpret_cast<void **>(device_buffer), *host_buffer, 0));
    else
        checkCuda(cudaMalloc(reinterpret_cast<void **>(device_buffer), HALO_PLANE_BYTES));
}

void queueInteriorAfterPreviousBoundary(DeviceDomain &domain)
{
    checkCuda(cudaSetDevice(domain.device_id));
    checkCuda(cudaStreamWaitEvent(domain.interior_stream, domain.boundary_complete, 0));
}

void queuePinnedBufferReuse(DeviceDomain &producer, const cudaEvent_t consumer_receive_complete)
{
    checkCuda(cudaSetDevice(producer.device_id));
    checkCuda(cudaStreamWaitEvent(producer.stream, consumer_receive_complete, 0));
}

void queueFrontHaloSend(DeviceDomain &domain, const LBMState &state)
{
    const dim3 block(HALO_BLOCK_X, HALO_BLOCK_Y);
    checkCuda(cudaSetDevice(domain.device_id));
    gpuPackFrontInterfacePopulations<<<haloGrid(), block, 0, domain.stream>>>(state, NZ - 1, domain.population_halos.front_send);
    checkCuda(cudaGetLastError());
    if (!domain.population_halos.mapped_staging)
        checkCuda(cudaMemcpyAsync(domain.population_halos.h_front_staging, domain.population_halos.front_send, HALO_PLANE_BYTES, cudaMemcpyDeviceToHost, domain.stream));
    checkCuda(cudaEventRecord(domain.send_complete, domain.stream));
}

void queueBackHaloSend(DeviceDomain &domain, const LBMState &state)
{
    const dim3 block(HALO_BLOCK_X, HALO_BLOCK_Y);
    checkCuda(cudaSetDevice(domain.device_id));
    gpuPackBackInterfacePopulations<<<haloGrid(), block, 0, domain.stream>>>(state, 0, domain.population_halos.back_send);
    checkCuda(cudaGetLastError());
    if (!domain.population_halos.mapped_staging)
        checkCuda(cudaMemcpyAsync(domain.population_halos.h_back_staging, domain.population_halos.back_send, HALO_PLANE_BYTES, cudaMemcpyDeviceToHost, domain.stream));
    checkCuda(cudaEventRecord(domain.send_complete, domain.stream));
}

void queueHaloReceive(DeviceDomain &destination, const cudaEvent_t source_send_complete, const dfloat *host_staging, dfloat *device_halo)
{
    checkCuda(cudaSetDevice(destination.device_id));
    checkCuda(cudaStreamWaitEvent(destination.stream, source_send_complete, 0));
    checkCuda(cudaMemcpyAsync(device_halo, host_staging, HALO_PLANE_BYTES, cudaMemcpyHostToDevice, destination.stream));
    checkCuda(cudaEventRecord(destination.receive_complete, destination.stream));
}
}

SubDomainInfo makeSubDomainInfo(const int subdomain_id)
{
    const int global_z_start = subdomain_id * LOCAL_NZ;
    const int global_z_end = global_z_start + LOCAL_NZ - 1;
    return SubDomainInfo(
        GPU_INDEX + subdomain_id,
        global_z_start,
        global_z_end,
        LOCAL_NZ,
        GLOBAL_NZ,
        subdomain_id == 0,
        subdomain_id == NUM_GPUS - 1);
}

bool validateMultiGpuTopology()
{
    int available_devices = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&available_devices);
    if (count_status != cudaSuccess)
        return false;

    constexpr int last_device = GPU_INDEX + NUM_GPUS - 1;
    if (GPU_INDEX < 0 || last_device >= available_devices)
        return false;

    return true;
}

void allocateMomentHalos(DeviceDomain &domain)
{
    checkCuda(cudaSetDevice(domain.device_id));

    if constexpr (NUM_GPUS == 2)
    {
        domain.population_halos.mapped_staging = deviceSupportsMappedHostMemory(domain.device_id);
        allocateHaloSendBuffer(&domain.population_halos.back_send, &domain.population_halos.h_back_staging, domain.population_halos.mapped_staging);
        allocateHaloSendBuffer(&domain.population_halos.front_send, &domain.population_halos.h_front_staging, domain.population_halos.mapped_staging);
        checkCuda(cudaStreamCreateWithFlags(&domain.stream, cudaStreamNonBlocking));
        checkCuda(cudaStreamCreateWithFlags(&domain.interior_stream, cudaStreamNonBlocking));
        checkCuda(cudaEventCreateWithFlags(&domain.interior_complete, cudaEventDisableTiming));
        checkCuda(cudaEventCreateWithFlags(&domain.boundary_complete, cudaEventDisableTiming));
        checkCuda(cudaEventCreateWithFlags(&domain.send_complete, cudaEventDisableTiming));
        checkCuda(cudaEventCreateWithFlags(&domain.receive_complete, cudaEventDisableTiming));
        checkCuda(cudaEventRecord(domain.receive_complete, domain.stream));
        if (!domain.info.has_back_wall)
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&domain.population_halos.remote.back), HALO_PLANE_BYTES));
        if (!domain.info.has_front_wall)
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&domain.population_halos.remote.front), HALO_PLANE_BYTES));
    }

    checkCuda(cudaMalloc(reinterpret_cast<void **>(&domain.population_halos.remote.initial_wall), Q * sizeof(dfloat)));
}

void beginMomentHaloExchange(DeviceDomain &back, DeviceDomain &front, const LBMState &back_state, const LBMState &front_state)
{
    // Cross-device stream dependencies keep pinned-buffer ownership on the
    // GPUs and avoid four host-blocking waits per simulation step.
    queuePinnedBufferReuse(back, front.receive_complete);
    queuePinnedBufferReuse(front, back.receive_complete);
    queueInteriorAfterPreviousBoundary(back);
    queueInteriorAfterPreviousBoundary(front);
    queueFrontHaloSend(back, back_state);
    queueBackHaloSend(front, front_state);
}

void completeMomentHaloExchange(DeviceDomain &back, DeviceDomain &front)
{
    queueHaloReceive(front, back.send_complete, back.population_halos.h_front_staging, front.population_halos.remote.back);
    queueHaloReceive(back, front.send_complete, front.population_halos.h_back_staging, back.population_halos.remote.front);
}

void freeMomentHalos(DeviceDomain &domain)
{
    checkCuda(cudaSetDevice(domain.device_id));

    if constexpr (NUM_GPUS == 2)
    {
        if (!domain.population_halos.mapped_staging)
        {
            cudaFree(domain.population_halos.back_send);
            cudaFree(domain.population_halos.front_send);
        }
        cudaFree(domain.population_halos.remote.back);
        cudaFree(domain.population_halos.remote.front);
        cudaFreeHost(domain.population_halos.h_back_staging);
        cudaFreeHost(domain.population_halos.h_front_staging);
        cudaEventDestroy(domain.interior_complete);
        cudaEventDestroy(domain.boundary_complete);
        cudaEventDestroy(domain.send_complete);
        cudaEventDestroy(domain.receive_complete);
        cudaStreamDestroy(domain.interior_stream);
        cudaStreamDestroy(domain.stream);
    }

    cudaFree(domain.population_halos.remote.initial_wall);
    domain.population_halos = {};
}
