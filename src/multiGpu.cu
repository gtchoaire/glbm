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
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&domain.population_halos.back_send), HALO_PLANE_BYTES));
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&domain.population_halos.front_send), HALO_PLANE_BYTES));
    checkCuda(cudaMallocHost(reinterpret_cast<void **>(&domain.population_halos.h_back_staging), HALO_PLANE_BYTES));
    checkCuda(cudaMallocHost(reinterpret_cast<void **>(&domain.population_halos.h_front_staging), HALO_PLANE_BYTES));
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

    checkCuda(cudaMalloc(reinterpret_cast<void **>(&domain.population_halos.remote.initial_wall), Q * sizeof(dfloat)));
}

void beginMomentHaloExchange(
    DeviceDomain &back, DeviceDomain &front,
    const LBMState &back_state, const LBMState &front_state)
{
    const dim3 block(HALO_BLOCK_X, HALO_BLOCK_Y);
    const dim3 grid((NX + block.x - 1) / block.x, (NY + block.y - 1) / block.y);

    // Each pinned buffer is produced by one device and consumed by the other.
    // Wait only for the previous host-to-device copy, not for the boundary
    // kernel queued after it on the same stream.
    checkCuda(cudaSetDevice(back.device_id));
    checkCuda(cudaEventSynchronize(back.receive_complete));
    checkCuda(cudaSetDevice(front.device_id));
    checkCuda(cudaEventSynchronize(front.receive_complete));

    checkCuda(cudaSetDevice(back.device_id));
    checkCuda(cudaStreamWaitEvent(back.interior_stream, back.boundary_complete, 0));
    gpuPackFrontInterfacePopulations<<<grid, block, 0, back.stream>>>(
        back_state, NZ - 1, back.population_halos.front_send);
    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpyAsync(back.population_halos.h_front_staging,
        back.population_halos.front_send, HALO_PLANE_BYTES,
        cudaMemcpyDeviceToHost, back.stream));
    checkCuda(cudaEventRecord(back.send_complete, back.stream));

    checkCuda(cudaSetDevice(front.device_id));
    checkCuda(cudaStreamWaitEvent(front.interior_stream, front.boundary_complete, 0));
    gpuPackBackInterfacePopulations<<<grid, block, 0, front.stream>>>(
        front_state, 0, front.population_halos.back_send);
    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpyAsync(front.population_halos.h_back_staging,
        front.population_halos.back_send, HALO_PLANE_BYTES,
        cudaMemcpyDeviceToHost, front.stream));
    checkCuda(cudaEventRecord(front.send_complete, front.stream));
}

void completeMomentHaloExchange(DeviceDomain &back, DeviceDomain &front)
{
    checkCuda(cudaSetDevice(back.device_id));
    checkCuda(cudaEventSynchronize(back.send_complete));
    checkCuda(cudaSetDevice(front.device_id));
    checkCuda(cudaMemcpyAsync(front.population_halos.remote.back,
        back.population_halos.h_front_staging, HALO_PLANE_BYTES,
        cudaMemcpyHostToDevice, front.stream));
    checkCuda(cudaEventRecord(front.receive_complete, front.stream));

    checkCuda(cudaSetDevice(front.device_id));
    checkCuda(cudaEventSynchronize(front.send_complete));
    checkCuda(cudaSetDevice(back.device_id));
    checkCuda(cudaMemcpyAsync(back.population_halos.remote.front,
        front.population_halos.h_back_staging, HALO_PLANE_BYTES,
        cudaMemcpyHostToDevice, back.stream));
    checkCuda(cudaEventRecord(back.receive_complete, back.stream));
}

void freeMomentHalos(DeviceDomain &domain)
{
    checkCuda(cudaSetDevice(domain.device_id));
    cudaFree(domain.population_halos.back_send);
    cudaFree(domain.population_halos.front_send);
    cudaFree(domain.population_halos.remote.back);
    cudaFree(domain.population_halos.remote.front);
    cudaFree(domain.population_halos.remote.initial_wall);
    cudaFreeHost(domain.population_halos.h_back_staging);
    cudaFreeHost(domain.population_halos.h_front_staging);
    cudaEventDestroy(domain.interior_complete);
    cudaEventDestroy(domain.boundary_complete);
    cudaEventDestroy(domain.send_complete);
    cudaEventDestroy(domain.receive_complete);
    cudaStreamDestroy(domain.interior_stream);
    cudaStreamDestroy(domain.stream);
    domain.population_halos = {};
}
