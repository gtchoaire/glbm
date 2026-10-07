#include "../include/mlbm.cuh"

__global__ void gpuMomCollisionStreamAB(
    LBMState source,
    LBMState destination,
    const unsigned int *dNodeType,
    RemotePopulationHalos remote_halos,
    const bool has_back_wall,
    const bool has_front_wall,
    const int global_z_start,
    const unsigned int block_z_offset,
    const unsigned int block_z_stride,
    const bool initial_step)
{
    const int global_block_z = block_z_offset + blockIdx.z * block_z_stride;

    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * global_block_z;

    if (x >= NX || y >= NY || z >= NZ)
        return;

    const size_t destination_index = idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, global_block_z);

    const unsigned int node_type = dNodeType[destination_index];

    dfloat pop[Q];

    __shared__ dfloat shared_populations[BLOCK_LBM_SIZE * (Q - 1)];

    dfloat rho =
        RHO_0 + source.d_rho[destination_index];

    dfloat ux = source.d_ux[destination_index];
    dfloat uy = source.d_uy[destination_index];
    dfloat uz = source.d_uz[destination_index];

    dfloat mxx = source.d_mxx[destination_index];
    dfloat mxy = source.d_mxy[destination_index];
    dfloat mxz = source.d_mxz[destination_index];
    dfloat myy = source.d_myy[destination_index];
    dfloat myz = source.d_myz[destination_index];
    dfloat mzz = source.d_mzz[destination_index];

    moment_collision(rho, ux, uy, uz, &mxx, &mxy, &mxz, &myy, &myz, &mzz);
    pop_reconstruction(rho, ux, uy, uz, mxx, mxy, mxz, myy, myz, mzz, pop);
    // save_pop(shared_populations, pop);
    // __syncthreads();
    // streaming(shared_populations, pop);
}
