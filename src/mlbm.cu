#include "../include/mlbm.cuh"

namespace
{
constexpr int SHARED_HALO_NX = BLOCK_NX + 2;
constexpr int SHARED_HALO_NY = BLOCK_NY + 2;
constexpr int SHARED_HALO_NZ = BLOCK_NZ + 2;
constexpr int SHARED_HALO_SIZE =
    SHARED_HALO_NX * SHARED_HALO_NY * SHARED_HALO_NZ;

__device__ inline size_t blockedIndex(const int x, const int y, const int z)
{
    return idxScalarBlock(
        x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
        x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
}

__device__ inline void restorePhysicalPopulations(dfloat *populations)
{
#pragma unroll
    for (int i = 0; i < Q; ++i)
        populations[i] += RHO_0 * w[i];
}

__device__ inline void reconstructPostCollisionPopulationsFromMoments(
    const dfloat stored_rho,
    const dfloat ux, const dfloat uy, const dfloat uz,
    dfloat mxx, dfloat mxy, dfloat mxz,
    dfloat myy, dfloat myz, dfloat mzz,
    dfloat *populations)
{
    const dfloat rho = RHO_0 + stored_rho;
    pop_reconstruction(rho, ux, uy, uz, mxx, mxy, mxz, myy, myz, mzz, populations);
}

__device__ __forceinline__ bool crossesABTileCoordinate(
    const int coordinate,
    const int lattice_velocity,
    const int tile_size)
{
    const int source_in_tile = coordinate % tile_size - lattice_velocity;
    return source_in_tile < 0 || source_in_tile >= tile_size;
}

__device__ __forceinline__ bool usesInitialWallPopulation(
    const int source_x, const int source_y, const int source_z,
    const bool crosses_x_tile,
    const bool crosses_y_tile,
    const bool crosses_z_tile,
    const SubDomainInfo subdomain)
{
    const bool crosses_x_wall = source_x < 0 || source_x >= NX;
    const bool crosses_y_wall = source_y < 0 || source_y >= NY;
    const bool crosses_z_wall =
        (source_z < 0 && subdomain.has_back_wall) ||
        (source_z >= NZ && subdomain.has_front_wall);

    // Preserve the AB halo ownership order: z, then y, then x.
    return (crosses_z_tile && crosses_z_wall) || (!crosses_z_tile && crosses_y_tile && crosses_y_wall) ||
        (!crosses_z_tile && !crosses_y_tile && crosses_x_tile && crosses_x_wall);
}

__device__ __forceinline__ int interfacePopulation(
    const int direction, const int slot)
{
    constexpr int positive[QF] = {5, 9, 11, 16, 18, 19, 22, 23, 25};
    constexpr int negative[QF] = {6, 10, 12, 15, 17, 20, 21, 24, 26};
    return direction > 0 ? positive[slot] : negative[slot];
}

__device__ __forceinline__ int interfacePopulationSlot(
    const int population)
{
#pragma unroll
    for (int slot = 0; slot < QF; ++slot)
        if (interfacePopulation(static_cast<int>(cz[population]), slot) ==
            population)
            return slot;
    return -1;
}

__device__ __forceinline__ dfloat remotePopulation(
    const dfloat *halo, const int population, const int x, const int y)
{
    const int slot = interfacePopulationSlot(population);
    return halo[x + static_cast<size_t>(NX) *
        (y + static_cast<size_t>(NY) * slot)];
}

__device__ __forceinline__ dfloat sourceMoment(
    const LBMState source,
    const RemotePopulationHalos remote_halos,
    const SubDomainInfo subdomain,
    const int moment, const int x, const int y, const int z)
{
    const int local_z = z < 0 ? 0 : (z >= NZ ? NZ - 1 : z);
    return source.device(moment, blockedIndex(x, y, local_z));
}

__device__ inline dfloat reconstructPostCollisionPopulation(
    const LBMState &source,
    const size_t source_index,
    const int population_index,
    const bool collide_source_moments)
{
    const dfloat rho = RHO_0 + source.device(M_RHO_INDEX, source_index);
    const dfloat ux = source.device(M_UX_INDEX, source_index);
    const dfloat uy = source.device(M_UY_INDEX, source_index);
    const dfloat uz = source.device(M_UZ_INDEX, source_index);
    dfloat mxx = source.device(M_MXX_INDEX, source_index);
    dfloat mxy = source.device(M_MXY_INDEX, source_index);
    dfloat mxz = source.device(M_MXZ_INDEX, source_index);
    dfloat myy = source.device(M_MYY_INDEX, source_index);
    dfloat myz = source.device(M_MYZ_INDEX, source_index);
    dfloat mzz = source.device(M_MZZ_INDEX, source_index);

    if (collide_source_moments)
        moment_collision(
            rho, ux, uy, uz,
            &mxx, &mxy, &mxz, &myy, &myz, &mzz);

    return reconstruct_population(
        population_index, rho, ux, uy, uz,
        mxx, mxy, mxz, myy, myz, mzz);
}

__device__ inline dfloat reconstructCollidedPopulation(
    const dfloat *shared_moments,
    const int shared_index,
    const int population_index)
{
    return reconstruct_population(
        population_index,
        RHO_0 + shared_moments[M_RHO_INDEX * SHARED_HALO_SIZE + shared_index],
        shared_moments[M_UX_INDEX * SHARED_HALO_SIZE + shared_index],
        shared_moments[M_UY_INDEX * SHARED_HALO_SIZE + shared_index],
        shared_moments[M_UZ_INDEX * SHARED_HALO_SIZE + shared_index],
        shared_moments[M_MXX_INDEX * SHARED_HALO_SIZE + shared_index],
        shared_moments[M_MXY_INDEX * SHARED_HALO_SIZE + shared_index],
        shared_moments[M_MXZ_INDEX * SHARED_HALO_SIZE + shared_index],
        shared_moments[M_MYY_INDEX * SHARED_HALO_SIZE + shared_index],
        shared_moments[M_MYZ_INDEX * SHARED_HALO_SIZE + shared_index],
        shared_moments[M_MZZ_INDEX * SHARED_HALO_SIZE + shared_index]);
}

} // namespace

template<int Direction>
__device__ __forceinline__ void packInterfacePopulations(
    const LBMState source,
    const int z,
    dfloat *packed_populations)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= NX || y >= NY)
        return;

    const size_t source_index = blockedIndex(x, y, z);
    const size_t plane_index = x + static_cast<size_t>(NX) * y;
#pragma unroll
    for (int slot = 0; slot < QF; ++slot)
    {
        const int population = interfacePopulation(Direction, slot);
        packed_populations[plane_index + static_cast<size_t>(slot) * NX * NY] =
            reconstruct_population(
                population,
                RHO_0 + source.device(M_RHO_INDEX, source_index),
                source.device(M_UX_INDEX, source_index),
                source.device(M_UY_INDEX, source_index),
                source.device(M_UZ_INDEX, source_index),
                source.device(M_MXX_INDEX, source_index),
                source.device(M_MXY_INDEX, source_index),
                source.device(M_MXZ_INDEX, source_index),
                source.device(M_MYY_INDEX, source_index),
                source.device(M_MYZ_INDEX, source_index),
                source.device(M_MZZ_INDEX, source_index));
    }
}

__global__ void gpuPackFrontInterfacePopulations(
    const LBMState source, const int z, dfloat *packed_populations)
{
    packInterfacePopulations<1>(source, z, packed_populations);
}

__global__ void gpuPackBackInterfacePopulations(
    const LBMState source, const int z, dfloat *packed_populations)
{
    packInterfacePopulations<-1>(source, z, packed_populations);
}

__global__ void gpuInitializeWallPopulations(
    LBMState source,
    dfloat *initial_wall_populations)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    reconstructPostCollisionPopulationsFromMoments(
        source.device(M_RHO_INDEX, 0),
        source.device(M_UX_INDEX, 0),
        source.device(M_UY_INDEX, 0),
        source.device(M_UZ_INDEX, 0),
        source.device(M_MXX_INDEX, 0),
        source.device(M_MXY_INDEX, 0),
        source.device(M_MXZ_INDEX, 0),
        source.device(M_MYY_INDEX, 0),
        source.device(M_MYZ_INDEX, 0),
        source.device(M_MZZ_INDEX, 0),
        initial_wall_populations);
}

__global__ void gpuBootstrapABState(
    LBMState source,
    LBMState destination,
    const unsigned int *node_type,
    const RemotePopulationHalos remote_halos,
    const SubDomainInfo subdomain,
    const int block_z_offset)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z *
        (static_cast<int>(blockIdx.z) + block_z_offset);

    if (x >= NX || y >= NY || z >= NZ)
        return;

    const size_t destination_index = blockedIndex(x, y, z);
    const unsigned int destination_type = node_type[destination_index];

    dfloat pop[Q];

#pragma unroll
    for (int i = 0; i < Q; ++i)
    {
        const int velocity_x = static_cast<int>(cx[i]);
        const int velocity_y = static_cast<int>(cy[i]);
        const int velocity_z = static_cast<int>(cz[i]);
        const int unwrapped_source_x = x - velocity_x;
        const int unwrapped_source_y = y - velocity_y;
        const int unwrapped_source_z = z - velocity_z;

        const bool crosses_x_tile = crossesABTileCoordinate(x, velocity_x, AB_TILE_NX);
        const bool crosses_y_tile = crossesABTileCoordinate(y, velocity_y, AB_TILE_NY);
        const bool crosses_z_tile = crossesABTileCoordinate(z, velocity_z, AB_TILE_NZ);
        const bool crosses_tile = crosses_x_tile || crosses_y_tile || crosses_z_tile;

        if (usesInitialWallPopulation(unwrapped_source_x, unwrapped_source_y, unwrapped_source_z, crosses_x_tile, crosses_y_tile, crosses_z_tile, subdomain))
        {
            pop[i] = remote_halos.initial_wall[i];
            continue;
        }

        const int source_x = (unwrapped_source_x + NX) % NX;
        const int source_y = (unwrapped_source_y + NY) % NY;
        if (unwrapped_source_z < 0 || unwrapped_source_z >= NZ)
        {
            const dfloat *remote = unwrapped_source_z < 0 ?
                remote_halos.back : remote_halos.front;
            pop[i] = remotePopulation(remote, i, source_x, source_y);
        }
        else
            pop[i] = reconstructPostCollisionPopulation(
                source, blockedIndex(source_x, source_y, unwrapped_source_z),
                i, !crosses_tile);
    }

    dfloat rho;
    dfloat ux;
    dfloat uy;
    dfloat uz;
    dfloat mxx;
    dfloat mxy;
    dfloat mxz;
    dfloat myy;
    dfloat myz;
    dfloat mzz;

    if (destination_type == BULK)
        eval_moments(
            &rho, &ux, &uy, &uz,
            &mxx, &mxy, &mxz, &myy, &myz, &mzz, pop);
    else
    {
        restorePhysicalPopulations(pop);
        apply_boundaries(
            pop, &rho, &ux, &uy, &uz,
            &mxx, &mxy, &mxz, &myy, &myz, &mzz,
            destination_type);
    }

    destination.device(M_RHO_INDEX, destination_index) = rho - RHO_0;
    destination.device(M_UX_INDEX, destination_index) = F_M_I_SCALE * ux;
    destination.device(M_UY_INDEX, destination_index) = F_M_I_SCALE * uy;
    destination.device(M_UZ_INDEX, destination_index) = F_M_I_SCALE * uz;
    destination.device(M_MXX_INDEX, destination_index) = F_M_II_SCALE * mxx;
    destination.device(M_MXY_INDEX, destination_index) = F_M_IJ_SCALE * mxy;
    destination.device(M_MXZ_INDEX, destination_index) = F_M_IJ_SCALE * mxz;
    destination.device(M_MYY_INDEX, destination_index) = F_M_II_SCALE * myy;
    destination.device(M_MYZ_INDEX, destination_index) = F_M_IJ_SCALE * myz;
    destination.device(M_MZZ_INDEX, destination_index) = F_M_II_SCALE * mzz;
}

__global__ void gpuCollideMoments(LBMState state)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
        return;

    const size_t index = blockedIndex(x, y, z);
    const dfloat rho = RHO_0 + state.device(M_RHO_INDEX, index);
    const dfloat ux = state.device(M_UX_INDEX, index);
    const dfloat uy = state.device(M_UY_INDEX, index);
    const dfloat uz = state.device(M_UZ_INDEX, index);
    dfloat mxx = state.device(M_MXX_INDEX, index);
    dfloat mxy = state.device(M_MXY_INDEX, index);
    dfloat mxz = state.device(M_MXZ_INDEX, index);
    dfloat myy = state.device(M_MYY_INDEX, index);
    dfloat myz = state.device(M_MYZ_INDEX, index);
    dfloat mzz = state.device(M_MZZ_INDEX, index);

    moment_collision(
        rho, ux, uy, uz,
        &mxx, &mxy, &mxz, &myy, &myz, &mzz);

    state.device(M_MXX_INDEX, index) = mxx;
    state.device(M_MXY_INDEX, index) = mxy;
    state.device(M_MXZ_INDEX, index) = mxz;
    state.device(M_MYY_INDEX, index) = myy;
    state.device(M_MYZ_INDEX, index) = myz;
    state.device(M_MZZ_INDEX, index) = mzz;
}

__global__ void gpuStreamMomentsAB(
    LBMState source,
    LBMState destination,
    const unsigned int *node_type,
    const RemotePopulationHalos remote_halos,
    const SubDomainInfo subdomain,
    const int block_z_offset)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int logical_block_z =
        static_cast<int>(blockIdx.z) + block_z_offset;
    const int z = threadIdx.z + blockDim.z * logical_block_z;

    __shared__ dfloat shared_moments[NUMBER_MOMENTS * SHARED_HALO_SIZE];

    const int thread_index = threadIdx.x + BLOCK_NX *(threadIdx.y + BLOCK_NY * threadIdx.z);

    for (int halo_index = thread_index;
         halo_index < SHARED_HALO_SIZE;
         halo_index += BLOCK_LBM_SIZE)
    {
        const int halo_x = halo_index % SHARED_HALO_NX;
        const int halo_y = (halo_index / SHARED_HALO_NX) % SHARED_HALO_NY;
        const int halo_z = halo_index / (SHARED_HALO_NX * SHARED_HALO_NY);

        const int source_x = (static_cast<int>(blockIdx.x) * BLOCK_NX + halo_x - 1 + NX) % NX;
        const int source_y = (static_cast<int>(blockIdx.y) * BLOCK_NY + halo_y - 1 + NY) % NY;
        const int source_z = logical_block_z * BLOCK_NZ + halo_z - 1;
#pragma unroll
        for (int moment = 0; moment < NUMBER_MOMENTS; ++moment)
            shared_moments[moment * SHARED_HALO_SIZE + halo_index] =
                sourceMoment(source, remote_halos, subdomain,
                    moment, source_x, source_y, source_z);
    }

    __syncthreads();

    const size_t destination_index = blockedIndex(x, y, z);
    const unsigned int destination_type = node_type[destination_index];
    dfloat pop[Q];
    const int tile_x = x % AB_TILE_NX;
    const int tile_y = y % AB_TILE_NY;
    const int tile_z = z % AB_TILE_NZ;

#pragma unroll
    for (int i = 0; i < Q; ++i)
    {
        const int velocity_x = static_cast<int>(cx[i]);
        const int velocity_y = static_cast<int>(cy[i]);
        const int velocity_z = static_cast<int>(cz[i]);
        const int unwrapped_source_x = x - velocity_x;
        const int unwrapped_source_y = y - velocity_y;
        const int unwrapped_source_z = z - velocity_z;

        const int tile_source_x = tile_x - velocity_x;
        const int tile_source_y = tile_y - velocity_y;
        const int tile_source_z = tile_z - velocity_z;
        const bool crosses_x_tile = tile_source_x < 0 || tile_source_x >= AB_TILE_NX;
        const bool crosses_y_tile = tile_source_y < 0 || tile_source_y >= AB_TILE_NY;
        const bool crosses_z_tile = tile_source_z < 0 || tile_source_z >= AB_TILE_NZ;

        if (usesInitialWallPopulation(unwrapped_source_x, unwrapped_source_y, unwrapped_source_z, crosses_x_tile, crosses_y_tile, crosses_z_tile, subdomain))
        {
            pop[i] = remote_halos.initial_wall[i];
            continue;
        }

        if (unwrapped_source_z < 0 || unwrapped_source_z >= NZ)
        {
            const dfloat *remote = unwrapped_source_z < 0 ?
                remote_halos.back : remote_halos.front;
            pop[i] = remotePopulation(remote, i,
                (unwrapped_source_x + NX) % NX,
                (unwrapped_source_y + NY) % NY);
            continue;
        }

        const int shared_x = static_cast<int>(threadIdx.x) + 1 - static_cast<int>(cx[i]);
        const int shared_y = static_cast<int>(threadIdx.y) + 1 - static_cast<int>(cy[i]);
        const int shared_z = static_cast<int>(threadIdx.z) + 1 - static_cast<int>(cz[i]);
        const int shared_index = shared_x + SHARED_HALO_NX * (shared_y + SHARED_HALO_NY * shared_z);

        pop[i] = reconstructCollidedPopulation(
            shared_moments, shared_index, i);
    }

    dfloat rho;
    dfloat ux;
    dfloat uy;
    dfloat uz;
    dfloat mxx;
    dfloat mxy;
    dfloat mxz;
    dfloat myy;
    dfloat myz;
    dfloat mzz;

    if (destination_type == BULK)
        eval_moments(
            &rho, &ux, &uy, &uz,
            &mxx, &mxy, &mxz, &myy, &myz, &mzz, pop);
    else
    {
        restorePhysicalPopulations(pop);
        apply_boundaries(
            pop, &rho, &ux, &uy, &uz,
            &mxx, &mxy, &mxz, &myy, &myz, &mzz,
            destination_type);
    }

    ux *= F_M_I_SCALE;
    uy *= F_M_I_SCALE;
    uz *= F_M_I_SCALE;
    mxx *= F_M_II_SCALE;
    mxy *= F_M_IJ_SCALE;
    mxz *= F_M_IJ_SCALE;
    myy *= F_M_II_SCALE;
    myz *= F_M_IJ_SCALE;
    mzz *= F_M_II_SCALE;

    moment_collision(
        rho, ux, uy, uz,
        &mxx, &mxy, &mxz, &myy, &myz, &mzz);

    destination.device(M_RHO_INDEX, destination_index) = rho - RHO_0;
    destination.device(M_UX_INDEX, destination_index) = ux;
    destination.device(M_UY_INDEX, destination_index) = uy;
    destination.device(M_UZ_INDEX, destination_index) = uz;
    destination.device(M_MXX_INDEX, destination_index) = mxx;
    destination.device(M_MXY_INDEX, destination_index) = mxy;
    destination.device(M_MXZ_INDEX, destination_index) = mxz;
    destination.device(M_MYY_INDEX, destination_index) = myy;
    destination.device(M_MYZ_INDEX, destination_index) = myz;
    destination.device(M_MZZ_INDEX, destination_index) = mzz;
}
