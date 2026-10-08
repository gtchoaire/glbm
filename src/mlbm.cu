#include "../include/mlbm.cuh"

namespace
{
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
    dfloat *populations,
    const bool collide_source_moments = true)
{
    const dfloat rho = RHO_0 + stored_rho;
    if (collide_source_moments)
        moment_collision(
            rho, ux, uy, uz,
            &mxx, &mxy, &mxz, &myy, &myz, &mzz);

    pop_reconstruction(
        rho, ux, uy, uz,
        mxx, mxy, mxz, myy, myz, mzz,
        populations);
}

__device__ inline dfloat reconstructPostCollisionPopulation(
    const LBMState &source,
    const size_t source_index,
    const int population_index,
    const bool collide_source_moments)
{
    const dfloat rho = RHO_0 + source.d_rho[source_index];
    const dfloat ux = source.d_ux[source_index];
    const dfloat uy = source.d_uy[source_index];
    const dfloat uz = source.d_uz[source_index];
    dfloat mxx = source.d_mxx[source_index];
    dfloat mxy = source.d_mxy[source_index];
    dfloat mxz = source.d_mxz[source_index];
    dfloat myy = source.d_myy[source_index];
    dfloat myz = source.d_myz[source_index];
    dfloat mzz = source.d_mzz[source_index];

    if (collide_source_moments)
        moment_collision(
            rho, ux, uy, uz,
            &mxx, &mxy, &mxz, &myy, &myz, &mzz);

    return reconstruct_population(
        population_index, rho, ux, uy, uz,
        mxx, mxy, mxz, myy, myz, mzz);
}

__device__ inline dfloat reconstructCollidedPopulation(
    const LBMState &source,
    const size_t source_index,
    const int population_index)
{
    return reconstruct_population(
        population_index,
        RHO_0 + source.d_rho[source_index],
        source.d_ux[source_index],
        source.d_uy[source_index],
        source.d_uz[source_index],
        source.d_mxx[source_index],
        source.d_mxy[source_index],
        source.d_mxz[source_index],
        source.d_myy[source_index],
        source.d_myz[source_index],
        source.d_mzz[source_index]);
}
} // namespace

__global__ void gpuInitializeWallPopulations(
    LBMState source,
    dfloat *initial_wall_populations)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    reconstructPostCollisionPopulationsFromMoments(
        source.d_rho[0], source.d_ux[0], source.d_uy[0], source.d_uz[0],
        source.d_mxx[0], source.d_mxy[0], source.d_mxz[0],
        source.d_myy[0], source.d_myz[0], source.d_mzz[0],
        initial_wall_populations, false);
}

__global__ void gpuMomCollisionStreamAB(
    LBMState source,
    LBMState destination,
    const unsigned int *node_type,
    const RemotePopulationHalos remote_halos,
    const bool initial_step)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
        return;

    const size_t destination_index = blockedIndex(x, y, z);
    const unsigned int destination_type = node_type[destination_index];

    dfloat pop[Q];

#pragma unroll
    for (int i = 0; i < Q; ++i)
    {
        const int unwrapped_source_x = x - static_cast<int>(cx[i]);
        const int unwrapped_source_y = y - static_cast<int>(cy[i]);
        const int unwrapped_source_z = z - static_cast<int>(cz[i]);

        const int source_x = (unwrapped_source_x + NX) % NX;
        const int source_y = (unwrapped_source_y + NY) % NY;
        const int source_z = (unwrapped_source_z + NZ) % NZ;

        const bool crosses_x_tile =
            source_x / AB_TILE_NX != x / AB_TILE_NX;
        const bool crosses_y_tile =
            source_y / AB_TILE_NY != y / AB_TILE_NY;
        const bool crosses_z_tile =
            source_z / AB_TILE_NZ != z / AB_TILE_NZ;

        const bool crosses_tile =
            crosses_x_tile || crosses_y_tile || crosses_z_tile;

        if (crosses_tile)
        {
            const bool crosses_physical_x_wall =
                unwrapped_source_x < 0 || unwrapped_source_x >= NX;
            const bool crosses_physical_y_wall =
                unwrapped_source_y < 0 || unwrapped_source_y >= NY;
            const bool crosses_physical_z_wall =
                unwrapped_source_z < 0 || unwrapped_source_z >= NZ;

            const bool reads_initialized_wall_population =
                (crosses_z_tile && crosses_physical_z_wall) ||
                (!crosses_z_tile && crosses_y_tile &&
                 crosses_physical_y_wall) ||
                (!crosses_z_tile && !crosses_y_tile && crosses_x_tile &&
                 crosses_physical_x_wall);

            if (reads_initialized_wall_population)
            {
                pop[i] = remote_halos.initial_wall[i];
                continue;
            }
        }

        pop[i] = reconstructPostCollisionPopulation(
            source, blockedIndex(source_x, source_y, source_z),
            i, !(initial_step && crosses_tile));
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

    destination.d_rho[destination_index] = rho - RHO_0;
    destination.d_ux[destination_index] = F_M_I_SCALE * ux;
    destination.d_uy[destination_index] = F_M_I_SCALE * uy;
    destination.d_uz[destination_index] = F_M_I_SCALE * uz;
    destination.d_mxx[destination_index] = F_M_II_SCALE * mxx;
    destination.d_mxy[destination_index] = F_M_IJ_SCALE * mxy;
    destination.d_mxz[destination_index] = F_M_IJ_SCALE * mxz;
    destination.d_myy[destination_index] = F_M_II_SCALE * myy;
    destination.d_myz[destination_index] = F_M_IJ_SCALE * myz;
    destination.d_mzz[destination_index] = F_M_II_SCALE * mzz;
}

__global__ void gpuCollideMoments(LBMState state)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
        return;

    const size_t index = blockedIndex(x, y, z);
    const dfloat rho = RHO_0 + state.d_rho[index];
    const dfloat ux = state.d_ux[index];
    const dfloat uy = state.d_uy[index];
    const dfloat uz = state.d_uz[index];
    dfloat mxx = state.d_mxx[index];
    dfloat mxy = state.d_mxy[index];
    dfloat mxz = state.d_mxz[index];
    dfloat myy = state.d_myy[index];
    dfloat myz = state.d_myz[index];
    dfloat mzz = state.d_mzz[index];

    moment_collision(
        rho, ux, uy, uz,
        &mxx, &mxy, &mxz, &myy, &myz, &mzz);

    state.d_mxx[index] = mxx;
    state.d_mxy[index] = mxy;
    state.d_mxz[index] = mxz;
    state.d_myy[index] = myy;
    state.d_myz[index] = myz;
    state.d_mzz[index] = mzz;
}

__global__ void gpuStreamMomentsAB(
    LBMState source,
    LBMState destination,
    const unsigned int *node_type,
    const RemotePopulationHalos remote_halos)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
        return;

    const size_t destination_index = blockedIndex(x, y, z);
    const unsigned int destination_type = node_type[destination_index];
    dfloat pop[Q];

#pragma unroll
    for (int i = 0; i < Q; ++i)
    {
        const int unwrapped_source_x = x - static_cast<int>(cx[i]);
        const int unwrapped_source_y = y - static_cast<int>(cy[i]);
        const int unwrapped_source_z = z - static_cast<int>(cz[i]);

        const int source_x = (unwrapped_source_x + NX) % NX;
        const int source_y = (unwrapped_source_y + NY) % NY;
        const int source_z = (unwrapped_source_z + NZ) % NZ;

        const bool crosses_x_tile =
            source_x / AB_TILE_NX != x / AB_TILE_NX;
        const bool crosses_y_tile =
            source_y / AB_TILE_NY != y / AB_TILE_NY;
        const bool crosses_z_tile =
            source_z / AB_TILE_NZ != z / AB_TILE_NZ;
        const bool crosses_tile =
            crosses_x_tile || crosses_y_tile || crosses_z_tile;

        if (crosses_tile)
        {
            const bool crosses_physical_x_wall =
                unwrapped_source_x < 0 || unwrapped_source_x >= NX;
            const bool crosses_physical_y_wall =
                unwrapped_source_y < 0 || unwrapped_source_y >= NY;
            const bool crosses_physical_z_wall =
                unwrapped_source_z < 0 || unwrapped_source_z >= NZ;

            const bool reads_initialized_wall_population =
                (crosses_z_tile && crosses_physical_z_wall) ||
                (!crosses_z_tile && crosses_y_tile &&
                 crosses_physical_y_wall) ||
                (!crosses_z_tile && !crosses_y_tile && crosses_x_tile &&
                 crosses_physical_x_wall);

            if (reads_initialized_wall_population)
            {
                pop[i] = remote_halos.initial_wall[i];
                continue;
            }
        }

        pop[i] = reconstructCollidedPopulation(
            source, blockedIndex(source_x, source_y, source_z), i);
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

    destination.d_rho[destination_index] = rho - RHO_0;
    destination.d_ux[destination_index] = ux;
    destination.d_uy[destination_index] = uy;
    destination.d_uz[destination_index] = uz;
    destination.d_mxx[destination_index] = mxx;
    destination.d_mxy[destination_index] = mxy;
    destination.d_mxz[destination_index] = mxz;
    destination.d_myy[destination_index] = myy;
    destination.d_myz[destination_index] = myz;
    destination.d_mzz[destination_index] = mzz;
}
