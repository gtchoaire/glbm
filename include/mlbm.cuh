#ifndef MLBM_H
#define MLBM_H

#include "globalFunctions.cuh"
#include "var.h"

#include PROPERTIES
#include COLLISION
#include RECONSTRUCTION
#include "../src/colrec/D3Q27/evalMoments.cuh"
#include "../src/colrec/D3Q27/boundary/secondOrder.cuh"

// Builds the first post-collision AB state. This bootstrap path performs the
// initial pull streaming and applies the physical boundary conditions.
__global__ void gpuBootstrapABState(
    LBMState source,
    LBMState destination,
    const unsigned int *node_type,
    RemotePopulationHalos remote_halos);

__global__ void gpuCollideMoments(LBMState state);

// Streams from a state whose non-conserved moments are already post-collision
// and stores the destination in the same post-collision representation.
__global__ void gpuStreamMomentsAB(
    LBMState source,
    LBMState destination,
    const unsigned int *node_type,
    RemotePopulationHalos remote_halos);

__global__ void gpuInitializeWallPopulations(
    LBMState source,
    dfloat *initial_wall_populations);

#endif
