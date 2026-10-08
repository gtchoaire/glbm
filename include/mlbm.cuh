#ifndef MLBM_H
#define MLBM_H

#include "globalFunctions.cuh"
#include "var.h"

#include PROPERTIES
#include COLLISION
#include RECONSTRUCTION
#include "../src/colrec/D3Q27/evalMoments.cuh"
#include "../src/colrec/D3Q27/boundary/secondOrder.cuh"

__global__ void gpuMomCollisionStreamAB(
    LBMState source,
    LBMState destination,
    const unsigned int *node_type,
    RemotePopulationHalos remote_halos,
    bool initial_step);

__global__ void gpuInitializeWallPopulations(
    LBMState source,
    dfloat *initial_wall_populations);

#endif
