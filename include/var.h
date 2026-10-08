#ifndef __VAR_H
#define __VAR_H

#include <builtin_types.h>
#include <stdint.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <chrono>

#if defined(double_precision)
typedef double dfloat;
#else
typedef float dfloat;
#endif

typedef std::chrono::high_resolution_clock::time_point timestep;

#ifndef GPU_INDEX
#define GPU_INDEX 0
#endif

#define STR_IMPL(A) #A
#define STR(A) STR_IMPL(A)

/* --------------------------  SIMULATION DEFINES -------------------------- */

#ifndef STENCIL
#define STENCIL D3Q27
#endif

#ifndef N_SIZE
#define N_SIZE 64
#endif

#ifndef RE_NUMBER
#define RE_NUMBER 3200.0f
#endif

#ifndef U_MAX_VALUE
#define U_MAX_VALUE 0.05f
#endif

#ifndef ORDER
#define ORDER 2
#endif

#ifndef SAR
#define SAR 1
#endif

#ifndef NUM_GPUS
#define NUM_GPUS 1
#endif

// clang-format off

#define PROPERTIES STR(../src/colrec/STENCIL/properties.cuh)
#define RECONSTRUCTION STR(../src/colrec/STENCIL/reconstruction.cuh)
#define COLLISION STR(../src/colrec/collision.cuh)

// clang-format on

#include "definitions.h"
#endif
