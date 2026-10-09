#ifndef LBM_STATE_CUH
#define LBM_STATE_CUH

#include "globalStructs.cuh"

void allocateLBMState(LBMState &state);
void freeLBMState(LBMState &state);
void copyMacroscopicFieldsToHost(LBMState &state);
void initializeEquilibrium(LBMState &state, dim3 grid, dim3 block);

#endif
