#ifndef LBM_STATE_CUH
#define LBM_STATE_CUH

#include "globalStructs.cuh"

bool allocateLBMState(LBMState &state);
void freeLBMState(LBMState &state);
bool copyMacroscopicFieldsToHost(LBMState &state);
bool initializeEquilibrium(LBMState &state, dim3 grid, dim3 block);

#endif
