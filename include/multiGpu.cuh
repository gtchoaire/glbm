#ifndef MULTI_GPU_CUH
#define MULTI_GPU_CUH

#include "globalStructs.cuh"

// Validates the devices selected by GPU_INDEX and NUM_GPUS.
bool validateMultiGpuTopology();

// The cubic cavity is decomposed into contiguous slabs along Z.  X and Y
// remain local and complete on every GPU.
SubDomainInfo makeSubDomainInfo(int subdomain_id);

void allocateMomentHalos(DeviceDomain &domain);
void beginMomentHaloExchange(DeviceDomain &back, DeviceDomain &front,
                             const LBMState &back_state,
                             const LBMState &front_state);
void completeMomentHaloExchange(DeviceDomain &back, DeviceDomain &front);
void freeMomentHalos(DeviceDomain &domain);

#endif
