#ifndef OUTPUT_H
#define OUTPUT_H

#include <vector>

#include "globalStructs.cuh"

// States must be ordered by increasing global Z subdomain.
bool writeVti(const std::vector<LBMState *> &states, int step);

#endif
