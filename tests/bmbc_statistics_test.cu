#include "bmbc.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace
{
bool close(const double actual, const double expected)
{
    const double scale = std::max(1.0, std::abs(expected));
    return std::abs(actual - expected) <= 1.0e-12 * scale;
}
}

int main()
{
    const BMBCEstimate direct = estimateBMBC({1.0, 2.0, 4.0, 8.0});
    const BMBCEstimate grouped = estimateBMBC({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

    const bool ok =
        close(direct.mean, 3.75) && direct.group == 1 && direct.groups == 4 &&
        close(direct.error, std::sqrt(39.625 / 6.0)) &&
        close(grouped.mean, 0.5) && grouped.group == 2 && grouped.groups == 12 &&
        close(grouped.error, std::sqrt(0.05)) &&
        usableBMBCBlocks(59) == 59 && usableBMBCBlocks(1735) == 1680;

    if (!ok)
        std::cerr << "Unexpected BMBC result: mean=" << direct.mean
                  << " error=" << direct.error
                  << " group=" << direct.group
                  << " groups=" << direct.groups
                  << " | group=" << grouped.group
                  << " groups=" << grouped.groups << '\n';

    return ok ? 0 : 1;
}
