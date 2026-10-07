#ifndef INDEX_H
#define INDEX_H

constexpr int M_RHO_INDEX = 0;
constexpr int M_UX_INDEX = 1;
constexpr int M_UY_INDEX = 2;
constexpr int M_UZ_INDEX = 3;
constexpr int M_MXX_INDEX = 4;
constexpr int M_MXY_INDEX = 5;
constexpr int M_MXZ_INDEX = 6;
constexpr int M_MYY_INDEX = 7;
constexpr int M_MYZ_INDEX = 8;
constexpr int M_MZZ_INDEX = 9;

#ifdef M_OFFSET
#undef M_OFFSET
#endif
#define M_OFFSET M_MZZ_INDEX

const size_t NUMBER_MOMENTS = M_OFFSET + 1;

#endif