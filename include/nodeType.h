#ifndef NODE_TYPE_MAP_H
#define NODE_TYPE_MAP_H

#include <builtin_types.h>
#include <stdint.h>
#include "definitions.h"

#include "globalStructs.cuh"

// DIRECTION DEFINES 00000000

inline constexpr unsigned int BULK = 0b00000000000000000000000000000000;
// FACE
inline constexpr unsigned int NORTH = 0b00000000000000000000000011001100; // y=NY
inline constexpr unsigned int SOUTH = 0b00000000000000000000000000110011; // y=0
inline constexpr unsigned int WEST = 0b00000000000000000000000001010101;  // x=0
inline constexpr unsigned int EAST = 0b00000000000000000000000010101010;  // x=NX
inline constexpr unsigned int FRONT = 0b00000000000000000000000011110000; // z=NZ
inline constexpr unsigned int BACK = 0b00000000000000000000000000001111;  // z=0
// EDGE
inline constexpr unsigned int NORTH_WEST = 0b00000000000000000000000011011101;
inline constexpr unsigned int NORTH_EAST = 0b00000000000000000000000011101110;
inline constexpr unsigned int NORTH_FRONT = 0b00000000000000000000000011111100;
inline constexpr unsigned int NORTH_BACK = 0b00000000000000000000000011001111;
inline constexpr unsigned int SOUTH_WEST = 0b00000000000000000000000001110111;
inline constexpr unsigned int SOUTH_EAST = 0b00000000000000000000000010111011;
inline constexpr unsigned int SOUTH_FRONT = 0b00000000000000000000000011110011;
inline constexpr unsigned int SOUTH_BACK = 0b00000000000000000000000000111111;
inline constexpr unsigned int WEST_FRONT = 0b00000000000000000000000011110101;
inline constexpr unsigned int WEST_BACK = 0b00000000000000000000000001011111;
inline constexpr unsigned int EAST_FRONT = 0b00000000000000000000000011111010;
inline constexpr unsigned int EAST_BACK = 0b00000000000000000000000010101111;
// CORNER
inline constexpr unsigned int NORTH_WEST_FRONT = 0b00000000000000000000000011111101;
inline constexpr unsigned int NORTH_WEST_BACK = 0b00000000000000000000000011011111;
inline constexpr unsigned int NORTH_EAST_FRONT = 0b00000000000000000000000011111110;
inline constexpr unsigned int NORTH_EAST_BACK = 0b00000000000000000000000011101111;
inline constexpr unsigned int SOUTH_WEST_FRONT = 0b00000000000000000000000011110111;
inline constexpr unsigned int SOUTH_WEST_BACK = 0b00000000000000000000000001111111;
inline constexpr unsigned int SOUTH_EAST_FRONT = 0b00000000000000000000000011111011;
inline constexpr unsigned int SOUTH_EAST_BACK = 0b00000000000000000000000010111111;

inline constexpr unsigned int SOLID_NODE = 0b00000000000000000000000011111111;

inline constexpr unsigned int MISSING_DEFINITION = 0b11111111111111111111111111111111;

bool initializeNodeTypes(DeviceDomain &domain);
void freeNodeTypes(DeviceDomain &domain);

#endif // !__NODE_TYPE_MAP_H
