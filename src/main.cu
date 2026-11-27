#include "../include/simulation.cuh"

#include <cstdio>
#include <cstdlib>
#include <exception>

int main()
{
    try
    {
        runSimulation();
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "\nError: %s\n", error.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
