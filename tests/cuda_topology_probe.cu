#include <cuda_runtime.h>

#include <cstdio>

namespace
{
bool check(const cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}
}

int main()
{
    int runtime_version = 0;
    int driver_version = 0;
    int device_count = 0;
    if (!check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion") ||
        !check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion") ||
        !check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount"))
        return 1;

    std::printf("CUDA runtime %d | driver %d | devices %d\n",
                runtime_version, driver_version, device_count);

    for (int device = 0; device < device_count; ++device)
    {
        cudaDeviceProp properties{};
        int mapped_host = 0;
        int unified_addressing = 0;
        char pci_bus_id[32]{};
        if (!check(cudaGetDeviceProperties(&properties, device),
                   "cudaGetDeviceProperties") ||
            !check(cudaDeviceGetAttribute(&mapped_host,
                                          cudaDevAttrCanMapHostMemory,
                                          device),
                   "cudaDevAttrCanMapHostMemory") ||
            !check(cudaDeviceGetAttribute(&unified_addressing,
                                          cudaDevAttrUnifiedAddressing,
                                          device),
                   "cudaDevAttrUnifiedAddressing") ||
            !check(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device),
                   "cudaDeviceGetPCIBusId"))
            return 1;

        std::printf("GPU %d | %s | CC %d.%d | PCI %s | mapped_host %d | UVA %d\n",
                    device, properties.name, properties.major, properties.minor,
                    pci_bus_id, mapped_host, unified_addressing);
    }

    for (int source = 0; source < device_count; ++source)
        for (int destination = 0; destination < device_count; ++destination)
        {
            if (source == destination)
                continue;
            int can_access = 0;
            if (!check(cudaDeviceCanAccessPeer(&can_access, source, destination),
                       "cudaDeviceCanAccessPeer"))
                return 1;
            std::printf("P2P %d -> %d | %s\n", source, destination,
                        can_access ? "available" : "unavailable");
        }

    return 0;
}
