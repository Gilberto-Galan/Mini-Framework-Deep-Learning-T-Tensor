#include "device_manager.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace ttensor {

std::vector<DeviceInfo> DeviceManager::devices = {};
DeviceInfo DeviceManager::current_device_info = {};
int DeviceManager::current_device_id = -1;
bool DeviceManager::initialized = false;
bool DeviceManager::prefer_managed_memory = false;

DeviceInfo DeviceManager::query_device_info(int device_id) {
    cudaDeviceProp prop;
    cudaError_t status = cudaGetDeviceProperties(&prop, device_id);
    if (status != cudaSuccess) {
        throw std::runtime_error("[DeviceManager] No se pudo obtener propiedades del dispositivo CUDA.");
    }

    DeviceInfo info{};
    info.id = device_id;
    info.name = prop.name;
    info.total_memory = prop.totalGlobalMem;
    info.free_memory = 0;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.multi_processor_count = prop.multiProcessorCount;
    info.warp_size = prop.warpSize;
    info.shared_memory_per_block = static_cast<int>(prop.sharedMemPerBlock);
    info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    info.memory_bus_width = prop.memoryBusWidth;
    info.clock_rate_khz = prop.clockRate;
    info.max_grid_size_x = prop.maxGridSize[0];
    info.max_grid_size_y = prop.maxGridSize[1];
    info.max_grid_size_z = prop.maxGridSize[2];
    info.concurrent_kernels = prop.concurrentKernels != 0;
    info.supports_unified_memory = prop.managedMemory != 0;
    info.supports_concurrent_managed_access = prop.concurrentManagedAccess != 0;
    info.supports_unified_addressing = prop.unifiedAddressing != 0;
    return info;
}

void DeviceManager::refresh_memory_info() {
    if (current_device_id < 0) {
        return;
    }

    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);
    if (status == cudaSuccess) {
        current_device_info.free_memory = free_mem;
        current_device_info.total_memory = total_mem;
    }
}

void DeviceManager::initialize(int preferred_device_id) {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess || device_count == 0) {
        throw std::runtime_error("No se detectaron GPUs compatibles con CUDA.");
    }

    devices.clear();
    devices.reserve(static_cast<size_t>(device_count));
    for (int id = 0; id < device_count; ++id) {
        devices.push_back(query_device_info(id));
    }

    int device_id = preferred_device_id;
    if (device_id < 0 || device_id >= device_count) {
        device_id = 0;
    }

    set_device(device_id);
    initialized = true;
}

void DeviceManager::set_device(int device_id) {
    if (!initialized) {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            throw std::runtime_error("[DeviceManager] No se detectaron GPUs compatibles con CUDA.");
        }
        devices.clear();
        devices.reserve(static_cast<size_t>(device_count));
        for (int id = 0; id < device_count; ++id) {
            devices.push_back(query_device_info(id));
        }
    }

    if (device_id < 0 || device_id >= static_cast<int>(devices.size())) {
        throw std::runtime_error("[DeviceManager] device_id fuera de rango.");
    }

    cudaError_t set_status = cudaSetDevice(device_id);
    if (set_status != cudaSuccess) {
        throw std::runtime_error("[DeviceManager] No se pudo activar el dispositivo CUDA solicitado.");
    }

    current_device_id = device_id;
    current_device_info = devices[static_cast<size_t>(device_id)];
    refresh_memory_info();
    initialized = true;
}

void DeviceManager::update_memory_info() {
    if (!initialized) {
        initialize();
    }
    refresh_memory_info();
}

bool DeviceManager::is_initialized() {
    return initialized;
}

int DeviceManager::device_count() {
    if (!initialized) {
        initialize();
    }
    return static_cast<int>(devices.size());
}

int DeviceManager::active_device_id() {
    if (!initialized) {
        initialize();
    }
    return current_device_id;
}

std::vector<DeviceInfo> DeviceManager::available_devices() {
    if (!initialized) {
        initialize();
    }
    return devices;
}

const DeviceInfo& DeviceManager::get_info() {
    if (!initialized) {
        initialize();
    }
    refresh_memory_info();
    return current_device_info;
}

void DeviceManager::print_report() {
    const auto& info = get_info();
    std::cout << "--- T-Tensor Hardware Report ---" << std::endl;
    std::cout << "Dispositivo activo: " << info.id << std::endl;
    std::cout << "GPU: " << info.name << std::endl;
    std::cout << "Compute Capability: " << info.compute_major << "." << info.compute_minor << std::endl;
    std::cout << "VRAM Total: " << info.total_memory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "VRAM Libre: " << info.free_memory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Multiprocesadores: " << info.multi_processor_count << std::endl;
    std::cout << "Max Threads/Block: " << info.max_threads_per_block << std::endl;
    std::cout << "Shared Mem/Block: " << info.shared_memory_per_block / 1024 << " KB" << std::endl;
    std::cout << "Grid Max (x,y,z): " << info.max_grid_size_x << ", " << info.max_grid_size_y << ", " << info.max_grid_size_z << std::endl;
    std::cout << "Unified Memory: " << (info.supports_unified_memory ? "si" : "no") << std::endl;
    std::cout << "Managed Access Concurrente: " << (info.supports_concurrent_managed_access ? "si" : "no") << std::endl;
    std::cout << "--------------------------------" << std::endl;
}

int DeviceManager::get_optimal_threads_1d(int total_elements, int max_threads_hint) {
    if (!initialized) {
        initialize();
    }

    int hint = std::max(32, max_threads_hint);
    int capped = std::min(hint, current_device_info.max_threads_per_block);
    if (current_device_info.compute_major <= 6) {
        capped = std::min(capped, 256);
    }
    if (total_elements > 0 && total_elements < capped) {
        int power_two = 32;
        while (power_two < total_elements && power_two < capped) {
            power_two <<= 1;
        }
        capped = std::max(32, std::min(power_two, capped));
    }
    return capped;
}

int DeviceManager::get_optimal_blocks(int total_elements, int threads_per_block) {
    if (threads_per_block <= 0) {
        threads_per_block = get_optimal_threads_1d(total_elements);
    }
    threads_per_block = std::max(1, threads_per_block);
    return (total_elements + threads_per_block - 1) / threads_per_block;
}

dim3 DeviceManager::get_optimal_block_2d() {
    if (!initialized) {
        initialize();
    }

    int side = 16;
    if (current_device_info.max_threads_per_block < 256) {
        side = 8;
    } else if (current_device_info.compute_major >= 9) {
        side = 32;
    }

    int threads = side * side;
    while (threads > current_device_info.max_threads_per_block && side > 8) {
        side /= 2;
        threads = side * side;
    }
    return dim3(static_cast<unsigned int>(side), static_cast<unsigned int>(side));
}

dim3 DeviceManager::get_grid_2d(int rows, int cols, dim3 block) {
    const unsigned int bx = (block.x == 0U) ? 16U : block.x;
    const unsigned int by = (block.y == 0U) ? 16U : block.y;
    const unsigned int gx = static_cast<unsigned int>((cols + static_cast<int>(bx) - 1) / static_cast<int>(bx));
    const unsigned int gy = static_cast<unsigned int>((rows + static_cast<int>(by) - 1) / static_cast<int>(by));
    return dim3(gx, gy, 1U);
}

void DeviceManager::set_prefer_managed_memory(bool enabled) {
    prefer_managed_memory = enabled;
}

bool DeviceManager::should_use_managed_memory() {
    if (!initialized) {
        initialize();
    }
    return prefer_managed_memory && current_device_info.supports_unified_memory;
}

bool DeviceManager::has_enough_vram(size_t required_bytes) {
    if (!initialized) {
        initialize();
    }
    refresh_memory_info();
    return current_device_info.free_memory > required_bytes;
}

} // namespace ttensor