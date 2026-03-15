#ifndef TTENSOR_DEVICE_MANAGER_HPP
#define TTENSOR_DEVICE_MANAGER_HPP

#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include <vector>

namespace ttensor {

struct DeviceInfo {
    int id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_major;
    int compute_minor;
    int max_threads_per_block;
    int multi_processor_count;
    int warp_size;
    int shared_memory_per_block;
    int max_threads_per_multiprocessor;
    int memory_bus_width;
    int clock_rate_khz;
    int max_grid_size_x;
    int max_grid_size_y;
    int max_grid_size_z;
    bool concurrent_kernels;
    bool supports_unified_memory;
    bool supports_concurrent_managed_access;
    bool supports_unified_addressing;
};

class DeviceManager {
private:
    static std::vector<DeviceInfo> devices;
    static DeviceInfo current_device_info;
    static int current_device_id;
    static bool initialized;
    static bool prefer_managed_memory;

    static DeviceInfo query_device_info(int device_id);
    static void refresh_memory_info();

public:
    // Detecta GPUs CUDA, selecciona dispositivo y cachea capacidades.
    static void initialize(int preferred_device_id = 0);

    // Fuerza la selección de un dispositivo específico.
    static void set_device(int device_id);

    // Actualiza la memoria libre/total del dispositivo actual.
    static void update_memory_info();

    // Estado global de runtime.
    static bool is_initialized();
    static int device_count();
    static int active_device_id();
    static std::vector<DeviceInfo> available_devices();

    // Getters para usar en kernels y capas de alto nivel.
    static const DeviceInfo& get_info();
    static void print_report();

    // Utilidades para launch dinámico.
    static int get_optimal_threads_1d(int total_elements, int max_threads_hint = 256);
    static int get_optimal_blocks(int total_elements, int threads_per_block = 256);
    static dim3 get_optimal_block_2d();
    static dim3 get_grid_2d(int rows, int cols, dim3 block);

    // Runtime memory policy.
    static void set_prefer_managed_memory(bool enabled);
    static bool should_use_managed_memory();
    
    // Verifica si hay suficiente VRAM antes de una operación grande.
    static bool has_enough_vram(size_t required_bytes);
};

} // namespace ttensor

#endif