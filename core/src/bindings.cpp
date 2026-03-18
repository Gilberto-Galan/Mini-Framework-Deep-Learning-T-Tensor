#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // Necesario para std::shared_ptr y std::vector
#include <pybind11/operators.h> 
#include "tensor.hpp"
#include "layers.hpp"
#include "optimizer.hpp"
#include "losses.hpp"
#include "dataloader.hpp"
#include "device_manager.hpp"

namespace py = pybind11;
using namespace ttensor;

PYBIND11_MODULE(_ttensor, m) {
    m.doc() = "T-Tensor: Motor con Autograd y optimizacion CUDA agnostica al hardware";

    // 1. Enum de Device
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::class_<DeviceInfo>(m, "DeviceInfo")
        .def_readonly("id", &DeviceInfo::id)
        .def_readonly("name", &DeviceInfo::name)
        .def_readonly("total_memory", &DeviceInfo::total_memory)
        .def_readonly("free_memory", &DeviceInfo::free_memory)
        .def_readonly("compute_major", &DeviceInfo::compute_major)
        .def_readonly("compute_minor", &DeviceInfo::compute_minor)
        .def_readonly("max_threads_per_block", &DeviceInfo::max_threads_per_block)
        .def_readonly("multi_processor_count", &DeviceInfo::multi_processor_count)
        .def_readonly("warp_size", &DeviceInfo::warp_size)
        .def_readonly("shared_memory_per_block", &DeviceInfo::shared_memory_per_block)
        .def_readonly("max_threads_per_multiprocessor", &DeviceInfo::max_threads_per_multiprocessor)
        .def_readonly("memory_bus_width", &DeviceInfo::memory_bus_width)
        .def_readonly("clock_rate_khz", &DeviceInfo::clock_rate_khz)
        .def_readonly("max_grid_size_x", &DeviceInfo::max_grid_size_x)
        .def_readonly("max_grid_size_y", &DeviceInfo::max_grid_size_y)
        .def_readonly("max_grid_size_z", &DeviceInfo::max_grid_size_z)
        .def_readonly("concurrent_kernels", &DeviceInfo::concurrent_kernels)
        .def_readonly("supports_unified_memory", &DeviceInfo::supports_unified_memory)
        .def_readonly("supports_concurrent_managed_access", &DeviceInfo::supports_concurrent_managed_access)
        .def_readonly("supports_unified_addressing", &DeviceInfo::supports_unified_addressing);

    py::class_<DeviceManager>(m, "DeviceManager")
        .def_static("initialize", &DeviceManager::initialize, py::arg("preferred_device_id") = 0)
        .def_static("set_device", &DeviceManager::set_device, py::arg("device_id"))
        .def_static("is_initialized", &DeviceManager::is_initialized)
        .def_static("device_count", &DeviceManager::device_count)
        .def_static("active_device_id", &DeviceManager::active_device_id)
        .def_static("available_devices", &DeviceManager::available_devices)
        .def_static("get_info", &DeviceManager::get_info, py::return_value_policy::reference)
        .def_static("update_memory_info", &DeviceManager::update_memory_info)
        .def_static("set_prefer_managed_memory", &DeviceManager::set_prefer_managed_memory, py::arg("enabled"))
        .def_static("should_use_managed_memory", &DeviceManager::should_use_managed_memory)
        .def_static("has_enough_vram", &DeviceManager::has_enough_vram, py::arg("required_bytes"))
        .def_static("print_report", &DeviceManager::print_report);

    // 2. Clase Tensor (con soporte para Shared Pointers y Autograd)
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<int, int, Device, bool>(), 
             py::arg("rows"), py::arg("cols"), py::arg("device") = Device::GPU, py::arg("requires_grad") = false)
           .def_static("from_list", &Tensor::from_vector,
               py::arg("values"), py::arg("rows"), py::arg("cols"), py::arg("device") = Device::GPU, py::arg("requires_grad") = false)
        
        // Autograd y Memoria
           .def("backward", &Tensor::backward, py::arg("grad_output") = nullptr, "Inicia la propagación de gradientes desde este tensor")
        .def("zero_grad", &Tensor::zero_grad, "Reinicia el gradiente acumulado a cero")
        .def("to_gpu", &Tensor::to_gpu)
        .def("to_cpu", &Tensor::to_cpu)
        
        // Propiedades
        .def_readwrite("grad", &Tensor::grad)
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def_property_readonly("shape", [](const Tensor &t) {
            return py::make_tuple(t.rows, t.cols);
        })

        // Inicialización y Utilidad
        .def("fill", &Tensor::fill)
        .def("fill_random", &Tensor::fill_random, py::arg("low") = -1.0f, py::arg("high") = 1.0f)
        .def("set_data", &Tensor::set_data, py::arg("values"))
        .def("tolist", &Tensor::to_vector)
        .def("item", &Tensor::item)
        .def("print", &Tensor::print, py::arg("limit") = 10)

        // En la clase Tensor
        .def("accuracy", &Tensor::accuracy)

        // Operaciones Estáticas (Factory methods para el Grafo)
        .def_static("matmul", &Tensor::matmul)
        .def_static("add", &Tensor::add)
        .def_static("sub", &Tensor::sub)
        .def_static("mul", &Tensor::multiply)
        .def_static("relu", &Tensor::relu)
        .def_static("sigmoid", &Tensor::sigmoid)
        .def_static("softmax", [](std::shared_ptr<Tensor> x) { return Tensor::softmax(x); })
        .def_static("log", &Tensor::log)
        .def_static("exp", &Tensor::exp)
        .def_static("sum", &Tensor::sum, py::arg("input"), py::arg("axis") = -1)
        .def_static("mean", &Tensor::mean, py::arg("input"), py::arg("axis") = -1)
        .def_static("reshape", &Tensor::reshape, py::arg("input"), py::arg("new_rows"), py::arg("new_cols"))
        
        // Sobrecarga de Operadores para permitir expresiones mas naturales
        .def("__add__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
            return Tensor::add(a, b);
        }, py::is_operator())
        .def("__sub__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
            return Tensor::sub(a, b);
        }, py::is_operator())
        .def("__mul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
            return Tensor::multiply(a, b);
        }, py::is_operator())
        .def("__matmul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
            return Tensor::matmul(a, b);
        }, py::is_operator());

    // 3. Jerarquía de Módulos (Estilo nn.Module)
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("__call__", &Module::forward)
        .def("parameters", &Module::parameters)
        .def("zero_grad", &Module::zero_grad)
        .def("save", &Module::save)
        .def("load", &Module::load);

    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int, Device>(), py::arg("in_features"), py::arg("out_features"), py::arg("device") = Device::GPU)
        .def("__call__", &Linear::forward)
        .def("forward", &Linear::forward);

    // 4. Optimizadores
    py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer")
        .def("step", &Optimizer::step)
        .def("zero_grad", &Optimizer::zero_grad);

    py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD")
        .def(py::init<std::vector<std::shared_ptr<Tensor>>, float>(), py::arg("params"), py::arg("lr"));

    py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(m, "Adam")
        .def(py::init<std::vector<std::shared_ptr<Tensor>>, float, float, float, float>(),
             py::arg("params"), py::arg("lr"), py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f);

    // 5. Funciones de Pérdida
    py::class_<MSELoss, std::shared_ptr<MSELoss>>(m, "MSELoss")
        .def(py::init<>())
        .def("forward",
             py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&MSELoss::forward),
             py::arg("input"), py::arg("target"));

     py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>())
        .def("__call__", &ReLU::forward)
        .def("forward", &ReLU::forward);

    py::class_<Sigmoid, Module, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>())
        .def("__call__", &Sigmoid::forward)
        .def("forward", &Sigmoid::forward);

    py::class_<Softmax, Module, std::shared_ptr<Softmax>>(m, "Softmax")
        .def(py::init<>())
        .def("__call__", &Softmax::forward)
        .def("forward", &Softmax::forward);

    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(m, "Sequential")
        .def(py::init<std::vector<std::shared_ptr<Module>>>())
        .def("__call__", &Sequential::forward)
        .def("forward", &Sequential::forward);   

    // Exponer la interfaz base de Dataset
    py::class_<Dataset, std::shared_ptr<Dataset>>(m, "Dataset");

    // Exponer CSVDataset
    py::class_<CSVDataset, Dataset, std::shared_ptr<CSVDataset>>(m, "CSVDataset")
        .def(py::init<const std::string&, int, Device>(), 
             py::arg("path"), py::arg("label_cols"), py::arg("device") = Device::GPU);

    // Exponer DataLoader
    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<std::shared_ptr<Dataset>, size_t, bool>(),
             py::arg("dataset"), py::arg("batch_size"), py::arg("shuffle") = true)
        .def("reset", &DataLoader::reset)
        .def("has_next", &DataLoader::has_next)
        .def("next_batch", &DataLoader::next_batch);

    py::class_<CrossEntropyLoss, std::shared_ptr<CrossEntropyLoss>>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward",
             py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&CrossEntropyLoss::forward),
             py::arg("input"), py::arg("target"));
}