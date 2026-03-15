#include "dataloader.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ttensor {

CSVDataset::CSVDataset(const std::string& path, int label_cols, Device dev)
    : device(dev) {
    if (label_cols <= 0) {
        throw std::runtime_error("[CSVDataset] label_cols debe ser mayor que 0.");
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("[CSVDataset] No se pudo abrir el archivo: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<float> values;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                values.push_back(std::stof(cell));
            }
        }

        if (values.size() <= static_cast<size_t>(label_cols)) {
            throw std::runtime_error("[CSVDataset] Fila invalida: columnas insuficientes.");
        }

        const size_t split = values.size() - static_cast<size_t>(label_cols);
        data.emplace_back(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(split));
        targets.emplace_back(values.begin() + static_cast<std::ptrdiff_t>(split), values.end());
    }
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DataLoader::next_batch() {
    size_t remaining = dataset->size() - current_index;
    size_t actual_batch_size = std::min(batch_size, remaining);
    if (actual_batch_size == 0) {
        throw std::runtime_error("[DataLoader] No hay mas lotes disponibles.");
    }

    // Obtenemos el primer elemento para conocer las dimensiones
    auto first_item = dataset->get_item(indices[current_index]);
    int input_dim = first_item.first->cols;
    int target_dim = first_item.second->cols;
    Device dev = first_item.first->device;

    // Creamos los tensores contenedores para el batch
    auto batch_inputs = std::make_shared<Tensor>(static_cast<int>(actual_batch_size), input_dim, dev);
    auto batch_targets = std::make_shared<Tensor>(static_cast<int>(actual_batch_size), target_dim, dev);

    // Llenamos los tensores del batch (esto se puede optimizar con hilos en el futuro)
    for (size_t i = 0; i < actual_batch_size; ++i) {
        auto item = dataset->get_item(indices[current_index + i]);

        if (dev == Device::GPU) {
            cudaMemcpy(batch_inputs->data_gpu + (i * input_dim),
                       item.first->data_gpu,
                       input_dim * sizeof(float),
                       cudaMemcpyDeviceToDevice);

            cudaMemcpy(batch_targets->data_gpu + (i * target_dim),
                       item.second->data_gpu,
                       target_dim * sizeof(float),
                       cudaMemcpyDeviceToDevice);
        } else {
            std::copy(item.first->data, item.first->data + input_dim, batch_inputs->data + (i * input_dim));
            std::copy(item.second->data, item.second->data + target_dim, batch_targets->data + (i * target_dim));
        }
    }

    current_index += actual_batch_size;
    return {batch_inputs, batch_targets};
}

} // namespace ttensor