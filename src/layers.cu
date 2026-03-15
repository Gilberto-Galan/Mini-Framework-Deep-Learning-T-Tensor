#include "layers.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#include "kernels.hpp"

namespace ttensor {

// --- LINEAR LAYER (Totalmente conectada) ---
Linear::Linear(int in_f, int out_f, Device dev) {
    // Pesos: (entradas x salidas), Bias: (1 x salidas)
    weights = std::make_shared<Tensor>(in_f, out_f, dev, true);
    bias = std::make_shared<Tensor>(1, out_f, dev, true);
    
    // Inicialización aleatoria pequeña
    weights->fill_random(-0.1f, 0.1f);
    bias->fill(0.0f);
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    // Implementación de Y = X*W + b con registro automático en el grafo
    auto mul = Tensor::matmul(input, weights);
    return Tensor::add(mul, bias);
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
    return {weights, bias};
}

// --- RELU ACTIVATION ---
std::shared_ptr<Tensor> ReLU::forward(std::shared_ptr<Tensor> input) {
    // Delegamos la lógica y el backward al método estático de Tensor
    return Tensor::relu(input);
}

std::shared_ptr<Tensor> Sigmoid::forward(std::shared_ptr<Tensor> input) {
    return Tensor::sigmoid(input);
}

std::shared_ptr<Tensor> Softmax::forward(std::shared_ptr<Tensor> input) {
    return Tensor::softmax(input);
}

// --- SEQUENTIAL CONTAINER ---
std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> input) {
    std::shared_ptr<Tensor> current = input;
    for (auto& layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

std::vector<std::shared_ptr<Tensor>> Sequential::parameters() {
    std::vector<std::shared_ptr<Tensor>> p;
    for (auto& l : layers) {
        auto lp = l->parameters();
        p.insert(p.end(), lp.begin(), lp.end());
    }
    return p;
}

void Sequential::save(const std::string& path) {
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i]->save(path + "_layer_" + std::to_string(i) + ".bin");
    }
}

void Sequential::load(const std::string& path) {
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i]->load(path + "_layer_" + std::to_string(i) + ".bin");
    }
}

// --- PERSISTENCIA (GUARDAR/CARGAR) ---
void Linear::save(const std::string& path) {
    weights->to_cpu();
    bias->to_cpu();
    std::ofstream ofs(path, std::ios::binary);
    if (ofs.is_open()) {
        ofs.write((char*)weights->data, weights->size * sizeof(float));
        ofs.write((char*)bias->data, bias->size * sizeof(float));
    }
    weights->to_gpu();
    bias->to_gpu();
}

void Linear::load(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return;
    weights->to_cpu();
    bias->to_cpu();
    ifs.read((char*)weights->data, weights->size * sizeof(float));
    ifs.read((char*)bias->data, bias->size * sizeof(float));
    weights->to_gpu();
    bias->to_gpu();
}

} // namespace ttensor