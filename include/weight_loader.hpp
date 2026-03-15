#ifndef WEIGHT_LOADER_HPP
#define WEIGHT_LOADER_HPP

#include "tensor.hpp"
#include <string>
#include <vector>
#include <fstream>

class WeightLoader {
public:
    // Lee un archivo binario y devuelve un vector de floats
    static std::vector<float> load_binary(const std::string& filename, size_t expected_size) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("No se pudo abrir el archivo de pesos: " + filename);
        }

        std::vector<float> buffer(expected_size);
        file.read(reinterpret_cast<char*>(buffer.data()), expected_size * sizeof(float));
        
        if (file.gcount() != static_cast<std::streamsize>(expected_size * sizeof(float))) {
            throw std::runtime_error("El archivo " + filename + " no tiene el tamaño esperado.");
        }

        file.close();
        return buffer;
    }

    // Guarda un vector de floats en un archivo binario
    static void save_binary(const std::string& filename, const float* data, size_t size) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("No se pudo crear el archivo de pesos: " + filename);
        }

        file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
        file.close();
    }
};

#endif