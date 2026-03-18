#ifndef TTENSOR_DATALOADER_HPP
#define TTENSOR_DATALOADER_HPP

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <algorithm>
#include <random>
#include <string>

namespace ttensor {

// Clase abstracta para representar cualquier conjunto de datos
class Dataset {
public:
    virtual size_t size() = 0;
    // Retorna un par: {Input_Tensor, Target_Tensor} para un índice dado
    virtual std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get_item(size_t index) = 0;
    virtual ~Dataset() {}
};

class DataLoader {
private:
    std::shared_ptr<Dataset> dataset;
    size_t batch_size;
    bool shuffle;
    std::vector<size_t> indices;
    size_t current_index;
    
    std::default_random_engine rng;

public:
    DataLoader(std::shared_ptr<Dataset> ds, size_t batch, bool shuff = true)
        : dataset(ds), batch_size(batch), shuffle(shuff), current_index(0) {
        
        // Inicializar índices secuenciales
        for (size_t i = 0; i < dataset->size(); ++i) indices.push_back(i);
        
        if (shuffle) {
            std::random_device rd;
            rng = std::default_random_engine(rd());
        }
    }

    // Mezcla los índices para una nueva época
    void reset() {
        current_index = 0;
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
    }

    // Obtiene el siguiente lote de datos
    // Retorna {Batch_Inputs, Batch_Targets}
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> next_batch();

    bool has_next() const {
        return current_index < dataset->size();
    }
};


class CSVDataset : public Dataset {
private:
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> targets;
    Device device;

public:
    CSVDataset(const std::string& path, int label_cols, Device dev = Device::GPU);
    
    size_t size() override { return data.size(); }
    
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get_item(size_t index) override {
        return {
            Tensor::from_vector(data[index], 1, static_cast<int>(data[index].size()), device, false),
            Tensor::from_vector(targets[index], 1, static_cast<int>(targets[index].size()), device, false)
        };
    }
};

} // namespace ttensor

#endif