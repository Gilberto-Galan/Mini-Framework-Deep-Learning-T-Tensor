#ifndef LOSSES_HPP
#define LOSSES_HPP

#include "layers.hpp"
#include <stdexcept> // <--- CRUCIAL para std::runtime_error

namespace ttensor {

class Loss : public Module {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        throw std::runtime_error("Loss requiere input y target. Usa forward(input, target).");
    }
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) = 0;
    std::vector<std::shared_ptr<Tensor>> parameters() override { return {}; }
    void save(const std::string& path) override {}
    void load(const std::string& path) override {}
};

class MSELoss : public Loss {
public:
    using Loss::forward;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) override;
};

class CrossEntropyLoss : public Loss {
public:
    using Loss::forward;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) override;
};

} // namespace ttensor

#endif