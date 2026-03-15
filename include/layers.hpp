#ifndef TTENSOR_LAYERS_HPP
#define TTENSOR_LAYERS_HPP

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <string>

namespace ttensor {

class Module {
public:
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> parameters() = 0;
    
    void zero_grad() {
        for (auto& p : parameters()) p->zero_grad();
    }

    virtual void save(const std::string& path) = 0;
    virtual void load(const std::string& path) = 0;
    virtual ~Module() {}
};

class Linear : public Module {
public:
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;

    Linear(int in_f, int out_f, Device dev = Device::GPU);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;
};

class ReLU : public Module {
public:
    ReLU() {}
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override { return {}; }
    void save(const std::string& path) override { (void)path; }
    void load(const std::string& path) override { (void)path; }
};

class Sigmoid : public Module {
public:
    Sigmoid() {}
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override { return {}; }
    void save(const std::string& path) override { (void)path; }
    void load(const std::string& path) override { (void)path; }
};

class Softmax : public Module {
public:
    Softmax() {}
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override { return {}; }
    void save(const std::string& path) override { (void)path; }
    void load(const std::string& path) override { (void)path; }
};

class Sequential : public Module {
public:
    std::vector<std::shared_ptr<Module>> layers;
    
    Sequential(std::vector<std::shared_ptr<Module>> l) : layers(l) {}
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;
};

} // namespace ttensor
#endif