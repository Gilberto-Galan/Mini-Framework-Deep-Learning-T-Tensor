#ifndef TTENSOR_OPTIMIZER_HPP
#define TTENSOR_OPTIMIZER_HPP

#include "tensor.hpp"
#include <vector>
#include <memory>

namespace ttensor {

// Clase base abstracta para todos los optimizadores
class Optimizer {
protected:
    std::vector<std::shared_ptr<Tensor>> params; // Referencias a los tensores entrenables
    float lr;                                    // Learning Rate

public:
    Optimizer(std::vector<std::shared_ptr<Tensor>> parameters, float learning_rate)
        : params(parameters), lr(learning_rate) {}

    // Limpia los gradientes de todos los parámetros (típico optimizer.zero_grad())
    void zero_grad() {
        for (auto& p : params) {
            p->zero_grad();
        }
    }

    // Método virtual puro: cada optimizador implementa su regla de actualización
    virtual void step() = 0;

    virtual ~Optimizer() {}
};

// Descenso de Gradiente Estocástico (SGD)
class SGD : public Optimizer {
public:
    // Heredamos el constructor de la base
    using Optimizer::Optimizer;

    // Implementación de la actualización simple: W = W - lr * grad
    void step() override;
};

class Adam : public Optimizer {
private:
    std::vector<std::shared_ptr<Tensor>> first_moment;
    std::vector<std::shared_ptr<Tensor>> second_moment;
    float beta1;
    float beta2;
    float eps;
    int timestep;

public:
    Adam(std::vector<std::shared_ptr<Tensor>> parameters, float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void step() override;
}; 

} // namespace ttensor

#endif