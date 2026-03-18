#include "optimizer.hpp"
#include "kernels.hpp"
#include <stdexcept>
#include <cmath>

namespace ttensor {

void SGD::step() {
    for (auto& p : params) {
        // Solo actualizamos si el tensor requiere gradiente y efectivamente lo tiene calculado
        if (p->requires_grad && p->grad) {
            
            // Verificamos que las dimensiones coincidan por seguridad
            if (p->rows != p->grad->rows || p->cols != p->grad->cols) {
                throw std::runtime_error("[Optimizer] Error: Dimensiones de gradiente no coinciden con el parámetro.");
            }

            // Llamamos al kernel de optimización que ya tienes en Tensor
            // p->data = p->data - lr * p->grad->data
            p->apply_gradient(*(p->grad), lr);
        }
    }
}

Adam::Adam(std::vector<std::shared_ptr<Tensor>> parameters, float learning_rate, float beta1_value, float beta2_value, float eps_value)
    : Optimizer(std::move(parameters), learning_rate), beta1(beta1_value), beta2(beta2_value), eps(eps_value), timestep(0) {
    first_moment.reserve(params.size());
    second_moment.reserve(params.size());

    for (const auto& parameter : params) {
        auto m = std::make_shared<Tensor>(parameter->rows, parameter->cols, parameter->device, false);
        auto v = std::make_shared<Tensor>(parameter->rows, parameter->cols, parameter->device, false);
        m->fill(0.0f);
        v->fill(0.0f);
        first_moment.push_back(m);
        second_moment.push_back(v);
    }
}

void Adam::step() {
    ++timestep;

    for (size_t index = 0; index < params.size(); ++index) {
        auto& p = params[index];
        if (!p->requires_grad || !p->grad) {
            continue;
        }

        if (p->rows != p->grad->rows || p->cols != p->grad->cols) {
            throw std::runtime_error("[Adam] Error: Dimensiones de gradiente no coinciden con el parametro.");
        }

        auto& m = first_moment[index];
        auto& v = second_moment[index];

        if (p->device == Device::GPU) {
            gpu_adam_update(p->data_gpu, p->grad->data_gpu, m->data_gpu, v->data_gpu, lr, beta1, beta2, eps, timestep, static_cast<int>(p->size));
        } else {
            const float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(timestep));
            const float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(timestep));
            for (size_t i = 0; i < p->size; ++i) {
                const float grad_value = p->grad->data[i];
                m->data[i] = beta1 * m->data[i] + (1.0f - beta1) * grad_value;
                v->data[i] = beta2 * v->data[i] + (1.0f - beta2) * grad_value * grad_value;

                const float m_hat = m->data[i] / bias_correction1;
                const float v_hat = v->data[i] / bias_correction2;
                p->data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }
}

} // namespace ttensor