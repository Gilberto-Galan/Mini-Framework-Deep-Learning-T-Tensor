#include "losses.hpp"
#include <stdexcept>
#include <cmath>
#include "kernels.hpp"

namespace ttensor {

// --- Mean Squared Error (MSE) ---
std::shared_ptr<Tensor> MSELoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    if (input->rows != target->rows || input->cols != target->cols) {
        throw std::runtime_error("[MSELoss] Dimensiones no coinciden entre input y target.");
    }

    // El resultado de la pérdida es un escalar (1x1)
    auto loss_val = std::make_shared<Tensor>(1, 1, input->device, true);
    
    // Calculamos el valor matemático del error
    float mse = input->mse_loss(*target);
    loss_val->fill(mse);

    // Registramos la operación en el grafo
    loss_val->inputs = {input};
    loss_val->backward_fn = [loss_val, input, target]() {
        if (input->requires_grad) {
            // dLoss/dInput = (2/N) * (Input - Target)
            Tensor grad_input = input->mse_backward(*target);
            
            // Acumulamos el gradiente en el buffer del input
            if (input->device == Device::GPU) {
                gpu_add_tensors(input->grad->data_gpu, grad_input.data_gpu,
                                input->grad->data_gpu, static_cast<int>(input->size));
            } else {
                for (size_t i = 0; i < input->size; ++i) {
                    input->grad->data[i] += grad_input.data[i];
                }
            }
        }
    };

    return loss_val;
}

// --- Cross Entropy Loss ---
std::shared_ptr<Tensor> CrossEntropyLoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    auto loss_val = std::make_shared<Tensor>(1, 1, input->device, true);
    
    // Softmax + CrossEntropy para estabilidad numérica
    auto probabilities = input->softmax(); 
    float ce_loss = probabilities->cross_entropy_with_target(*target);
    loss_val->fill(ce_loss);

    loss_val->inputs = {input};
    loss_val->backward_fn = [probabilities, input, target]() {
        if (input->requires_grad) {
            // La derivada de Softmax + CE es (Predicciones - Etiquetas_Reales)
            if (input->device == Device::GPU) {
                Tensor grad_input(input->rows, input->cols, input->device, false);
                gpu_sub_tensors(probabilities->data_gpu, target->data_gpu,
                                grad_input.data_gpu, static_cast<int>(input->size));
                gpu_add_tensors(input->grad->data_gpu, grad_input.data_gpu,
                                input->grad->data_gpu, static_cast<int>(input->size));
            } else {
                std::vector<float> probs = probabilities->to_vector();
                std::vector<float> targets = target->to_vector();
                for (size_t i = 0; i < input->size; ++i) {
                    input->grad->data[i] += probs[i] - targets[i];
                }
            }
        }
    };

    return loss_val;
}

} // namespace ttensor