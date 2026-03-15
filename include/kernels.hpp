#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <cuda_runtime.h>

extern "C" {
    // --- OPERACIONES DE FORWARD (INFERENCIA) ---
    void gpu_matmul(float* A, float* B, float* C, int M, int N, int K);
    void gpu_relu(const float* input, float* output, int size);
    void gpu_sigmoid(const float* input, float* output, int size);
    void gpu_add_bias(float* data, float* bias, int rows, int cols);
    void gpu_fill(float* data, float value, int size);
    void gpu_fill_random(float* data, int size, float low, float high, unsigned long long seed);
    
    // --- MÉTRICAS Y BACKPROPAGATION (ENTRENAMIENTO) ---
    
    // Pérdida (Loss)
    void gpu_mse_loss(const float* pred, const float* target, float* loss_out, int size);
    void gpu_mse_backward(const float* pred, const float* target, float* grad_out, int size);
    
    // Activación Backward
    void gpu_relu_backward(const float* grad_out, const float* original_input, float* grad_in, int size);
    void gpu_sigmoid_backward(const float* grad_out, const float* output, float* grad_in, int size);
    // grad_in[row,i] += softmax[row,i] * (grad_out[row,i] - dot(grad_out[row], softmax[row]))
    void gpu_softmax_backward(const float* grad_out, const float* softmax_out, float* grad_in, int rows, int cols);
    
    // Gradientes de Parámetros (Pesos y Sesgos)
    // X^T * dZ -> Para calcular el gradiente respecto a los pesos (W)
    void gpu_matmul_transA(const float* A, const float* B, float* C, int M, int N, int K);
    // dZ * W^T -> Para calcular el gradiente respecto a la entrada (X)
    void gpu_matmul_transB(const float* A, const float* B, float* C, int M, int N, int K);
    
    // Reducción de dZ -> Para calcular el gradiente respecto al sesgo (B)
    void gpu_bias_backward(const float* grad_z, float* grad_b, int rows, int cols);

    // Operaciones elemento a elemento para acumulación de gradientes
    void gpu_add_tensors(const float* a, const float* b, float* out, int size);
    void gpu_sub_tensors(const float* a, const float* b, float* out, int size);
    void gpu_mul_tensors(const float* a, const float* b, float* out, int size);

    // --- OPTIMIZACIÓN ---
    // Actualización de parámetros: W = W - (lr * grad)
    void gpu_apply_gradient(float* weights, const float* grad, float lr, int size);
    void gpu_adam_update(float* weights, const float* grad, float* m, float* v, float lr, float beta1, float beta2, float eps, int timestep, int size);

    // --- OPERACIONES ELEMENTO A ELEMENTO (LOG, EXP, DIV) ---
    void gpu_log(const float* input, float* output, int size);
    void gpu_exp(const float* input, float* output, int size);
    void gpu_div_tensors(const float* a, const float* b, float* out, int size);

    // --- REDUCCIÓN Y BROADCASTING ---
    void gpu_sum_all(const float* input, float* output, int size);            // Suma global → escalar
    void gpu_sum_rows(const float* input, float* output, int rows, int cols); // axis=0 → (1, cols)
    void gpu_sum_cols(const float* input, float* output, int rows, int cols); // axis=1 → (rows, 1)
    void gpu_broadcast_rowvec(const float* rowvec, float* out, int rows, int cols);
    void gpu_broadcast_colvec(const float* colvec, float* out, int rows, int cols);
}

#endif