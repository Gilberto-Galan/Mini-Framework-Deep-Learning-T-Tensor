#include "kernels.hpp"
#include "device_manager.hpp"
#include <device_launch_parameters.h>
#include <curand_kernel.h> 
#include <math.h>
#include <algorithm>

using ttensor::DeviceManager;

// --- KERNELS ---

// Inicialización aleatoria uniforme (cuRAND)
__global__ void random_uniform_kernel(float *data, int size, float low, float high, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float val = curand_uniform(&state);
        data[idx] = low + val * (high - low);
    }
}

// Multiplicación de matrices estándar (Forward)
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
            sum += A[row * N + i] * B[i * K + col];
        C[row * K + col] = sum;
    }
}

// Activación ReLU
__global__ void relu_kernel(const float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
}

__global__ void sigmoid_kernel(const float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// Suma de sesgo (Bias Forward)
__global__ void add_bias_kernel(float *data, float *bias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
        data[row * cols + col] += bias[col];
}

// Relleno de valor constante
__global__ void fill_kernel(float *data, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] = value;
}

// Cálculo de pérdida MSE (Usa atomicAdd para reducción)
__global__ void mse_loss_kernel(const float *pred, const float *target, float *loss_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = pred[idx] - target[idx];
        atomicAdd(loss_out, (diff * diff) / size);
    }
}

// Gradiente de MSE: (2/N) * (pred - target)
__global__ void mse_backward_kernel(const float *pred, const float *target, float *grad_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_out[idx] = (2.0f / size) * (pred[idx] - target[idx]);
    }
}

// Gradiente de ReLU (Máscara)
__global__ void relu_backward_kernel(const float* grad_out, const float* original_input, float* grad_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_in[idx] = (original_input[idx] > 0.0f) ? grad_out[idx] : 0.0f;
    }
}

__global__ void sigmoid_backward_kernel(const float* grad_out, const float* output, float* grad_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float sigmoid_value = output[idx];
        grad_in[idx] = grad_out[idx] * sigmoid_value * (1.0f - sigmoid_value);
    }
}

// Gradiente de Sesgo (Suma de columnas de dZ)
__global__ void bias_backward_kernel(const float* grad_z, float* grad_b, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {
            sum += grad_z[row * cols + col];
        }
        atomicAdd(&grad_b[col], sum); 
    }
}

// MatMul con Transposición de A (X^T * dZ) para gradiente de pesos
__global__ void matmul_transA_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[i * M + row] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

// MatMul con Transposición de B (dZ * W^T) para gradiente de entrada
__global__ void matmul_transB_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[col * N + i];
        }
        C[row * K + col] = sum;
    }
}

__global__ void add_tensors_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx];
}

__global__ void sub_tensors_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] - b[idx];
}

__global__ void mul_tensors_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] * b[idx];
}

// Optimizador SGD: W = W - (lr * grad)
__global__ void apply_gradient_kernel(float* weights, const float* grad, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx];
    }
}

__global__ void adam_update_kernel(float* weights, const float* grad, float* m, float* v, float lr, float beta1, float beta2, float eps, int timestep, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float grad_value = grad[idx];
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad_value;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad_value * grad_value;

        const float m_hat = m[idx] / (1.0f - powf(beta1, static_cast<float>(timestep)));
        const float v_hat = v[idx] / (1.0f - powf(beta2, static_cast<float>(timestep)));
        weights[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// --- log, exp, div elemento a elemento ---
__global__ void log_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        output[idx] = logf(fmaxf(input[idx], 1e-8f));
}

__global__ void exp_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        output[idx] = expf(input[idx]);
}

__global__ void div_tensors_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        out[idx] = a[idx] / fmaxf(b[idx], 1e-8f);
}

// --- Reducción suma global (todos los elementos) ---
__global__ void sum_all_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        atomicAdd(output, input[idx]);
}

// --- Suma por eje 0 (reduce filas) → resultado (1, cols) ---
__global__ void sum_rows_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float s = 0.0f;
        for (int r = 0; r < rows; ++r)
            s += input[r * cols + col];
        output[col] = s;
    }
}

// --- Suma por eje 1 (reduce columnas) → resultado (rows, 1) ---
__global__ void sum_cols_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float s = 0.0f;
        for (int c = 0; c < cols; ++c)
            s += input[row * cols + c];
        output[row] = s;
    }
}

// --- Broadcast vector fila (1, cols) → (rows, cols) ---
__global__ void broadcast_rowvec_kernel(const float* rowvec, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
        out[row * cols + col] = rowvec[col];
}

// Broadcast vector columna (rows, 1) → (rows, cols)
__global__ void broadcast_colvec_kernel(const float* colvec, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
        out[row * cols + col] = colvec[row];
}

// Softmax backward: un hilo por fila computa el producto punto y aplica el Jacobiano
// grad_in[row,i] += s[row,i] * (grad_out[row,i] - dot(grad_out[row], s[row]))
__global__ void softmax_backward_kernel(const float* grad_out, const float* s, float* grad_in, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        const int base = row * cols;
        float dot = 0.0f;
        for (int c = 0; c < cols; ++c)
            dot += grad_out[base + c] * s[base + c];
        for (int c = 0; c < cols; ++c)
            grad_in[base + c] += s[base + c] * (grad_out[base + c] - dot);
    }
}

namespace {
    template <typename KernelT>
    void resolve_launch_1d(KernelT kernel, int total_elements, int& blocks, int& threads) {
        DeviceManager::initialize();
        int suggested_blocks = 0;
        int suggested_threads = 0;
        cudaError_t occ_status = cudaOccupancyMaxPotentialBlockSize(
            &suggested_blocks,
            &suggested_threads,
            kernel,
            0,
            0
        );

        if (occ_status == cudaSuccess && suggested_threads > 0) {
            threads = std::min(DeviceManager::get_info().max_threads_per_block, suggested_threads);
        } else {
            threads = DeviceManager::get_optimal_threads_1d(total_elements, 256);
        }

        threads = std::max(32, threads);
        blocks = DeviceManager::get_optimal_blocks(total_elements, threads);
    }
}

// --- WRAPPERS ---
extern "C" {

    void gpu_fill_random(float *data, int size, float low, float high, unsigned long long seed) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(random_uniform_kernel, size, blocks, threads);
        random_uniform_kernel<<<blocks, threads>>>(data, size, low, high, seed);
    }

    void gpu_matmul(float *A, float *B, float *C, int M, int N, int K) {
        DeviceManager::initialize();
        dim3 threads = DeviceManager::get_optimal_block_2d();
        dim3 blocks = DeviceManager::get_grid_2d(M, K, threads);
        matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    }

    void gpu_relu(const float *input, float *output, int size) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(relu_kernel, size, blocks, threads);
        relu_kernel<<<blocks, threads>>>(input, output, size);
    }

    void gpu_sigmoid(const float *input, float *output, int size) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(sigmoid_kernel, size, blocks, threads);
        sigmoid_kernel<<<blocks, threads>>>(input, output, size);
    }

    void gpu_add_bias(float *data, float *bias, int rows, int cols) {
        DeviceManager::initialize();
        dim3 threads = DeviceManager::get_optimal_block_2d();
        dim3 blocks = DeviceManager::get_grid_2d(rows, cols, threads);
        add_bias_kernel<<<blocks, threads>>>(data, bias, rows, cols);
    }

    void gpu_fill(float *data, float value, int size) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(fill_kernel, size, blocks, threads);
        fill_kernel<<<blocks, threads>>>(data, value, size);
    }

    void gpu_mse_loss(const float *pred, const float *target, float *loss_out, int size) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(mse_loss_kernel, size, blocks, threads);
        mse_loss_kernel<<<blocks, threads>>>(pred, target, loss_out, size);
    }

    void gpu_mse_backward(const float *pred, const float *target, float *grad_out, int size) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(mse_backward_kernel, size, blocks, threads);
        mse_backward_kernel<<<blocks, threads>>>(pred, target, grad_out, size);
    }

    void gpu_relu_backward(const float* grad_out, const float* original_input, float* grad_in, int size) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(relu_backward_kernel, size, blocks, threads);
        relu_backward_kernel<<<blocks, threads>>>(grad_out, original_input, grad_in, size);
    }

    void gpu_sigmoid_backward(const float* grad_out, const float* output, float* grad_in, int size) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(sigmoid_backward_kernel, size, blocks, threads);
        sigmoid_backward_kernel<<<blocks, threads>>>(grad_out, output, grad_in, size);
    }

    void gpu_bias_backward(const float* grad_z, float* grad_b, int rows, int cols) {
        int threads = DeviceManager::get_optimal_threads_1d(cols, 256);
        int blocks = DeviceManager::get_optimal_blocks(cols, threads);
        bias_backward_kernel<<<blocks, threads>>>(grad_z, grad_b, rows, cols);
    }

    void gpu_matmul_transA(const float* A, const float* B, float* C, int M, int N, int K) {
        DeviceManager::initialize();
        dim3 threads = DeviceManager::get_optimal_block_2d();
        dim3 blocks = DeviceManager::get_grid_2d(M, K, threads);
        matmul_transA_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    }

    void gpu_matmul_transB(const float* A, const float* B, float* C, int M, int N, int K) {
        DeviceManager::initialize();
        dim3 threads = DeviceManager::get_optimal_block_2d();
        dim3 blocks = DeviceManager::get_grid_2d(M, K, threads);
        matmul_transB_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    }

    void gpu_add_tensors(const float* a, const float* b, float* out, int size) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(add_tensors_kernel, size, blocks, threads);
        add_tensors_kernel<<<blocks, threads>>>(a, b, out, size);
    }

    void gpu_sub_tensors(const float* a, const float* b, float* out, int size) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(sub_tensors_kernel, size, blocks, threads);
        sub_tensors_kernel<<<blocks, threads>>>(a, b, out, size);
    }

    void gpu_mul_tensors(const float* a, const float* b, float* out, int size) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(mul_tensors_kernel, size, blocks, threads);
        mul_tensors_kernel<<<blocks, threads>>>(a, b, out, size);
    }

    void gpu_apply_gradient(float* weights, const float* grad, float lr, int size) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(apply_gradient_kernel, size, blocks, threads);
        apply_gradient_kernel<<<blocks, threads>>>(weights, grad, lr, size);
    }

    void gpu_adam_update(float* weights, const float* grad, float* m, float* v, float lr, float beta1, float beta2, float eps, int timestep, int size) {
        int threads = 0;
        int blocks = 0;
        resolve_launch_1d(adam_update_kernel, size, blocks, threads);
        adam_update_kernel<<<blocks, threads>>>(weights, grad, m, v, lr, beta1, beta2, eps, timestep, size);
    }

    void gpu_log(const float* input, float* output, int size) {
        int threads = 0, blocks = 0;
        resolve_launch_1d(log_kernel, size, blocks, threads);
        log_kernel<<<blocks, threads>>>(input, output, size);
    }

    void gpu_exp(const float* input, float* output, int size) {
        int threads = 0, blocks = 0;
        resolve_launch_1d(exp_kernel, size, blocks, threads);
        exp_kernel<<<blocks, threads>>>(input, output, size);
    }

    void gpu_div_tensors(const float* a, const float* b, float* out, int size) {
        int threads = 0, blocks = 0;
        resolve_launch_1d(div_tensors_kernel, size, blocks, threads);
        div_tensors_kernel<<<blocks, threads>>>(a, b, out, size);
    }

    void gpu_sum_all(const float* input, float* output, int size) {
        int threads = 0, blocks = 0;
        resolve_launch_1d(sum_all_kernel, size, blocks, threads);
        sum_all_kernel<<<blocks, threads>>>(input, output, size);
    }

    void gpu_sum_rows(const float* input, float* output, int rows, int cols) {
        int threads = DeviceManager::get_optimal_threads_1d(cols, 256);
        int blocks = DeviceManager::get_optimal_blocks(cols, threads);
        sum_rows_kernel<<<blocks, threads>>>(input, output, rows, cols);
    }

    void gpu_sum_cols(const float* input, float* output, int rows, int cols) {
        int threads = DeviceManager::get_optimal_threads_1d(rows, 256);
        int blocks = DeviceManager::get_optimal_blocks(rows, threads);
        sum_cols_kernel<<<blocks, threads>>>(input, output, rows, cols);
    }

    void gpu_broadcast_rowvec(const float* rowvec, float* out, int rows, int cols) {
        DeviceManager::initialize();
        dim3 threads = DeviceManager::get_optimal_block_2d();
        dim3 blocks = DeviceManager::get_grid_2d(rows, cols, threads);
        broadcast_rowvec_kernel<<<blocks, threads>>>(rowvec, out, rows, cols);
    }

    void gpu_broadcast_colvec(const float* colvec, float* out, int rows, int cols) {
        DeviceManager::initialize();
        dim3 threads = DeviceManager::get_optimal_block_2d();
        dim3 blocks = DeviceManager::get_grid_2d(rows, cols, threads);
        broadcast_colvec_kernel<<<blocks, threads>>>(colvec, out, rows, cols);
    }

    void gpu_softmax_backward(const float* grad_out, const float* softmax_out, float* grad_in, int rows, int cols) {
        int threads = DeviceManager::get_optimal_threads_1d(rows, 256);
        int blocks = DeviceManager::get_optimal_blocks(rows, threads);
        softmax_backward_kernel<<<blocks, threads>>>(grad_out, softmax_out, grad_in, rows, cols);
    }
}