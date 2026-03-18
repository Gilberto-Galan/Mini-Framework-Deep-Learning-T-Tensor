#include "tensor.hpp"
#include "kernels.hpp" // Asegúrate de que este archivo tenga las declaraciones gpu_
#include "device_manager.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <ctime>
#include <random>
#include <set>
#include <cmath>

namespace ttensor {

    namespace {
        void ensure_same_device(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b, const char* op_name) {
            if (a->device != b->device) {
                throw std::runtime_error(std::string("[") + op_name + "] Ambos tensores deben estar en el mismo device.");
            }
        }

        void ensure_same_shape(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b, const char* op_name) {
            if (a->rows != b->rows || a->cols != b->cols) {
                throw std::runtime_error(std::string("[") + op_name + "] Dimensiones incompatibles.");
            }
        }

        void accumulate_bias_grad_cpu(const Tensor& grad_z, Tensor& grad_b) {
            for (int c = 0; c < grad_z.cols; ++c) {
                float sum = 0.0f;
                for (int r = 0; r < grad_z.rows; ++r) {
                    sum += grad_z.data[r * grad_z.cols + c];
                }
                grad_b.data[c] += sum;
            }
        }
    }

    // Constructor unificado
    Tensor::Tensor(int r, int c, Device dev, bool req_grad)
        : rows(r), cols(c), size(static_cast<size_t>(r) * static_cast<size_t>(c)), device(dev), data(nullptr), data_gpu(nullptr), grad(nullptr), requires_grad(req_grad), backward_fn(nullptr) {
        allocate();
        if (requires_grad) {
            grad = std::make_shared<Tensor>(rows, cols, device, false);
            grad->fill(0.0f);
        }
    }

    Tensor::Tensor(const Tensor &other)
        : rows(other.rows), cols(other.cols), size(other.size), device(other.device), data(nullptr), data_gpu(nullptr), grad(nullptr), requires_grad(false), backward_fn(nullptr) {
        allocate();
        if (device == Device::GPU) {
            cudaMemcpy(data_gpu, other.data_gpu, size * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            std::copy(other.data, other.data + size, data);
        }
    }

    Tensor::~Tensor() { deallocate(); }

    void Tensor::allocate() {
        if (device == Device::GPU) {
            DeviceManager::initialize();
            if (DeviceManager::should_use_managed_memory()) {
                cudaMallocManaged(&data_gpu, size * sizeof(float));
            } else {
                cudaMalloc(&data_gpu, size * sizeof(float));
            }
            data = nullptr;
        } else {
            data = new float[size];
            data_gpu = nullptr;
        }
    }

    void Tensor::deallocate() {
        if (device == Device::GPU) {
            if (data_gpu) cudaFree(data_gpu);
        } else {
            if (data) delete[] data;
        }
    }

    // --- MOTOR DE AUTOGRAD (OPERACIONES ESTÁTICAS) ---

    std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        ensure_same_device(a, b, "Tensor::matmul");
        if (a->cols != b->rows) {
            throw std::runtime_error("[Tensor::matmul] Dimensiones incompatibles.");
        }

        auto res = std::make_shared<Tensor>(a->rows, b->cols, a->device, a->requires_grad || b->requires_grad);
        if (a->device == Device::GPU) {
            gpu_matmul(a->data_gpu, b->data_gpu, res->data_gpu, a->rows, a->cols, b->cols);
        } else {
            for (int r = 0; r < a->rows; ++r) {
                for (int c = 0; c < b->cols; ++c) {
                    float sum = 0.0f;
                    for (int k = 0; k < a->cols; ++k) {
                        sum += a->data[r * a->cols + k] * b->data[k * b->cols + c];
                    }
                    res->data[r * res->cols + c] = sum;
                }
            }
        }

        if (res->requires_grad) {
            res->inputs = {a, b};
            res->backward_fn = [res, a, b]() {
                if (a->requires_grad) {
                    Tensor dX = res->grad->matmul_transB(*b);
                    if (a->device == Device::GPU) {
                        gpu_add_tensors(a->grad->data_gpu, dX.data_gpu, a->grad->data_gpu, static_cast<int>(a->size));
                    } else {
                        for (size_t i = 0; i < a->size; ++i) {
                            a->grad->data[i] += dX.data[i];
                        }
                    }
                }
                if (b->requires_grad) {
                    Tensor dW = a->matmul_transA(*(res->grad));
                    if (b->device == Device::GPU) {
                        gpu_add_tensors(b->grad->data_gpu, dW.data_gpu, b->grad->data_gpu, static_cast<int>(b->size));
                    } else {
                        for (size_t i = 0; i < b->size; ++i) {
                            b->grad->data[i] += dW.data[i];
                        }
                    }
                }
            };
        }
        return res;
    }

    std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        ensure_same_device(a, b, "Tensor::add");

        const bool same_shape = a->rows == b->rows && a->cols == b->cols;
        const bool b_is_bias = b->rows == 1 && b->cols == a->cols;
        const bool a_is_bias = a->rows == 1 && a->cols == b->cols;

        if (!same_shape && !b_is_bias && !a_is_bias) {
            throw std::runtime_error("[Tensor::add] Dimensiones incompatibles.");
        }

        const int out_rows = a_is_bias ? b->rows : a->rows;
        const int out_cols = a_is_bias ? b->cols : a->cols;
        auto res = std::make_shared<Tensor>(out_rows, out_cols, a->device, a->requires_grad || b->requires_grad);

        if (a->device == Device::GPU) {
            if (same_shape) {
                gpu_add_tensors(a->data_gpu, b->data_gpu, res->data_gpu, static_cast<int>(a->size));
            } else if (b_is_bias) {
                cudaMemcpy(res->data_gpu, a->data_gpu, a->size * sizeof(float), cudaMemcpyDeviceToDevice);
                gpu_add_bias(res->data_gpu, b->data_gpu, a->rows, a->cols);
            } else {
                cudaMemcpy(res->data_gpu, b->data_gpu, b->size * sizeof(float), cudaMemcpyDeviceToDevice);
                gpu_add_bias(res->data_gpu, a->data_gpu, b->rows, b->cols);
            }
        } else if (same_shape) {
            for (size_t i = 0; i < a->size; ++i) {
                res->data[i] = a->data[i] + b->data[i];
            }
        } else if (b_is_bias) {
            for (int r = 0; r < a->rows; ++r) {
                for (int c = 0; c < a->cols; ++c) {
                    res->data[r * a->cols + c] = a->data[r * a->cols + c] + b->data[c];
                }
            }
        } else {
            for (int r = 0; r < b->rows; ++r) {
                for (int c = 0; c < b->cols; ++c) {
                    res->data[r * b->cols + c] = b->data[r * b->cols + c] + a->data[c];
                }
            }
        }

        if (res->requires_grad) {
            res->inputs = {a, b};
            res->backward_fn = [res, a, b, same_shape, b_is_bias, a_is_bias]() {
                if (a->requires_grad) {
                    if (same_shape || b_is_bias) {
                        if (a->device == Device::GPU) {
                            gpu_add_tensors(a->grad->data_gpu, res->grad->data_gpu, a->grad->data_gpu, static_cast<int>(a->size));
                        } else {
                            for (size_t i = 0; i < a->size; ++i) {
                                a->grad->data[i] += res->grad->data[i];
                            }
                        }
                    } else if (a_is_bias) {
                        if (a->device == Device::GPU) {
                            gpu_bias_backward(res->grad->data_gpu, a->grad->data_gpu, res->rows, res->cols);
                        } else {
                            accumulate_bias_grad_cpu(*(res->grad), *(a->grad));
                        }
                    }
                }

                if (b->requires_grad) {
                    if (same_shape || a_is_bias) {
                        if (b->device == Device::GPU) {
                            gpu_add_tensors(b->grad->data_gpu, res->grad->data_gpu, b->grad->data_gpu, static_cast<int>(b->size));
                        } else {
                            for (size_t i = 0; i < b->size; ++i) {
                                b->grad->data[i] += res->grad->data[i];
                            }
                        }
                    } else if (b_is_bias) {
                        if (b->device == Device::GPU) {
                            gpu_bias_backward(res->grad->data_gpu, b->grad->data_gpu, res->rows, res->cols);
                        } else {
                            accumulate_bias_grad_cpu(*(res->grad), *(b->grad));
                        }
                    }
                }
            };
        }
        return res;
    }

    std::shared_ptr<Tensor> Tensor::multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        ensure_same_device(a, b, "Tensor::multiply");
        ensure_same_shape(a, b, "Tensor::multiply");

        auto res = std::make_shared<Tensor>(a->rows, a->cols, a->device, a->requires_grad || b->requires_grad);
        if (a->device == Device::GPU) {
            gpu_mul_tensors(a->data_gpu, b->data_gpu, res->data_gpu, static_cast<int>(a->size));
        } else {
            for (size_t i = 0; i < a->size; ++i) {
                res->data[i] = a->data[i] * b->data[i];
            }
        }

        if (res->requires_grad) {
            res->inputs = {a, b};
            res->backward_fn = [res, a, b]() {
                if (a->requires_grad) {
                    Tensor local_grad(a->rows, a->cols, a->device, false);
                    if (a->device == Device::GPU) {
                        gpu_mul_tensors(res->grad->data_gpu, b->data_gpu, local_grad.data_gpu, static_cast<int>(a->size));
                        gpu_add_tensors(a->grad->data_gpu, local_grad.data_gpu, a->grad->data_gpu, static_cast<int>(a->size));
                    } else {
                        for (size_t i = 0; i < a->size; ++i) {
                            a->grad->data[i] += res->grad->data[i] * b->data[i];
                        }
                    }
                }

                if (b->requires_grad) {
                    Tensor local_grad(b->rows, b->cols, b->device, false);
                    if (b->device == Device::GPU) {
                        gpu_mul_tensors(res->grad->data_gpu, a->data_gpu, local_grad.data_gpu, static_cast<int>(b->size));
                        gpu_add_tensors(b->grad->data_gpu, local_grad.data_gpu, b->grad->data_gpu, static_cast<int>(b->size));
                    } else {
                        for (size_t i = 0; i < b->size; ++i) {
                            b->grad->data[i] += res->grad->data[i] * a->data[i];
                        }
                    }
                }
            };
        }

        return res;
    }

    std::shared_ptr<Tensor> Tensor::sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        ensure_same_device(a, b, "Tensor::sub");
        if (a->rows != b->rows || a->cols != b->cols) {
            throw std::runtime_error("[Tensor::sub] Dimensiones incompatibles.");
        }

        auto res = std::make_shared<Tensor>(a->rows, a->cols, a->device, a->requires_grad || b->requires_grad);
        if (a->device == Device::GPU) {
            gpu_sub_tensors(a->data_gpu, b->data_gpu, res->data_gpu, static_cast<int>(a->size));
        } else {
            for (size_t i = 0; i < a->size; ++i) {
                res->data[i] = a->data[i] - b->data[i];
            }
        }

        if (res->requires_grad) {
            res->inputs = {a, b};
            res->backward_fn = [res, a, b]() {
                if (a->requires_grad) {
                    if (a->device == Device::GPU) {
                        gpu_add_tensors(a->grad->data_gpu, res->grad->data_gpu, a->grad->data_gpu, static_cast<int>(a->size));
                    } else {
                        for (size_t i = 0; i < a->size; ++i) {
                            a->grad->data[i] += res->grad->data[i];
                        }
                    }
                }
                if (b->requires_grad) {
                    if (b->device == Device::GPU) {
                        gpu_sub_tensors(b->grad->data_gpu, res->grad->data_gpu, b->grad->data_gpu, static_cast<int>(b->size));
                    } else {
                        for (size_t i = 0; i < b->size; ++i) {
                            b->grad->data[i] -= res->grad->data[i];
                        }
                    }
                }
            };
        }

        return res;
    }

    std::shared_ptr<Tensor> Tensor::relu(std::shared_ptr<Tensor> input) {
        auto res = std::make_shared<Tensor>(input->rows, input->cols, input->device, input->requires_grad);
        if (input->device == Device::GPU) {
            gpu_relu(input->data_gpu, res->data_gpu, static_cast<int>(input->size));
        } else {
            for (size_t i = 0; i < input->size; ++i) {
                res->data[i] = std::max(0.0f, input->data[i]);
            }
        }

        if (res->requires_grad) {
            res->inputs = {input};
            res->backward_fn = [res, input]() {
                if (input->requires_grad) {
                    if (input->device == Device::GPU) {
                        Tensor local_grad(input->rows, input->cols, input->device, false);
                        gpu_relu_backward(res->grad->data_gpu, input->data_gpu, local_grad.data_gpu, static_cast<int>(input->size));
                        gpu_add_tensors(input->grad->data_gpu, local_grad.data_gpu, input->grad->data_gpu, static_cast<int>(input->size));
                    } else {
                        for (size_t i = 0; i < input->size; ++i) {
                            input->grad->data[i] += (input->data[i] > 0.0f) ? res->grad->data[i] : 0.0f;
                        }
                    }
                }
            };
        }
        return res;
    }

    std::shared_ptr<Tensor> Tensor::sigmoid(std::shared_ptr<Tensor> input) {
        auto res = std::make_shared<Tensor>(input->rows, input->cols, input->device, input->requires_grad);
        if (input->device == Device::GPU) {
            gpu_sigmoid(input->data_gpu, res->data_gpu, static_cast<int>(input->size));
        } else {
            for (size_t i = 0; i < input->size; ++i) {
                res->data[i] = 1.0f / (1.0f + std::exp(-input->data[i]));
            }
        }

        if (res->requires_grad) {
            res->inputs = {input};
            res->backward_fn = [res, input]() {
                if (input->requires_grad) {
                    if (input->device == Device::GPU) {
                        Tensor local_grad(input->rows, input->cols, input->device, false);
                        gpu_sigmoid_backward(res->grad->data_gpu, res->data_gpu, local_grad.data_gpu, static_cast<int>(input->size));
                        gpu_add_tensors(input->grad->data_gpu, local_grad.data_gpu, input->grad->data_gpu, static_cast<int>(input->size));
                    } else {
                        for (size_t i = 0; i < input->size; ++i) {
                            input->grad->data[i] += res->grad->data[i] * res->data[i] * (1.0f - res->data[i]);
                        }
                    }
                }
            };
        }

        return res;
    }

    // --- SOFTMAX diferenciable (nodo del grafo con backward Jacobiano) ---
    std::shared_ptr<Tensor> Tensor::softmax(std::shared_ptr<Tensor> input) {
        const int rows = input->rows;
        const int cols = input->cols;

        auto res = std::make_shared<Tensor>(rows, cols, input->device, input->requires_grad);

        // Forward: softmax estable por filas (siempre en CPU, luego se transfiere si GPU)
        std::vector<float> h_in = input->to_vector();
        std::vector<float> h_out(input->size);
        for (int r = 0; r < rows; ++r) {
            const int base = r * cols;
            float row_max = h_in[base];
            for (int c = 1; c < cols; ++c) row_max = std::max(row_max, h_in[base + c]);
            float sum_e = 0.0f;
            for (int c = 0; c < cols; ++c) {
                h_out[base + c] = std::exp(h_in[base + c] - row_max);
                sum_e += h_out[base + c];
            }
            for (int c = 0; c < cols; ++c) h_out[base + c] /= sum_e;
        }
        res->set_data(h_out);

        if (res->requires_grad) {
            res->inputs = {input};
            res->backward_fn = [res, input]() {
                if (!input->requires_grad) return;
                if (input->device == Device::GPU) {
                    gpu_softmax_backward(res->grad->data_gpu, res->data_gpu,
                                         input->grad->data_gpu, input->rows, input->cols);
                } else {
                    const int rows = input->rows;
                    const int cols = input->cols;
                    for (int r = 0; r < rows; ++r) {
                        const int base = r * cols;
                        float dot = 0.0f;
                        for (int c = 0; c < cols; ++c)
                            dot += res->grad->data[base + c] * res->data[base + c];
                        for (int c = 0; c < cols; ++c)
                            input->grad->data[base + c] += res->data[base + c] *
                                                            (res->grad->data[base + c] - dot);
                    }
                }
            };
        }
        return res;
    }

    // --- LOG diferenciable: d/dx log(x) = 1/x ---
    std::shared_ptr<Tensor> Tensor::log(std::shared_ptr<Tensor> input) {
        auto res = std::make_shared<Tensor>(input->rows, input->cols, input->device, input->requires_grad);
        if (input->device == Device::GPU) {
            gpu_log(input->data_gpu, res->data_gpu, static_cast<int>(input->size));
        } else {
            for (size_t i = 0; i < input->size; ++i)
                res->data[i] = std::log(std::max(input->data[i], 1e-8f));
        }
        if (res->requires_grad) {
            res->inputs = {input};
            res->backward_fn = [res, input]() {
                if (!input->requires_grad) return;
                if (input->device == Device::GPU) {
                    Tensor local_grad(input->rows, input->cols, input->device, false);
                    gpu_div_tensors(res->grad->data_gpu, input->data_gpu, local_grad.data_gpu, static_cast<int>(input->size));
                    gpu_add_tensors(input->grad->data_gpu, local_grad.data_gpu, input->grad->data_gpu, static_cast<int>(input->size));
                } else {
                    for (size_t i = 0; i < input->size; ++i)
                        input->grad->data[i] += res->grad->data[i] / std::max(input->data[i], 1e-8f);
                }
            };
        }
        return res;
    }

    // --- EXP diferenciable: d/dx exp(x) = exp(x) ---
    std::shared_ptr<Tensor> Tensor::exp(std::shared_ptr<Tensor> input) {
        auto res = std::make_shared<Tensor>(input->rows, input->cols, input->device, input->requires_grad);
        if (input->device == Device::GPU) {
            gpu_exp(input->data_gpu, res->data_gpu, static_cast<int>(input->size));
        } else {
            for (size_t i = 0; i < input->size; ++i)
                res->data[i] = std::exp(input->data[i]);
        }
        if (res->requires_grad) {
            res->inputs = {input};
            res->backward_fn = [res, input]() {
                if (!input->requires_grad) return;
                if (input->device == Device::GPU) {
                    Tensor local_grad(input->rows, input->cols, input->device, false);
                    gpu_mul_tensors(res->grad->data_gpu, res->data_gpu, local_grad.data_gpu, static_cast<int>(input->size));
                    gpu_add_tensors(input->grad->data_gpu, local_grad.data_gpu, input->grad->data_gpu, static_cast<int>(input->size));
                } else {
                    for (size_t i = 0; i < input->size; ++i)
                        input->grad->data[i] += res->grad->data[i] * res->data[i];
                }
            };
        }
        return res;
    }

    // --- SUM diferenciable (axis = -1: global, 0: filas, 1: columnas) ---
    std::shared_ptr<Tensor> Tensor::sum(std::shared_ptr<Tensor> input, int axis) {
        const int rows = input->rows;
        const int cols = input->cols;
        const int N    = static_cast<int>(input->size);

        std::shared_ptr<Tensor> res;
        if (axis == -1) {
            res = std::make_shared<Tensor>(1, 1, input->device, input->requires_grad);
            if (input->device == Device::GPU) {
                res->fill(0.0f);
                gpu_sum_all(input->data_gpu, res->data_gpu, N);
            } else {
                float s = 0.0f;
                for (int i = 0; i < N; ++i) s += input->data[i];
                res->data[0] = s;
            }
        } else if (axis == 0) {
            res = std::make_shared<Tensor>(1, cols, input->device, input->requires_grad);
            if (input->device == Device::GPU) {
                gpu_sum_rows(input->data_gpu, res->data_gpu, rows, cols);
            } else {
                for (int c = 0; c < cols; ++c) {
                    float s = 0.0f;
                    for (int r = 0; r < rows; ++r) s += input->data[r * cols + c];
                    res->data[c] = s;
                }
            }
        } else if (axis == 1) {
            res = std::make_shared<Tensor>(rows, 1, input->device, input->requires_grad);
            if (input->device == Device::GPU) {
                gpu_sum_cols(input->data_gpu, res->data_gpu, rows, cols);
            } else {
                for (int r = 0; r < rows; ++r) {
                    float s = 0.0f;
                    for (int c = 0; c < cols; ++c) s += input->data[r * cols + c];
                    res->data[r] = s;
                }
            }
        } else {
            throw std::runtime_error("[Tensor::sum] axis debe ser -1, 0 o 1.");
        }

        if (res->requires_grad) {
            res->inputs = {input};
            res->backward_fn = [res, input, axis, rows, cols, N]() {
                if (!input->requires_grad) return;
                if (axis == -1) {
                    if (input->device == Device::GPU) {
                        Tensor bcast(rows, cols, input->device, false);
                        float scalar_g = 0.0f;
                        cudaMemcpy(&scalar_g, res->grad->data_gpu, sizeof(float), cudaMemcpyDeviceToHost);
                        gpu_fill(bcast.data_gpu, scalar_g, N);
                        gpu_add_tensors(input->grad->data_gpu, bcast.data_gpu, input->grad->data_gpu, N);
                    } else {
                        const float g = res->grad->data[0];
                        for (int i = 0; i < N; ++i) input->grad->data[i] += g;
                    }
                } else if (axis == 0) {
                    if (input->device == Device::GPU) {
                        Tensor expanded(rows, cols, input->device, false);
                        gpu_broadcast_rowvec(res->grad->data_gpu, expanded.data_gpu, rows, cols);
                        gpu_add_tensors(input->grad->data_gpu, expanded.data_gpu, input->grad->data_gpu, N);
                    } else {
                        for (int r = 0; r < rows; ++r)
                            for (int c = 0; c < cols; ++c)
                                input->grad->data[r * cols + c] += res->grad->data[c];
                    }
                } else {
                    if (input->device == Device::GPU) {
                        Tensor expanded(rows, cols, input->device, false);
                        gpu_broadcast_colvec(res->grad->data_gpu, expanded.data_gpu, rows, cols);
                        gpu_add_tensors(input->grad->data_gpu, expanded.data_gpu, input->grad->data_gpu, N);
                    } else {
                        for (int r = 0; r < rows; ++r)
                            for (int c = 0; c < cols; ++c)
                                input->grad->data[r * cols + c] += res->grad->data[r];
                    }
                }
            };
        }
        return res;
    }

    // --- MEAN diferenciable: encadena sum + multiply(1/N) a través del grafo ---
    std::shared_ptr<Tensor> Tensor::mean(std::shared_ptr<Tensor> input, int axis) {
        float count;
        if (axis == -1)     count = static_cast<float>(input->size);
        else if (axis == 0) count = static_cast<float>(input->rows);
        else                count = static_cast<float>(input->cols);

        auto s = Tensor::sum(input, axis);
        auto factor = std::make_shared<Tensor>(s->rows, s->cols, s->device, false);
        factor->fill(1.0f / count);
        return Tensor::multiply(s, factor);
    }

    // --- RESHAPE diferenciable: mismos datos, nueva forma ---
    std::shared_ptr<Tensor> Tensor::reshape(std::shared_ptr<Tensor> input, int new_rows, int new_cols) {
        if (new_rows * new_cols != static_cast<int>(input->size)) {
            throw std::runtime_error("[Tensor::reshape] Total de elementos no coincide con la nueva forma.");
        }
        auto res = std::make_shared<Tensor>(new_rows, new_cols, input->device, input->requires_grad);
        if (input->device == Device::GPU) {
            cudaMemcpy(res->data_gpu, input->data_gpu, input->size * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            std::copy(input->data, input->data + input->size, res->data);
        }
        if (res->requires_grad) {
            res->inputs = {input};
            const int orig_rows = input->rows;
            const int orig_cols = input->cols;
            res->backward_fn = [res, input, orig_rows, orig_cols]() {
                if (!input->requires_grad) return;
                if (input->device == Device::GPU) {
                    Tensor local_grad(orig_rows, orig_cols, input->device, false);
                    cudaMemcpy(local_grad.data_gpu, res->grad->data_gpu, input->size * sizeof(float), cudaMemcpyDeviceToDevice);
                    gpu_add_tensors(input->grad->data_gpu, local_grad.data_gpu, input->grad->data_gpu, static_cast<int>(input->size));
                } else {
                    for (size_t i = 0; i < input->size; ++i)
                        input->grad->data[i] += res->grad->data[i];
                }
            };
        }
        return res;
    }

    std::shared_ptr<Tensor> Tensor::from_vector(const std::vector<float>& values, int rows, int cols, Device dev, bool req_grad) {
        auto tensor = std::make_shared<Tensor>(rows, cols, dev, req_grad);
        tensor->set_data(values);
        return tensor;
    }

    // --- HELPERS DE MATRIZ PARA GRADIENTES ---

    Tensor Tensor::matmul_transB(const Tensor &other) const {
        Tensor result(rows, other.rows, device);
        if (device == Device::GPU) {
            gpu_matmul_transB(data_gpu, other.data_gpu, result.data_gpu, rows, cols, other.rows);
        } else {
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < other.rows; ++c) {
                    float sum = 0.0f;
                    for (int k = 0; k < cols; ++k) {
                        sum += data[r * cols + k] * other.data[c * other.cols + k];
                    }
                    result.data[r * result.cols + c] = sum;
                }
            }
        }
        return result;
    }

    Tensor Tensor::matmul_transA(const Tensor &other) const {
        Tensor res(cols, other.cols, device);
        if (device == Device::GPU) {
            gpu_matmul_transA(data_gpu, other.data_gpu, res.data_gpu, cols, rows, other.cols);
        } else {
            for (int r = 0; r < cols; ++r) {
                for (int c = 0; c < other.cols; ++c) {
                    float sum = 0.0f;
                    for (int k = 0; k < rows; ++k) {
                        sum += data[k * cols + r] * other.data[k * other.cols + c];
                    }
                    res.data[r * res.cols + c] = sum;
                }
            }
        }
        return res;
    }

    // --- CORE DE ENTRENAMIENTO ---

    void Tensor::backward(std::shared_ptr<Tensor> grad_output) {
        if (!grad) {
            grad = std::make_shared<Tensor>(rows, cols, device, false);
            grad->fill(0.0f);
        }

        if (grad_output) {
            ensure_same_device(shared_from_this(), grad_output, "Tensor::backward");
            if (rows != grad_output->rows || cols != grad_output->cols) {
                throw std::runtime_error("[Tensor::backward] grad_output debe tener la misma forma que el tensor.");
            }
            grad->set_data(grad_output->to_vector());
        } else {
            if (size != 1) {
                throw std::runtime_error("[Tensor::backward] grad_output es obligatorio para tensores no escalares.");
            }
            grad->fill(1.0f);
        }

        std::vector<Tensor*> order;
        std::set<Tensor*> visited;
        std::function<void(Tensor*)> build_order = [&](Tensor* node) {
            if (visited.count(node)) return;
            visited.insert(node);
            for (auto& input : node->inputs) build_order(input.get());
            order.push_back(node);
        };
        build_order(this);
        std::reverse(order.begin(), order.end());

        for (auto* node : order) {
            if (node->backward_fn) node->backward_fn();
        }
    }

    void Tensor::apply_gradient(const Tensor &grad_tensor, float lr) {
        if (device == Device::GPU) {
            gpu_apply_gradient(data_gpu, grad_tensor.data_gpu, lr, static_cast<int>(size));
        } else {
            for (size_t i = 0; i < size; ++i) {
                data[i] -= lr * grad_tensor.data[i];
            }
        }
    }

    void Tensor::zero_grad() {
        if (grad) grad->fill(0.0f);
    }

    // --- MÉTODOS DE PÉRDIDA (LOSS) ---

    float Tensor::mse_loss(const Tensor &target) {
        if (rows != target.rows || cols != target.cols) {
            throw std::runtime_error("[Tensor::mse_loss] Dimensiones incompatibles.");
        }

        if (device == Device::CPU) {
            float loss = 0.0f;
            for (size_t i = 0; i < size; ++i) {
                const float diff = data[i] - target.data[i];
                loss += diff * diff;
            }
            return loss / static_cast<float>(size);
        }

        float h_loss = 0.0f;
        float *d_loss;
        cudaMalloc(&d_loss, sizeof(float));
        cudaMemset(d_loss, 0, sizeof(float));
        gpu_mse_loss(data_gpu, target.data_gpu, d_loss, static_cast<int>(size));
        cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_loss);
        return h_loss;
    }

    Tensor Tensor::mse_backward(const Tensor &target) {
        Tensor grad_out(rows, cols, device);
        if (device == Device::GPU) {
            gpu_mse_backward(data_gpu, target.data_gpu, grad_out.data_gpu, static_cast<int>(size));
        } else {
            for (size_t i = 0; i < size; ++i) {
                grad_out.data[i] = (2.0f / static_cast<float>(size)) * (data[i] - target.data[i]);
            }
        }
        return grad_out;
    }

    std::shared_ptr<Tensor> Tensor::softmax() {
        auto probs = std::make_shared<Tensor>(rows, cols, device, false);
        std::vector<float> host_input = to_vector();
        std::vector<float> host_probs(size);
        for (int r = 0; r < rows; ++r) {
            const int base = r * cols;
            float row_max = host_input[base];
            for (int c = 1; c < cols; ++c) {
                row_max = std::max(row_max, host_input[base + c]);
            }

            float sum_exp = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float e = std::exp(host_input[base + c] - row_max);
                host_probs[base + c] = e;
                sum_exp += e;
            }

            for (int c = 0; c < cols; ++c) {
                host_probs[base + c] /= sum_exp;
            }
        }

        if (probs->device == Device::GPU) {
            cudaMemcpy(probs->data_gpu, host_probs.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            std::copy(host_probs.begin(), host_probs.end(), probs->data);
        }

        return probs;
    }

    float Tensor::cross_entropy_with_target(const Tensor &target) {
        if (rows != target.rows || cols != target.cols) {
            throw std::runtime_error("[Tensor::cross_entropy_with_target] Dimensiones incompatibles.");
        }

        std::vector<float> pred_host = to_vector();
        std::vector<float> target_host = target.to_vector();

        const float eps = 1e-8f;
        float loss = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            float p = std::max(pred_host[i], eps);
            loss += -target_host[i] * std::log(p);
        }

        return loss / static_cast<float>(rows);
    }

    // --- UTILIDADES ---

    void Tensor::fill(float value) {
        if (device == Device::GPU) gpu_fill(data_gpu, value, static_cast<int>(size));
        else std::fill(data, data + size, value);
    }

    void Tensor::fill_random(float low, float high) {
        if (device == Device::GPU) {
            unsigned long long seed = static_cast<unsigned long long>(std::time(nullptr));
            gpu_fill_random(data_gpu, static_cast<int>(size), low, high, seed);
        } else {
            std::mt19937 generator(static_cast<unsigned int>(std::time(nullptr)));
            std::uniform_real_distribution<float> distribution(low, high);
            for (size_t i = 0; i < size; ++i) {
                data[i] = distribution(generator);
            }
        }
    }

    void Tensor::set_data(const std::vector<float>& values) {
        if (values.size() != size) {
            throw std::runtime_error("[Tensor::set_data] El numero de valores no coincide con la forma del tensor.");
        }

        if (device == Device::GPU) {
            cudaMemcpy(data_gpu, values.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            std::copy(values.begin(), values.end(), data);
        }
    }

    std::vector<float> Tensor::to_vector() const {
        std::vector<float> host_values(size);
        if (device == Device::GPU) {
            cudaMemcpy(host_values.data(), data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            std::copy(data, data + size, host_values.begin());
        }
        return host_values;
    }

    float Tensor::item() const {
        if (size != 1) {
            throw std::runtime_error("[Tensor::item] Solo disponible para tensores escalares.");
        }
        return to_vector()[0];
    }

    void Tensor::print(int limit) const {
        const int max_items = std::max(0, limit);
        const std::vector<float> values = to_vector();
        const size_t count = std::min<size_t>(size, static_cast<size_t>(max_items));
        std::cout << "Tensor(" << rows << "x" << cols << "): [";
        for (size_t i = 0; i < count; ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << values[i];
        }
        if (size > count) {
            std::cout << ", ...";
        }
        std::cout << "]" << std::endl;
    }

    float Tensor::accuracy(const Tensor &target) {
        if (rows != target.rows || cols != target.cols) {
            throw std::runtime_error("[Tensor::accuracy] Dimensiones incompatibles.");
        }

        std::vector<float> pred_host = to_vector();
        std::vector<float> target_host = target.to_vector();

        int correct = 0;
        for (int r = 0; r < rows; ++r) {
            int pred_idx = 0;
            int target_idx = 0;
            float pred_max = pred_host[r * cols];
            float target_max = target_host[r * cols];

            for (int c = 1; c < cols; ++c) {
                const float pv = pred_host[r * cols + c];
                const float tv = target_host[r * cols + c];
                if (pv > pred_max) {
                    pred_max = pv;
                    pred_idx = c;
                }
                if (tv > target_max) {
                    target_max = tv;
                    target_idx = c;
                }
            }

            if (pred_idx == target_idx) {
                ++correct;
            }
        }

        return static_cast<float>(correct) / static_cast<float>(rows);
    }

    void Tensor::to_cpu() {
        if (device == Device::GPU && data_gpu) {
            data = new float[size];
            cudaMemcpy(data, data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(data_gpu);
            data_gpu = nullptr;
            device = Device::CPU;
        }
    }

    void Tensor::to_gpu() {
        if (device == Device::CPU && data) {
            DeviceManager::initialize();
            if (DeviceManager::should_use_managed_memory()) {
                cudaMallocManaged(&data_gpu, size * sizeof(float));
                std::copy(data, data + size, data_gpu);
            } else {
                cudaMalloc(&data_gpu, size * sizeof(float));
                cudaMemcpy(data_gpu, data, size * sizeof(float), cudaMemcpyHostToDevice);
            }
            delete[] data;
            data = nullptr;
            device = Device::GPU;
        }
    }

} // namespace ttensor