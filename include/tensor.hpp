#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace ttensor {

enum class Device { CPU, GPU };

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    int rows, cols;
    size_t size;
    Device device;
    float* data;      
    float* data_gpu;

    // --- MIEMBROS DE AUTOGRAD ---
    std::shared_ptr<Tensor> grad; 
    bool requires_grad;
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::function<void()> backward_fn;

    // Constructores y Destructor
    Tensor(int r, int c, Device dev = Device::GPU, bool req_grad = false);
    Tensor(const Tensor& other);
    ~Tensor();

    // Gestión de memoria
    void allocate();
    void deallocate();
    void to_gpu();
    void to_cpu();
    
    // Operaciones de Grafo (Estáticas para Autograd)
    static std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    static std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    static std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    static std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> input);
    static std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> input);
    static std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> input);
    static std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    static std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> input, int axis = -1);
    static std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> input, int axis = -1);
    static std::shared_ptr<Tensor> log(std::shared_ptr<Tensor> input);
    static std::shared_ptr<Tensor> exp(std::shared_ptr<Tensor> input);
    static std::shared_ptr<Tensor> reshape(std::shared_ptr<Tensor> input, int new_rows, int new_cols);
    static std::shared_ptr<Tensor> from_vector(const std::vector<float>& values, int rows, int cols, Device dev = Device::GPU, bool req_grad = false);
    
    // Autograd
    void backward(std::shared_ptr<Tensor> grad_output = nullptr);
    void zero_grad();

    // Utilidades de Tensor
    void fill(float value);
    void fill_random(float low = -1.0f, float high = 1.0f);
    void set_data(const std::vector<float>& values);
    std::vector<float> to_vector() const;
    float item() const;
    void print(int limit = 10) const;
    float accuracy(const Tensor& target);

    // Métodos de cálculo de pérdida (utilizados por losses.cu)
    float mse_loss(const Tensor& target);
    Tensor mse_backward(const Tensor& target);
    
    // Métodos para CrossEntropy (requieren implementación en kernels)
    std::shared_ptr<Tensor> softmax();
    float cross_entropy_with_target(const Tensor& target);

    // Primitivas de bajo nivel y transposiciones para gradientes
    void apply_gradient(const Tensor& grad_tensor, float learning_rate);
    Tensor matmul_transB(const Tensor& other) const;
    Tensor matmul_transA(const Tensor& other) const;
};

} // namespace ttensor

#endif