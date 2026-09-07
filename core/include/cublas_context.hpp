#ifndef TTENSOR_CUBLAS_CONTEXT_HPP
#define TTENSOR_CUBLAS_CONTEXT_HPP

#include <cublas_v2.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace ttensor {

// Convierte el estado de cuBLAS en una excepción de C++ sin ocultar la llamada
// que falló. El estado numérico resulta útil incluso cuando la versión de
// cuBLAS no expone una función de texto compatible.
#define CHECK_CUBLAS(call)                                                       \
    do {                                                                        \
        const cublasStatus_t status = (call);                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            throw std::runtime_error(                                           \
                std::string("cuBLAS error (status=") +                        \
                std::to_string(static_cast<int>(status)) + ") en " #call);     \
        }                                                                       \
    } while (false)

// Un único handle por proceso evita pagar cublasCreate/cublasDestroy en cada
// matmul. El handle se crea bajo demanda, después de seleccionar el dispositivo
// CUDA activo, y se destruye automáticamente al finalizar el runtime.
class CublasContext {
public:
    static cublasHandle_t handle();

    CublasContext(const CublasContext&) = delete;
    CublasContext& operator=(const CublasContext&) = delete;

private:
    CublasContext() = default;
    ~CublasContext();

    static CublasContext& instance();

    cublasHandle_t handle_ = nullptr;
    std::once_flag init_once_;
};

} // namespace ttensor

#endif