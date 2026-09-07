#include "cublas_context.hpp"

#include "device_manager.hpp"

namespace ttensor {

CublasContext& CublasContext::instance() {
    // La inicialización thread-safe de una variable estática local evita una
    // carrera entre dos primeras operaciones matmul desde Python.
    static CublasContext context;
    return context;
}

cublasHandle_t CublasContext::handle() {
    CublasContext& context = instance();
    std::call_once(context.init_once_, [&context]() {
        DeviceManager::initialize();
        CHECK_CUBLAS(cublasCreate(&context.handle_));
    });
    return context.handle_;
}

CublasContext::~CublasContext() {
    if (handle_ != nullptr) {
        // El destructor no puede propagar excepciones. En condiciones normales
        // cublasDestroy devuelve SUCCESS; el resultado se ignora durante la
        // finalización del proceso porque no hay un canal seguro para reportarlo.
        cublasDestroy(handle_);
        handle_ = nullptr;
    }
}

} // namespace ttensor