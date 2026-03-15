cmake_minimum_required(VERSION 3.18)

project(T_Tensor LANGUAGES CXX CUDA)

# --- CONFIGURACIÓN DE COMPILADOR ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Soporte para Visual Studio y CUDA 12.8
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

option(TTENSOR_CUDA_NATIVE "Compilar solo para la GPU del host (desarrollo local)." OFF)
set(TTENSOR_CUDA_FAT_ARCHS "75;80;86;89;90" CACHE STRING "Lista de SM para fat binary cuando TTENSOR_CUDA_NATIVE=OFF")

if(TTENSOR_CUDA_NATIVE)
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    set(CMAKE_CUDA_ARCHITECTURES ${TTENSOR_CUDA_FAT_ARCHS})
endif()

message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

# 1. BUSCAR DEPENDENCIAS
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
# CUDAToolkit es la forma moderna de encontrar curand, cudart, etc.
find_package(CUDAToolkit REQUIRED)

# 2. ARCHIVOS FUENTE
# El core usa implementaciones CUDA (.cu)
file(GLOB_RECURSE CORE_SOURCES "src/*.cu") 
set(BINDING_SOURCE "src/bindings.cpp")

# 3. LIBRERÍA CORE (STATIC)
add_library(ttensor_core STATIC ${CORE_SOURCES})

set_target_properties(ttensor_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ttensor_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Enlazamos curand a la librería core
target_link_libraries(ttensor_core PUBLIC 
    CUDA::curand 
    CUDA::cudart
)

# 4. MÓDULO DE PYTHON (pybind11)
pybind11_add_module(ttensor_python ${BINDING_SOURCE})

target_link_libraries(ttensor_python PRIVATE 
    ttensor_core 
    pybind11::module
    Python3::Python
)

set_target_properties(ttensor_python PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "ttensor"
)

# 5. EJECUTABLE DE PRUEBA (C++)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/main.cpp")
    add_executable(ttensor_test examples/main.cpp)
    target_link_libraries(ttensor_test PRIVATE ttensor_core)
endif()

# 6. AJUSTES DE MSVC
if(MSVC)
    target_compile_options(ttensor_python PRIVATE /EHsc /bigobj /W3)
    if(TARGET ttensor_test)
        target_compile_options(ttensor_test PRIVATE /EHsc /W4)
    endif()
endif()