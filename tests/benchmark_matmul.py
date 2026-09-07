"""Benchmark corto de matmul CPU (fallback) frente a cuBLAS en GPU.

La conversión a ``tolist`` sincroniza CUDA antes de parar el cronómetro, por lo
que cada medición incluye la ejecución real del producto y no sólo el coste de
encolarlo en el stream por defecto.
"""

from __future__ import annotations

import argparse
import time

from ttensor import Device, DeviceManager, Tensor


def benchmark(device: Device, rows: int, inner: int, cols: int, repeats: int) -> float:
    left = Tensor(rows, inner, device, False)
    right = Tensor(inner, cols, device, False)
    left.fill_random()
    right.fill_random()

    # Warm-up: incluye la inicialización perezosa de CUDA y, en GPU, cuBLAS.
    Tensor.matmul(left, right).tolist()

    start = time.perf_counter()
    for _ in range(repeats):
        Tensor.matmul(left, right).tolist()
    return (time.perf_counter() - start) * 1000.0 / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--inner", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    DeviceManager.initialize()
    if DeviceManager.device_count() == 0:
        raise SystemExit("No hay una GPU CUDA disponible para comparar cuBLAS.")

    cpu_ms = benchmark(Device.CPU, args.rows, args.inner, args.cols, args.repeats)
    gpu_ms = benchmark(Device.GPU, args.rows, args.inner, args.cols, args.repeats)
    print(f"CPU fallback: {cpu_ms:.3f} ms/op")
    print(f"GPU cuBLAS:   {gpu_ms:.3f} ms/op")
    print(f"Aceleracion:  {cpu_ms / gpu_ms:.2f}x")


if __name__ == "__main__":
    main()