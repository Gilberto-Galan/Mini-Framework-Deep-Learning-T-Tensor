"""Pruebas unitarias base para operaciones de Tensor y autograd."""

from __future__ import annotations

import pytest

from ttensor import Device
from ttensor import DeviceManager
from ttensor import Tensor


def _has_gpu() -> bool:
    try:
        DeviceManager.initialize()
        return DeviceManager.device_count() > 0
    except Exception:
        return False


def test_tensor_add_cpu() -> None:
    a = Tensor.from_list([1.0, 2.0, 3.0, 4.0], rows=2, cols=2, device=Device.CPU, requires_grad=False)
    b = Tensor.from_list([10.0, 20.0, 30.0, 40.0], rows=2, cols=2, device=Device.CPU, requires_grad=False)

    c = Tensor.add(a, b)

    assert c.tolist() == [11.0, 22.0, 33.0, 44.0]


def test_autograd_square_grad_cpu() -> None:
    x = Tensor.from_list([3.0], rows=1, cols=1, device=Device.CPU, requires_grad=True)

    y = Tensor.mul(x, x)  # f(x) = x^2
    y.backward()

    assert x.grad is not None
    assert x.grad.item() == pytest.approx(6.0, rel=1e-6, abs=1e-6)


def test_matmul_cpu_gpu_consistency() -> None:
    if not _has_gpu():
        pytest.skip("No hay GPU CUDA disponible en este entorno")

    values_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    values_b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    a_cpu = Tensor.from_list(values_a, rows=2, cols=3, device=Device.CPU, requires_grad=False)
    b_cpu = Tensor.from_list(values_b, rows=3, cols=2, device=Device.CPU, requires_grad=False)
    out_cpu = Tensor.matmul(a_cpu, b_cpu).tolist()

    a_gpu = Tensor.from_list(values_a, rows=2, cols=3, device=Device.GPU, requires_grad=False)
    b_gpu = Tensor.from_list(values_b, rows=3, cols=2, device=Device.GPU, requires_grad=False)
    out_gpu = Tensor.matmul(a_gpu, b_gpu).tolist()

    assert len(out_cpu) == len(out_gpu)
    for cpu_v, gpu_v in zip(out_cpu, out_gpu):
        assert gpu_v == pytest.approx(cpu_v, rel=1e-5, abs=1e-5)
