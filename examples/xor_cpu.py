"""Ejemplo base: entrenamiento XOR en CPU con T-Tensor.

Uso:
  python examples/xor_cpu.py
"""

from ttensor import Adam
from ttensor import Device
from ttensor import Linear
from ttensor import MSELoss
from ttensor import ReLU
from ttensor import Sequential
from ttensor import Sigmoid
from ttensor import Tensor


def main() -> None:
    x = Tensor.from_list(
        [
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ],
        rows=4,
        cols=2,
        device=Device.CPU,
        requires_grad=False,
    )

    y = Tensor.from_list(
        [0.0, 1.0, 1.0, 0.0],
        rows=4,
        cols=1,
        device=Device.CPU,
        requires_grad=False,
    )

    model = Sequential(
        [
            Linear(2, 8, Device.CPU),
            ReLU(),
            Linear(8, 1, Device.CPU),
            Sigmoid(),
        ]
    )

    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.05)

    for epoch in range(1200):
        pred = model(x)
        loss = criterion.forward(pred, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"epoch={epoch} loss={loss.item():.6f}")

    out = model(x)
    print("\nPredicciones finales:")
    out.print(limit=8)


if __name__ == "__main__":
    main()
