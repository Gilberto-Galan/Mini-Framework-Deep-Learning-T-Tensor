# T-Tensor

Mini-Framework de tensores con autograd, capas tipo `nn`, optimizadores y soporte CUDA.

## Contenido

1. Vision general
2. Instalacion
3. Importacion y organizacion
4. Modulo `device` (GPU/CPU)
5. Modulo `tensor` (autograd y operaciones)
6. Modulo `nn` (capas y modelos)
7. Modulo `optim` (SGD y Adam)
8. Modulo `loss` (funciones de perdida)
9. Modulo `data` (dataset y dataloader)
10. Guia de entrenamiento (paso a paso)
11. Ejemplos listos para copiar
12. Buenas practicas y errores comunes

---

## 1. Vision general

`T-Tensor` incluye:

- Tensores en CPU y GPU (`Device.CPU`, `Device.GPU`)
- Autograd con grafo dinamico y `backward()`
- Operaciones diferenciables (matmul, add, sub, mul, relu, sigmoid, softmax, log, exp, sum, mean, reshape)
- Capas (`Linear`, `ReLU`, `Sigmoid`, `Softmax`, `Sequential`)
- Optimizadores (`SGD`, `Adam`)
- Perdidas (`MSELoss`, `CrossEntropyLoss`)
- Carga de datos (`CSVDataset`, `DataLoader`)
- Gestion de hardware con `DeviceManager`

---

## 2. Instalacion

Desde la raiz del proyecto:

```bash
pip install .
```

Si usas entorno virtual en Windows:

```powershell
.\venv\Scripts\Activate.ps1
pip install .
```

---

## 3. Importacion y organizacion tipo PyTorch

En `T-Tensor`, todo se importa desde `ttensor`, y lo puedes organizar mentalmente igual:

```python
import ttensor
from ttensor import (
	Device, Tensor, DeviceManager,
	Module, Linear, ReLU, Sigmoid, Softmax, Sequential,
	Optimizer, SGD, Adam,
	MSELoss, CrossEntropyLoss,
	Dataset, CSVDataset, DataLoader,
)
```

Equivalencia conceptual:

- `ttensor.Tensor` -> parecido a `torch.Tensor`
- `Linear/ReLU/Sigmoid/Softmax/Sequential` -> parecido a `torch.nn`
- `SGD/Adam` -> parecido a `torch.optim`
- `CSVDataset/DataLoader` -> parecido a `torch.utils.data`

---

## 4. Modulo `device` (GPU/CPU)

### Enum `Device`

- `Device.CPU`
- `Device.GPU`

### Clase `DeviceManager`

Metodos principales:

- `initialize(preferred_device_id=0)`
- `set_device(device_id)`
- `device_count()`
- `active_device_id()`
- `available_devices()`
- `get_info()`
- `update_memory_info()`
- `has_enough_vram(required_bytes)`
- `set_prefer_managed_memory(enabled)`
- `should_use_managed_memory()`
- `print_report()`

Ejemplo:

```python
from ttensor import DeviceManager

DeviceManager.initialize()
DeviceManager.print_report()

info = DeviceManager.get_info()
print("GPU:", info.name)
print("VRAM total:", info.total_memory)
print("VRAM libre:", info.free_memory)
```

---

## 5. Modulo `tensor` (autograd y operaciones)

### Crear tensores

```python
from ttensor import Tensor, Device

x = Tensor(4, 3, Device.GPU, True)  # rows, cols, device, requires_grad
x.fill_random(-0.1, 0.1)

y = Tensor.from_list([
	1.0, 2.0, 3.0,
	4.0, 5.0, 6.0,
], rows=2, cols=3, device=Device.CPU, requires_grad=False)
```

### Utilidades

- `fill(value)`
- `fill_random(low, high)`
- `set_data(values)`
- `tolist()`
- `item()` para tensores `1x1`
- `print(limit=10)`
- `to_gpu()` / `to_cpu()`
- `shape`
- `requires_grad`
- `grad`

### Operaciones diferenciables

Estilo funcional:

```python
z = Tensor.matmul(a, b)
z = Tensor.add(a, b)
z = Tensor.sub(a, b)
z = Tensor.mul(a, b)

z = Tensor.relu(x)
z = Tensor.sigmoid(x)
z = Tensor.softmax(x)
z = Tensor.log(x)
z = Tensor.exp(x)

s_all = Tensor.sum(x, axis=-1)   # escalar
s0 = Tensor.sum(x, axis=0)       # (1, cols)
s1 = Tensor.sum(x, axis=1)       # (rows, 1)

m_all = Tensor.mean(x, axis=-1)

r = Tensor.reshape(x, new_rows=2, new_cols=6)
```

Tambien con operadores:

```python
z = a + b
z = a - b
z = a * b          # elemento a elemento
z = a @ b          # matmul
```

### Backpropagation

```python
loss.backward()        # si loss es escalar

# Si el tensor no es escalar, pasa gradiente explicito:
# output.backward(grad_output)
```

Reiniciar gradientes:

```python
param.zero_grad()
```

---

## 6. Modulo `nn` (capas y modelos)

### `Module`

Base para capas y modelos. API:

- `forward(input)`
- `__call__(input)` (alias de forward)
- `parameters()`
- `zero_grad()`
- `save(path)`
- `load(path)`

### `Linear(in_features, out_features, device=Device.GPU)`

Hace `X @ W + b`.

### Activaciones

- `ReLU()`
- `Sigmoid()`
- `Softmax()`

### `Sequential([...])`

Encadena capas.

Ejemplo:

```python
from ttensor import Sequential, Linear, ReLU, Sigmoid

model = Sequential([
	Linear(2, 16),
	ReLU(),
	Linear(16, 1),
	Sigmoid(),
])

pred = model(x)
```

---

## 7. Modulo `optim` (SGD y Adam)

### `SGD(params, lr)`

```python
opt = SGD(model.parameters(), lr=0.01)
```

### `Adam(params, lr, beta1=0.9, beta2=0.999, eps=1e-8)`

```python
opt = Adam(model.parameters(), lr=0.001)
```

Metodos comunes:

- `step()`
- `zero_grad()`

---

## 8. Modulo `loss` (funciones de perdida)

### `MSELoss`

Ideal para regresion.

```python
criterion = MSELoss()
loss = criterion.forward(pred, target)
```

### `CrossEntropyLoss`

Ideal para clasificacion multiclase con targets one-hot.

```python
criterion = CrossEntropyLoss()
loss = criterion.forward(logits_or_probs, target_one_hot)
```

---

## 9. Modulo `data` (dataset y dataloader)

### `CSVDataset(path, label_cols, device=Device.GPU)`

- `path`: archivo CSV
- `label_cols`: cuantas columnas finales son etiqueta(s)

### `DataLoader(dataset, batch_size, shuffle=True)`

Metodos:

- `reset()`
- `has_next()`
- `next_batch()` -> retorna `(X_batch, y_batch)`

Ejemplo:

```python
from ttensor import CSVDataset, DataLoader, Device

dataset = CSVDataset("data/train.csv", label_cols=1, device=Device.GPU)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

loader.reset()
while loader.has_next():
	xb, yb = loader.next_batch()
	# entrenar
```

---

## 10. Guia de entrenamiento (paso a paso)

Plantilla general:

```python
# 1) Definir datos
# 2) Definir modelo
# 3) Definir loss y optimizador
# 4) Loop: forward -> loss -> zero_grad -> backward -> step

for epoch in range(num_epochs):
	pred = model(x)
	loss = criterion.forward(pred, y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
```

---

## 11. Ejemplos listos para copiar

### 11.1 XOR (clasificacion binaria)

```python
from ttensor import Tensor, Device, Sequential, Linear, ReLU, Sigmoid, Adam, MSELoss

X = Tensor.from_list([
	0, 0,
	0, 1,
	1, 0,
	1, 1,
], rows=4, cols=2, device=Device.GPU, requires_grad=False)

y = Tensor.from_list([
	0,
	1,
	1,
	0,
], rows=4, cols=1, device=Device.GPU, requires_grad=False)

model = Sequential([
	Linear(2, 8),
	ReLU(),
	Linear(8, 1),
	Sigmoid(),
])

criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.05)

for epoch in range(2000):
	pred = model(X)
	loss = criterion.forward(pred, y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	if epoch % 200 == 0:
		print(f"epoch={epoch} loss={loss.item():.6f}")

print("pred:", model(X).tolist())
```

### 11.2 Clasificacion multiclase (con Softmax + CrossEntropy)

```python
from ttensor import Sequential, Linear, ReLU, Softmax, CrossEntropyLoss, Adam

# x: (batch, features)
# y: (batch, num_classes) one-hot

model = Sequential([
	Linear(4, 32),
	ReLU(),
	Linear(32, 3),
	Softmax(),
])

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

pred = model(x)
loss = criterion.forward(pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 12. Buenas practicas y errores comunes

### Buenas practicas

- Usa `DeviceManager.initialize()` al inicio para verificar hardware.
- Mantiene `X`, `y` y el modelo en el mismo `Device`.
- Llama `optimizer.zero_grad()` antes de `backward()` en cada iteracion.
- En clasificacion multiclase, usa targets one-hot para `CrossEntropyLoss`.
- Verifica formas (`shape`) con frecuencia.

### Errores comunes

- Mezclar tensores CPU y GPU en una operacion.
- Usar `item()` en tensores que no son `1x1`.
- Llamar `backward()` sin gradiente de salida para tensores no escalares.
- Dimensiones incompatibles en `matmul`:
  - Si `a` es `(m, n)`, `b` debe ser `(n, k)`.

---

## Estado del framework

T-Tensor ya cubre el flujo para MLPs:

- Tensores + autograd
- Modulos y modelos
- Optimizadores
- Funciones de perdida
- Data pipeline
- Ejecucion CUDA con seleccion automatica de capacidades de GPU

