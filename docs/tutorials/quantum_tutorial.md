# 量子力学计算教程

本教程介绍如何使用 PySymmetry 进行量子力学计算。

## 目录

1. [量子态表示](#量子态表示)
2. [量子测量](#量子测量)
3. [密度矩阵](#密度矩阵)
4. [量子纠缠](#量子纠缠)
5. [时间演化](#时间演化)

---

## 量子态表示

### Dirac 符号

PySymmetry 使用 Dirac 符号表示量子态：

```python
from PySymmetry.phys.quantum.states import Ket, Bra, basis_state
import numpy as np

# 计算基底
ket0 = basis_state(0, 2)  # |0⟩
ket1 = basis_state(1, 2)  # |1⟩

# 从向量创建
vec = np.array([1, 0], dtype=complex)
ket = Ket(vec)

# 从标签创建
ket = Ket('0')  # 等价于 |0⟩
ket = Ket('1')  # 等价于 |1⟩
```

### 态叠加

```python
from PySymmetry.phys.quantum.states import Ket
import numpy as np

# 创建叠加态 (|0⟩ + |1⟩)/√2
ket0 = Ket('0')
ket1 = Ket('1')

plus = (ket0 + ket1) / np.sqrt(2)
print(f"叠加态: {plus.vector}")
# [0.70710678+0.j, 0.70710678+0.j]

# 归一化
normalized = plus.normalize()
```

### 内积

```python
from PySymmetry.phys.quantum.states import Ket, Bra

ket0 = Ket('0')
ket1 = Ket('1')

# ⟨0|1⟩ = 0 (正交)
bra0 = Bra(ket0)
inner = bra0.inner_product(ket1)
print(f"⟨0|1⟩ = {inner}")  # 0

# ⟨0|0⟩ = 1
inner = bra0.inner_product(ket0)
print(f"⟨0|0⟩ = {inner}")  # 1
```

---

## 量子测量

### 概率分布

```python
from PySymmetry.phys.quantum.states import StateVector
import numpy as np

# 创建叠加态
vec = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
sv = StateVector(vec)

# 获取测量概率
probs = sv.probabilities
print(f"测量 |0⟩ 的概率: {probs[0]}")  # 0.5
print(f"测量 |1⟩ 的概率: {probs[1]}")  # 0.5
```

### 模拟测量

```python
import numpy as np

# 创建叠加态
vec = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
sv = StateVector(vec)

# 多次测量（蒙特卡洛）
results = {'0': 0, '1': 0}
n_samples = 1000

for _ in range(n_samples):
    idx, prob = sv.measure()
    results[str(idx)] += 1

print(f"|0⟩ 出现次数: {results['0']}")  # 约500
print(f"|1⟩ 出现次数: {results['1']}")  # 约500
```

---

## 密度矩阵

### 从纯态创建

```python
from PySymmetry.phys.quantum.states import Ket, DensityMatrix
import numpy as np

# 从 Ket 创建
ket = Ket('0')
rho = DensityMatrix(ket)

# 从向量创建
vec = np.array([1, 0], dtype=complex)
rho = DensityMatrix(vec)

# 检查性质
print(f"维度: {rho.dimension}")        # 2
print(f"是否为纯态: {rho.is_pure}")     # True
print(f"纯度 Tr(ρ²): {rho.purity}")   # 1.0
print(f"熵: {rho.entropy()}")          # 0.0
```

### 混合态

```python
from PySymmetry.phys.quantum.states import DensityMatrix
import numpy as np

# 最大混合态 (完全随机)
rho_mixed = np.eye(4, dtype=complex) / 4
dm = DensityMatrix(rho_mixed, is_pure=False)

print(f"纯度: {dm.purity}")   # 0.25
print(f"熵: {dm.entropy()}")   # 2.0 (最大熵 = log2(4))
```

### 期望值

```python
from PySymmetry.phys.quantum.states import DensityMatrix, Ket
import numpy as np

# 创建态 |+⟩ = (|0⟩ + |1⟩)/√2
plus = (Ket('0') + Ket('1')) / np.sqrt(2)
rho = DensityMatrix(plus)

# Pauli Z 期望值
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
exp_z = rho.expectation(sigma_z)
print(f"⟨σz⟩ = {exp_z}")  # 0

# Pauli X 期望值
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
exp_x = rho.expectation(sigma_x)
print(f"⟨σx⟩ = {exp_x}")  # 1

# 方差
var_x = rho.variance(sigma_x)
print(f"Var(σx) = {var_x}")  # 0
```

### 约化密度矩阵

```python
from PySymmetry.phys.quantum.states import DensityMatrix, Ket, tensor_product
import numpy as np

# 两 qubit 系统 |00⟩
ket00 = tensor_product(Ket('0'), Ket('0'))
rho_full = DensityMatrix(ket00)

# 对第一个 qubit 求迹
rho_reduced = rho_full.partial_trace(subsystems=[0], dims=[2, 2])

print(f"约化态维度: {rho_reduced.dimension}")  # 2
print(f"约化态熵: {rho_reduced.entropy()}")     # 0 (纯态)
```

---

## 量子纠缠

### Bell 态

```python
from PySymmetry.phys.quantum.states import bell_state
import numpy as np

# 四个 Bell 态
phi_plus = bell_state(0)   # |Φ+⟩ = (|00⟩ + |11⟩)/√2
phi_minus = bell_state(1)  # |Φ-⟩ = (|00⟩ - |11⟩)/√2
psi_plus = bell_state(2)   # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
psi_minus = bell_state(3)  # |Ψ-⟩ = (|01⟩ - |10⟩)/√2

print(f"|Φ+⟩ 向量:\n{phi_plus.vector}")
```

### GHZ 态

```python
from PySymmetry.phys.quantum.states import ghz_state

# 3-qubit GHZ 态
ghz = ghz_state(3)
# |GHZ⟩ = (|000⟩ + |111⟩)/√2
```

### W 态

```python
from PySymmetry.phys.quantum.states import w_state

# 3-qubit W 态
w = w_state(3)
# |W⟩ = (|100⟩ + |010⟩ + |001⟩)/√3
```

### 纠缠度量（保真度）

```python
from PySymmetry.phys.quantum.states import DensityMatrix, Ket, bell_state
import numpy as np

# 两个 Bell 态的保真度
phi1 = bell_state(0)
phi2 = bell_state(0)
dm1 = DensityMatrix(phi1)
dm2 = DensityMatrix(phi2)

f = dm1.fidelity(dm2)
print(f"F(|Φ+⟩, |Φ+⟩) = {f}")  # 1.0 (相同态)

# 不同 Bell 态的保真度
psi = bell_state(2)
dm_psi = DensityMatrix(psi)
f = dm1.fidelity(dm_psi)
print(f"F(|Φ+⟩, |Ψ+⟩) = {f}")  # 0.0 (正交)
```

---

## 时间演化

### 薛定谔方程

对于哈密顿量 H，时间演化为：
```
|ψ(t)⟩ = exp(-iHt/ℏ) |ψ(0)⟩
```

```python
from PySymmetry.core.matrix.operations import MatrixOperations
from PySymmetry.phys.quantum.states import Ket
import numpy as np

# Pauli X 作为哈密顿量
H = np.array([[0, 1], [1, 0]], dtype=complex)

# 初始态 |0⟩
psi0 = Ket('0')

# 时间 t = π/2
t = np.pi / 2
U = MatrixOperations.matrix_exponential(-1j * H * t)

# 时间演化
psi_t = Ket(U @ psi0.vector)

print(f"时间演化后的态: {psi_t.vector}")
# 应该等于 |1⟩ (翻转)
```

### 旋转门

```python
from PySymmetry.core.matrix.factory import MatrixFactory
from PySymmetry.phys.quantum.states import Ket
import numpy as np

# 初始态 |0⟩
psi = Ket('0')

# Hadamard 门
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
psi_H = Ket(H @ psi.vector)

# R_x(θ) 门
def rx(theta):
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

psi_rx = Ket(rx(np.pi) @ psi.vector)
# R_x(π) 将 |0⟩ -> |1⟩
```

### 量子电路模拟

```python
from PySymmetry.phys.quantum.states import Ket, tensor_product
import numpy as np

# 初始态 |00⟩
psi = tensor_product(Ket('0'), Ket('0'))

# Pauli X 门（NOT）
X = np.array([[0, 1], [1, 0]], dtype=complex)

# 在第二个 qubit 上应用 X
I = np.eye(2, dtype=complex)
CNOT_like = np.kron(I, X)

psi_final = Ket(CNOT_like @ psi.vector)
# |00⟩ -> |01⟩
```

---

## 实用示例

### 量子态层析

```python
from PySymmetry.phys.quantum.states import DensityMatrix, Ket
import numpy as np

# 假设我们有一个未知态
# 通过测量重构密度矩阵

# Pauli 基测量
sigma_i = [
    np.array([[1, 0], [0, 1]], dtype=complex),  # I
    np.array([[0, 1], [1, 0]], dtype=complex),  # X
    np.array([[0, -1j], [1j, 0]], dtype=complex),  # Y
    np.array([[1, 0], [0, -1]], dtype=complex)  # Z
]

# 假设测量结果
measurements = {'I': 1, 'X': 0, 'Y': 0, 'Z': 1}

# 重构密度矩阵
rho = np.zeros((2, 2), dtype=complex)
for i, m in enumerate(measurements.values()):
    rho += m * sigma_i[i]
rho = rho / 2

dm = DensityMatrix(rho)
print(f"重构的密度矩阵:\n{dm.matrix}")
```

### 量子通道

```python
from PySymmetry.phys.quantum.states import DensityMatrix, Ket
import numpy as np

# Bit-flip 通道
def bit_flip_channel(rho, p):
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    return (1-p)*rho + p*X@rho@X

# 应用到纯态 |0⟩
ket = Ket('0')
rho = DensityMatrix(ket)

# 10% 的翻转概率
rho_depol = bit_flip_channel(rho.matrix, 0.1)
dm_depol = DensityMatrix(rho_depol)

print(f"退相干后纯度: {dm_depol.purity}")  # < 1
```
