# Quantum States API

量子态模块 (`phys.quantum.states`) 提供量子态的表示和操作。

## 目录

- [Ket (右矢)](#ket-右矢)
- [Bra (左矢)](#bra-左矢)
- [StateVector (态向量)](#statevector-态向量)
- [DensityMatrix (密度矩阵)](#densitymatrix-密度矩阵)
- [辅助函数](#辅助函数)

---

## Ket (右矢)

表示量子态的 Dirac 符号 |ψ⟩。

### 创建

```python
from PySymmetry.phys.quantum.states import Ket
import numpy as np

# 从数组创建
vec = np.array([1, 0], dtype=complex)
ket = Ket(vec)

# 从整数创建（维度，默认第一个分量=1）
ket = Ket(2)  # |0⟩

# 从标签创建
ket = Ket('0')  # |0⟩
ket = Ket('1')  # |1⟩
```

### 属性

```python
ket.vector       # 返回态向量副本
ket.dimension    # 希尔伯特空间维度
ket.hilbert_space  # 关联的希尔伯特空间
```

### 方法

#### `norm()`

返回态的范数 ⟨ψ|ψ⟩^{1/2}。

```python
n = ket.norm()  # float
```

#### `normalize()`

归一化态。

```python
normalized_ket = ket.normalize()
```

#### `copy()`

复制态。

```python
copied = ket.copy()
```

#### `to_vector()`

转换为复向量。

```python
vec = ket.to_vector()  # np.ndarray
```

### 运算

```python
# 加法（态叠加）
ket0 = Ket('0')
ket1 = Ket('1')
result = ket0 + ket1

# 标量乘法
result = ket * 2
result = 2 * ket

# 取负
result = -ket
```

---

## Bra (左矢)

表示量子态的 Dirac 符号 ⟨ψ|，是 Ket 的对偶空间元素。

### 创建

```python
from PySymmetry.phys.quantum.states import Bra, Ket

# 从数组创建
vec = np.array([1, 0], dtype=complex)
bra = Bra(vec)

# 从 Ket 创建
ket = Ket('0')
bra = Bra(ket)

# 从标签创建
bra = Bra('0')
bra = Bra('1')
```

### 属性

```python
bra.vector    # 返回态向量副本（共轭）
bra.dimension  # 维度
bra.ket       # 返回对应的右矢
```

### 方法

与 Ket 类似：norm(), normalize(), copy(), to_vector()

---

## StateVector (态向量)

封装 numpy 数组的量子态表示，支持更多操作。

### 创建

```python
from PySymmetry.phys.quantum.states import StateVector
import numpy as np

vec = np.array([1, 0], dtype=complex)
sv = StateVector(vec, labels=['0', '1'])
```

### 属性

```python
sv.data           # 原始数据
sv.dimension      # 维度
sv.labels         # 标签列表
sv.probabilities  # 测量概率分布
sv.phases         # 相对相位
```

### 方法

```python
# 归一化
normalized = sv.normalize()

# 测量
idx, prob = sv.measure()

# 对子系统求迹
dm = sv.partial_trace(subsystems=[0], dims=[2, 2])
```

---

## DensityMatrix (密度矩阵)

描述混合态的密度算符 ρ = Σ_i p_i |ψ_i⟩⟨ψ_i|。

### 创建

```python
from PySymmetry.phys.quantum.states import DensityMatrix, Ket

# 从纯态创建
ket = Ket('0')
dm = DensityMatrix(ket)  # 自动设置 is_pure=True

# 从向量创建
vec = np.array([1, 0], dtype=complex)
dm = DensityMatrix(vec)

# 从矩阵创建
matrix = np.array([[1, 0], [0, 0]], dtype=complex)
dm = DensityMatrix(matrix, is_pure=False)
```

### 属性

```python
dm.matrix        # 密度矩阵
dm.dimension     # 希尔伯特空间维度
dm.is_pure      # 是否为纯态
dm.purity       # 纯度 Tr(ρ²)
dm.probabilities  # 对角元素（测量概率）
```

### 方法

#### `entropy()`

返回 von Neumann 熵 S = -Tr(ρ log ρ)。

```python
s = dm.entropy()
```

#### `normalize()`

归一化（使迹为1）。

```python
normalized = dm.normalize()
```

#### `expectation(operator)`

计算算符期望值 Tr(ρ O)。

```python
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
exp = dm.expectation(sigma_z)
```

#### `variance(operator)`

计算算符方差 Var(O) = Tr(ρ O²) - Tr(ρ O)²。

```python
var = dm.variance(sigma_z)
```

#### `partial_trace(subsystems, dims)`

对子系统求迹。

```python
# 两 qubit 系统，对第一个 qubit 求迹
reduced_dm = dm.partial_trace(subsystems=[0], dims=[2, 2])
```

#### `fidelity(other)`

计算与另一个密度矩阵的保真度。

```python
dm0 = DensityMatrix(Ket('0'))
dm1 = DensityMatrix(Ket('1'))
f = dm0.fidelity(dm1)  # F(ρ, σ) = (Tr √(√ρ σ √ρ))²
```

---

## 辅助函数

### `basis_state(index, dimension)`

创建计算基底 |n⟩。

```python
from PySymmetry.phys.quantum.states import basis_state

ket0 = basis_state(0, 2)  # |0⟩
ket1 = basis_state(1, 2)  # |1⟩
```

### `bell_state(index=0)`

创建 Bell 态。

```python
from PySymmetry.phys.quantum.states import bell_state

phi_plus = bell_state(0)   # |Φ+⟩ = (|00⟩ + |11⟩)/√2
phi_minus = bell_state(1)  # |Φ-⟩ = (|00⟩ - |11⟩)/√2
psi_plus = bell_state(2)   # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
psi_minus = bell_state(3)  # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
```

### `w_state(n)`

创建 W 态（n qubits）。

```python
from PySymmetry.phys.quantum.states import w_state

ket = w_state(3)  # 3 qubit W 态
# |W⟩ = (|100⟩ + |010⟩ + |001⟩)/√3
```

### `ghz_state(n)`

创建 GHZ 态（n qubits）。

```python
from PySymmetry.phys.quantum.states import ghz_state

ket = ghz_state(3)  # 3 qubit GHZ 态
# |GHZ⟩ = (|000⟩ + |111⟩)/√2
```

### `tensor_product(*states)`

计算态的张量积 |ψ⟩ ⊗ |φ⟩。

```python
from PySymmetry.phys.quantum.states import tensor_product, Ket

ket0 = Ket('0')
ket1 = Ket('1')
result = tensor_product(ket0, ket1)  # |00⟩
```

### `superposition(*states)`

创建态的叠加。

```python
from PySymmetry.phys.quantum.states import superposition, Ket

ket0 = Ket('0')
ket1 = Ket('1')
result = superposition(
    (ket0, 1/np.sqrt(2)),
    (ket1, 1/np.sqrt(2))
)  # (|0⟩ + |1⟩)/√2
```

---

## 示例

### 纯态操作

```python
from PySymmetry.phys.quantum.states import Ket, bell_state, tensor_product

# 创建 Bell 态
phi = bell_state(0)  # |Φ+⟩

# 计算范数
print(phi.norm())  # 1.0

# 张量积
ket0 = Ket('0')
ket1 = Ket('1')
combined = tensor_product(ket0, ket1)  # |01⟩
```

### 混合态操作

```python
from PySymmetry.phys.quantum.states import DensityMatrix, Ket
import numpy as np

# 创建最大混合态
rho = np.eye(4, dtype=complex) / 4
dm = DensityMatrix(rho, is_pure=False)

# 计算熵
print(dm.entropy())  # 2.0 (最大熵)

# 计算保真度
ket0 = Ket(np.array([1, 0, 0, 0], dtype=complex))
dm0 = DensityMatrix(ket0)
f = dm0.fidelity(dm)
print(f)  # 0.25
```
