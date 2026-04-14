# PySymmetry 入门教程

本教程帮助你快速上手 PySymmetry 库。

## 目录

1. [安装](#安装)
2. [快速开始](#快速开始)
3. [矩阵运算](#矩阵运算)
4. [量子态](#量子态)
5. [下一步](#下一步)

---

## 安装

### 使用 pip 安装

```bash
pip install PySymmetry
```

### 从源码安装

```bash
git clone https://github.com/your-repo/PySymmetry.git
cd PySymmetry
pip install -e .
```

### 依赖

PySymmetry 依赖以下库：
- numpy >= 1.20
- scipy >= 1.7
- matplotlib >= 3.4
- sympy >= 1.8 (部分功能)

---

## 快速开始

### 导入模块

```python
import numpy as np
from PySymmetry.core.matrix.factory import MatrixFactory
from PySymmetry.core.matrix.operations import MatrixOperations
from PySymmetry.phys.quantum.states import Ket, DensityMatrix
```

### 创建矩阵

```python
# 创建单位矩阵
I = MatrixFactory.identity(4)

# 创建随机矩阵
A = MatrixFactory.random(3, 3)

# 创建特殊矩阵
H = MatrixFactory.hilbert(4)  # 希尔伯特矩阵
D = MatrixFactory.dft(4)      # 离散傅里叶变换矩阵
```

### 矩阵运算

```python
from PySymmetry.core.matrix.operations import MatrixOperations

# 矩阵乘法
C = MatrixOperations.multiply(A, B)

# 矩阵指数
exp_A = MatrixOperations.matrix_exponential(A)

# 求逆
A_inv = MatrixOperations.inverse(A)
```

### 量子态

```python
from PySymmetry.phys.quantum.states import Ket, basis_state, bell_state

# 创建量子态
ket0 = basis_state(0, 2)  # |0⟩
ket1 = basis_state(1, 2)  # |1⟩

# 创建 Bell 态
phi_plus = bell_state(0)  # |Φ+⟩ = (|00⟩ + |11⟩)/√2

# 检查归一化
print(f"范数: {ket0.norm()}")  # 1.0
```

---

## 矩阵运算

### 创建矩阵

PySymmetry 提供丰富的矩阵创建函数：

```python
from PySymmetry.core.matrix.factory import MatrixFactory
import numpy as np

# 零矩阵和全1矩阵
zeros = MatrixFactory.zeros(3, 4)
ones = MatrixFactory.ones(3, 4)

# 单位矩阵
I = MatrixFactory.identity(4)

# 随机矩阵
uniform = MatrixFactory.random(3, 3)      # 均匀分布 [0, 1]
normal = MatrixFactory.random_normal(3, 3, mean=0, std=1)  # 正态分布

# 对角矩阵
D = MatrixFactory.from_diagonal(np.array([1, 2, 3]))

# 从函数创建
M = MatrixFactory.from_function(3, 3, lambda i, j: i + j)
```

### 特殊矩阵

```python
# 旋转矩阵
R_z = MatrixFactory.rotation_2d(np.pi/4)        # 2D 旋转
R_x = MatrixFactory.rotation_3d('x', np.pi/2) # 3D 绕X轴旋转

# 反射矩阵
ref = MatrixFactory.reflection_2d(0)

# 缩放矩阵
scale = MatrixFactory.scaling_2d(2, 3)

# 剪切矩阵
shear = MatrixFactory.shear_2d('x', 2)
```

### 矩阵分解

```python
from PySymmetry.core.matrix.decompositions import MatrixDecompositions
import numpy as np

A = np.array([[4, 2], [1, 3]], dtype=float)

# 特征值分解
evals, evecs = MatrixDecompositions.eigen_decomposition(A)

# 奇异值分解
U, S, Vh = MatrixDecompositions.svd(A)

# QR分解
Q, R = MatrixDecompositions.qr_decomposition(A)

# Cholesky分解（对称正定矩阵）
B = np.array([[4, 2], [2, 5]], dtype=float)
L = MatrixDecompositions.cholesky_decomposition(B)

# LU分解
P, L, U = MatrixDecompositions.lu_decomposition(A)
```

---

## 量子态

### Ket（右矢）

```python
from PySymmetry.phys.quantum.states import Ket, basis_state
import numpy as np

# 从数组创建
vec = np.array([1, 0], dtype=complex)
ket = Ket(vec)

# 从标签创建
ket0 = Ket('0')  # |0⟩
ket1 = Ket('1')  # |1⟩

# 计算基底
ket = basis_state(0, 2)  # |0⟩

# 态运算
plus = ket0 + ket1                    # 叠加
scaled = ket * (1/np.sqrt(2))        # 标量乘法
```

### 密度矩阵

```python
from PySymmetry.phys.quantum.states import DensityMatrix, Ket

# 从纯态创建
ket = Ket('0')
rho = DensityMatrix(ket)

# 纯态性质
print(f"纯度: {rho.purity}")  # 1.0
print(f"熵: {rho.entropy()}")  # 0.0

# 期望值
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
exp_val = rho.expectation(sigma_z)
```

### 纠缠态

```python
from PySymmetry.phys.quantum.states import bell_state, w_state, ghz_state, tensor_product, Ket

# Bell 态
phi = bell_state(0)   # |Φ+⟩
psi = bell_state(2)   # |Ψ+⟩

# GHZ 态
ghz = ghz_state(3)    # 3-qubit GHZ

# W 态
w = w_state(3)        # 3-qubit W

# 张量积
ket0 = Ket('0')
ket1 = Ket('1')
combined = tensor_product(ket0, ket1)  # |01⟩
```

---

## 下一步

- [矩阵模块详细教程](./tutorials/matrix_tutorial.md)
- [量子力学教程](./tutorials/quantum_tutorial.md)
- [API 参考文档](../api/)
- [量子模块完整文档](../quantum_module.md)
