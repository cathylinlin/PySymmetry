# 矩阵运算教程

本教程深入介绍 PySymmetry 的矩阵运算功能。

## 目录

1. [矩阵创建](#矩阵创建)
2. [矩阵运算](#矩阵运算)
3. [矩阵分解](#矩阵分解)
4. [物理应用](#物理应用)

---

## 矩阵创建

### 基础矩阵

```python
from PySymmetry.core.matrix.factory import MatrixFactory
import numpy as np

# 零矩阵
zeros = MatrixFactory.zeros(3, 4)
# [[0, 0, 0, 0],
#  [0, 0, 0, 0],
#  [0, 0, 0, 0]]

# 全1矩阵
ones = MatrixFactory.ones(2, 3)

# 单位矩阵
I = MatrixFactory.identity(4)
```

### 随机矩阵

```python
# 均匀分布 [0, 1]
A = MatrixFactory.random(3, 3)

# 正态分布
B = MatrixFactory.random_normal(3, 3, mean=5.0, std=2.0)

# 固定随机种子
np.random.seed(42)
C = MatrixFactory.random(2, 2)
```

### 从已有数据创建

```python
# 从对角线创建
D = MatrixFactory.from_diagonal(np.array([1, 2, 3, 4]))

# 从列表创建
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
M = MatrixFactory.from_list(data)

# 从函数创建
M = MatrixFactory.from_function(3, 3, lambda i, j: i * j)
# [[0, 0, 0],
#  [0, 1, 2],
#  [0, 2, 4]]
```

---

## 矩阵运算

### 基本运算

```python
from PySymmetry.core.matrix.operations import MatrixOperations
import numpy as np

A = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[5, 6], [7, 8]], dtype=float)

# 加减乘除
C = MatrixOperations.add(A, B)
D = MatrixOperations.subtract(A, B)
E = MatrixOperations.multiply(A, B)        # 矩阵乘法
F = MatrixOperations.elementwise_multiply(A, B)  # 逐元素乘法
```

### 转置和共轭

```python
# 转置
A_T = MatrixOperations.transpose(A)

# 共轭转置（厄米伴随）
A_dag = MatrixOperations.conjugate_transpose(A)
```

### 幂运算

```python
# 矩阵幂
A2 = MatrixOperations.power(A, 2)   # A^2
A3 = MatrixOperations.power(A, 3)   # A^3
A0 = MatrixOperations.power(A, 0)   # I
```

### 矩阵函数

```python
# 逆矩阵
A_inv = MatrixOperations.inverse(A)

# 伪逆（Moore-Penrose）
A_pinv = MatrixOperations.pseudo_inverse(A)

# 矩阵指数 exp(A)
exp_A = MatrixOperations.matrix_exponential(A)

# 矩阵对数 log(A)
log_A = MatrixOperations.matrix_logarithm(A)

# 矩阵平方根
sqrt_A = MatrixOperations.matrix_sqrt(A)
```

### 解线性方程组

```python
# 求解 Ax = b
A = np.array([[1, 1], [3, 1]], dtype=float)
b = np.array([3, 7], dtype=float)
x = MatrixOperations.solve(A, b)

# 最小二乘
A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
b = np.array([1, 2, 3], dtype=float)
x, residuals, rank, s = MatrixOperations.least_squares(A, b)
```

---

## 矩阵分解

### 特征值分解

```python
from PySymmetry.core.matrix.decompositions import MatrixDecompositions
import numpy as np

A = np.array([[4, 2], [1, 3]], dtype=float)

# 特征值分解 A = VΛV^(-1)
evals, evecs = MatrixDecompositions.eigen_decomposition(A)

print(f"特征值: {evals}")
# [5., 2.]

print(f"特征向量:\n{evecs}")
```

### 谱分解（对称矩阵）

```python
# 对称矩阵的特征值分解
S = np.array([[2, 1], [1, 2]], dtype=float)
evals, evecs = MatrixDecompositions.spectral_decomposition(S)

# 验证 S = QΛQ^T
Q = evecs
Λ = np.diag(evals)
reconstructed = Q @ Λ @ Q.T
```

### 奇异值分解

```python
# A = UΣV^H
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
U, S, Vh = MatrixDecompositions.svd(A)

print(f"奇异值: {S}")
# [9.508  0.772]
```

### QR分解

```python
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
Q, R = MatrixDecompositions.qr_decomposition(A)
# A = QR
```

### Cholesky分解

```python
# 对称正定矩阵 A = LL^T
A = np.array([[4, 2], [2, 5]], dtype=float)
L = MatrixDecompositions.cholesky_decomposition(A)
# [[2. , 0. ],
#  [1. , 2. ]]
```

### LU分解

```python
A = np.array([[1, 2], [3, 4]], dtype=float)
P, L, U = MatrixDecompositions.lu_decomposition(A)
# PA = LU
```

---

## 物理应用

### 旋转矩阵

```python
from PySymmetry.core.matrix.factory import MatrixFactory
import numpy as np

# 2D 旋转
theta = np.pi / 4  # 45度
R = MatrixFactory.rotation_2d(theta)

# 3D 旋转（绕不同轴）
Rx = MatrixFactory.rotation_3d('x', np.pi/2)  # 绕X轴
Ry = MatrixFactory.rotation_3d('y', np.pi/2)  # 绕Y轴
Rz = MatrixFactory.rotation_3d('z', np.pi/2)  # 绕Z轴

# 验证：连续旋转回到原点
R_total = Rz @ Ry @ Rx
```

### Pauli 矩阵

```python
# Pauli 矩阵
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# 验证对易关系 [σ_i, σ_j] = 2iε_ijkσ_k
from PySymmetry.core.matrix.operations import MatrixOperations
comm_xy = MatrixOperations.subtract(
    sigma_x @ sigma_y,
    sigma_y @ sigma_x
)
# 应该等于 2j*sigma_z
```

### 量子门的矩阵表示

```python
# Hadamard 门
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Pauli 门
X = sigma_x  # NOT 门
Y = sigma_y
Z = sigma_z

# CNOT 门 (control=0, target=1)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

# 验证 Hadamard 变换
H0 = H @ np.array([1, 0], dtype=complex)  # |0⟩ -> |+⟩
H1 = H @ np.array([0, 1], dtype=complex)  # |1⟩ -> |-⟩
```

### 本征问题求解

```python
# 简谐振子哈密顿量 (离散化)
N = 50
dx = 0.1
x = np.linspace(-N*dx/2, N*dx/2, N)

# 动能项 (二阶导数近似)
T = np.zeros((N, N))
for i in range(1, N-1):
    T[i, i-1] = -1
    T[i, i] = 2
    T[i, i+1] = -1
T = T / (dx**2)

# 势能项
V = np.diag(x**2)

# 总哈密顿量
H = -T/2 + V  # ℏ = m = 1

# 对角化
evals, evecs = MatrixDecompositions.eigen_decomposition(H)
print(f"前5个能量本征值: {evals[:5]}")
# 应该接近 0.5, 1.5, 2.5, 3.5, 4.5 (单位 ℏω)
```
