# Core Matrix Module API

矩阵运算模块提供丰富的矩阵创建、运算和分解功能。

## MatrixFactory

`MatrixFactory` 类提供静态方法创建各种类型的矩阵。

### 基础矩阵

#### `zeros(rows, cols)`

创建零矩阵。

```python
from PySymmetry.core.matrix.factory import MatrixFactory

m = MatrixFactory.zeros(3, 4)  # 3x4 零矩阵
```

**参数:**
- `rows` (int): 行数，必须 > 0
- `cols` (int): 列数，必须 > 0

**返回:** `np.ndarray`

**抛出:** `ValueError` 当 rows 或 cols <= 0

---

#### `ones(rows, cols)`

创建全1矩阵。

```python
m = MatrixFactory.ones(2, 3)  # 2x3 全1矩阵
```

---

#### `identity(n)`

创建单位矩阵。

```python
m = MatrixFactory.identity(4)  # 4x4 单位矩阵
```

---

#### `random(rows, cols)`

创建随机矩阵（0-1均匀分布）。

```python
m = MatrixFactory.random(3, 3)  # 3x3 随机矩阵
```

---

#### `random_normal(rows, cols, mean=0.0, std=1.0)`

创建正态分布随机矩阵。

```python
m = MatrixFactory.random_normal(3, 3, mean=5.0, std=2.0)
```

---

#### `from_diagonal(diagonal)`

从对角元素创建对角矩阵。

```python
d = np.array([1, 2, 3])
m = MatrixFactory.from_diagonal(d)
# [[1, 0, 0],
#  [0, 2, 0],
#  [0, 0, 3]]
```

---

#### `from_list(data)`

从二维列表创建矩阵。

```python
data = [[1, 2], [3, 4]]
m = MatrixFactory.from_list(data)
```

---

#### `from_function(rows, cols, func)`

从函数创建矩阵。

```python
m = MatrixFactory.from_function(3, 3, lambda i, j: i + j)
```

---

### 特殊矩阵

#### `toeplitz(first_row, first_col=None)`

创建托普利茨矩阵。

```python
row = np.array([1, 2, 3, 4])
m = MatrixFactory.toeplitz(row)
```

---

#### `circulant(first_row)`

创建循环矩阵。

```python
row = np.array([1, 2, 3])
m = MatrixFactory.circulant(row)
```

---

#### `vandermonde(x, n=None)`

创建范德蒙德矩阵。

```python
x = np.array([1, 2, 3])
m = MatrixFactory.vandermonde(x)
# [[1, 1, 1],
#  [1, 2, 4],
#  [1, 3, 9]]
```

---

#### `hilbert(n)`

创建希尔伯特矩阵。

```python
m = MatrixFactory.hilbert(4)
```

---

#### `pascal(n, kind='symmetric')`

创建帕斯卡矩阵。

```python
m = MatrixFactory.pascal(3, kind='symmetric')
m = MatrixFactory.pascal(3, kind='lower')
```

---

#### `dft(n)`

创建离散傅里叶变换矩阵。

```python
m = MatrixFactory.dft(4)  # 4x4 DFT 矩阵
```

---

#### `hadamard(n)`

创建哈达玛矩阵（n必须是2的幂）。

```python
m = MatrixFactory.hadamard(4)
```

---

#### `companion(polynomial_coefficients)`

创建伴随矩阵。

```python
coeffs = np.array([1, -3, 2])  # x^2 - 3x + 2
m = MatrixFactory.companion(coeffs)
```

---

### 变换矩阵

#### `rotation_2d(theta)`

创建2D旋转矩阵。

```python
m = MatrixFactory.rotation_2d(np.pi / 4)  # 45度旋转
```

---

#### `rotation_3d(axis, theta)`

创建3D旋转矩阵。

```python
m = MatrixFactory.rotation_3d('x', np.pi)  # X轴旋转180度
m = MatrixFactory.rotation_3d('y', np.pi)
m = MatrixFactory.rotation_3d('z', np.pi)
```

---

#### `reflection_2d(theta)`

创建2D反射矩阵。

```python
m = MatrixFactory.reflection_2d(0)  # X轴反射
```

---

#### `shear_2d(axis, factor)`

创建2D剪切矩阵。

```python
m = MatrixFactory.shear_2d('x', 2)  # X方向剪切
m = MatrixFactory.shear_2d('y', 2)  # Y方向剪切
```

---

#### `scaling_2d(sx, sy)`

创建2D缩放矩阵。

```python
m = MatrixFactory.scaling_2d(2, 3)
```

---

#### `scaling_3d(sx, sy, sz)`

创建3D缩放矩阵。

```python
m = MatrixFactory.scaling_3d(1, 2, 3)
```

---

#### `permutation(n, perm)`

创建置换矩阵。

```python
m = MatrixFactory.permutation(3, [2, 0, 1])
```

---

#### `block_diagonal(blocks)`

创建块对角矩阵。

```python
blocks = [np.eye(2), np.eye(3)]
m = MatrixFactory.block_diagonal(blocks)
```

---

#### `tridiagonal(diagonal, upper, lower)`

创建三对角矩阵。

```python
d = np.array([1, 2, 3, 4])
u = np.array([1, 2, 3])
l = np.array([1, 2, 3])
m = MatrixFactory.tridiagonal(d, u, l)
```

---

### 特殊类型矩阵

#### `symmetric(rows, cols)`

创建随机对称矩阵。

```python
m = MatrixFactory.symmetric(3, 3)
```

---

#### `positive_definite(n)`

创建随机正定矩阵。

```python
m = MatrixFactory.positive_definite(3)
```

---

#### `orthogonal(n)`

创建随机正交矩阵。

```python
m = MatrixFactory.orthogonal(3)
```

---

#### `unitary(n)`

创建随机酉矩阵。

```python
m = MatrixFactory.unitary(3)
```

---

#### `sparse(rows, cols, density=0.1)`

创建稀疏矩阵。

```python
m = MatrixFactory.sparse(10, 10, density=0.1)
```

---

## MatrixOperations

`MatrixOperations` 类提供矩阵运算方法。

### 算术运算

#### `add(a, b)`

矩阵加法。

```python
result = MatrixOperations.add(A, B)
```

---

#### `subtract(a, b)`

矩阵减法。

```python
result = MatrixOperations.subtract(A, B)
```

---

#### `multiply(a, b)`

矩阵乘法。

```python
result = MatrixOperations.multiply(A, B)  # A @ B
```

---

#### `elementwise_multiply(a, b)`

逐元素乘法。

```python
result = MatrixOperations.elementwise_multiply(A, B)
```

---

#### `elementwise_divide(a, b)`

逐元素除法。

```python
result = MatrixOperations.elementwise_divide(A, B)
```

---

### 矩阵属性

#### `transpose(matrix)`

转置。

```python
result = MatrixOperations.transpose(A)
```

---

#### `conjugate_transpose(matrix)`

共轭转置。

```python
result = MatrixOperations.conjugate_transpose(A)
```

---

### 积运算

#### `kronecker_product(a, b)`

克罗内克积。

```python
result = MatrixOperations.kronecker_product(A, B)
```

---

#### `hadamard_product(a, b)`

阿达玛积（逐元素乘法）。

```python
result = MatrixOperations.hadamard_product(A, B)
```

---

#### `outer_product(a, b)`

外积。

```python
result = MatrixOperations.outer_product(a, b)
```

---

#### `inner_product(a, b)`

内积。

```python
result = MatrixOperations.inner_product(a, b)
```

---

#### `dot_product(a, b)`

点积。

```python
result = MatrixOperations.dot_product(a, b)
```

---

### 矩阵函数

#### `power(matrix, n)`

矩阵幂运算。

```python
result = MatrixOperations.power(A, 3)  # A^3
```

---

#### `inverse(matrix)`

矩阵求逆。

```python
result = MatrixOperations.inverse(A)
```

---

#### `pseudo_inverse(matrix)`

Moore-Penrose伪逆。

```python
result = MatrixOperations.pseudo_inverse(A)
```

---

#### `solve(A, b)`

求解线性方程组 Ax = b。

```python
x = MatrixOperations.solve(A, b)
```

---

#### `least_squares(A, b)`

最小二乘解。

```python
x = MatrixOperations.least_squares(A, b)
```

---

#### `matrix_exponential(matrix)`

矩阵指数 exp(A)。

```python
result = MatrixOperations.matrix_exponential(A)
```

---

#### `matrix_logarithm(matrix)`

矩阵对数 log(A)。

```python
result = MatrixOperations.matrix_logarithm(A)
```

---

#### `matrix_sqrt(matrix)`

矩阵平方根。

```python
result = MatrixOperations.matrix_sqrt(A)
```

---

### 矩阵属性

#### `trace(matrix)`

迹。

```python
tr = MatrixOperations.trace(A)
```

---

#### `determinant(matrix)`

行列式。

```python
det = MatrixOperations.determinant(A)
```

---

#### `rank(matrix)`

秩。

```python
r = MatrixOperations.rank(A)
```

---

#### `norm(matrix, ord='fro')`

范数。

```python
n = MatrixOperations.norm(A, 'fro')   # Frobenius范数
n = MatrixOperations.norm(A, 1)       # L1范数
n = MatrixOperations.norm(A, 2)       # L2范数
n = MatrixOperations.norm(A, np.inf)  # 无穷范数
```

---

#### `condition_number(matrix)`

条件数。

```python
cond = MatrixOperations.condition_number(A)
```

---

### 矩阵变换

#### `concatenate(matrices, axis=0)`

矩阵拼接。

```python
result = MatrixOperations.concatenate([A, B], axis=0)
```

---

#### `stack(matrices, axis=0)`

矩阵堆叠。

```python
result = MatrixOperations.stack([A, B], axis=0)
```

---

#### `split(matrix, indices_or_sections, axis=0)`

矩阵分割。

```python
parts = MatrixOperations.split(A, 2, axis=0)
```

---

#### `tile(matrix, reps)`

矩阵重复。

```python
result = MatrixOperations.tile(A, (2, 3))
```

---

#### `repeat(matrix, repeats, axis=None)`

元素重复。

```python
result = MatrixOperations.repeat(A, 2, axis=0)
```

---

## MatrixDecompositions

`MatrixDecompositions` 类提供矩阵分解方法。

### 特征值分解

#### `eigen_decomposition(matrix)`

特征值分解。

```python
evals, evecs = MatrixDecompositions.eigen_decomposition(A)
```

**返回:** (eigenvalues, eigenvectors) 元组

---

#### `spectral_decomposition(matrix)`

谱分解（对称/埃尔米特矩阵）。

```python
evals, evecs = MatrixDecompositions.spectral_decomposition(A)
```

---

### 奇异值分解

#### `svd(matrix, full_matrices=True)`

奇异值分解。

```python
U, S, Vh = MatrixDecompositions.svd(A)
# U: 左奇异向量
# S: 奇异值
# Vh: 右奇异向量的共轭转置
```

---

### QR分解

#### `qr_decomposition(matrix, mode='reduced')`

QR分解。

```python
Q, R = MatrixDecompositions.qr_decomposition(A, mode='reduced')
```

---

### Cholesky分解

#### `cholesky_decomposition(matrix)`

Cholesky分解。

```python
L = MatrixDecompositions.cholesky_decomposition(A)  # A = L @ L.T
```

---

### LU分解

#### `lu_decomposition(matrix)`

LU分解。

```python
P, L, U = MatrixDecompositions.lu_decomposition(A)
```

---

### Schur分解

#### `schur_decomposition(matrix, output='real')`

Schur分解。

```python
T, Z = MatrixDecompositions.schur_decomposition(A, output='real')
```

---

### Hessenberg分解

#### `hessenberg_decomposition(matrix)`

Hessenberg分解。

```python
H = MatrixDecompositions.hessenberg_decomposition(A)
```

---

### 极分解

#### `polar_decomposition(matrix, side='right')`

极分解。

```python
U, P = MatrixDecompositions.polar_decomposition(A, side='right')
# A = U @ P (right polar)
# A = P @ U (left polar)
```

---

### Jordan分解

#### `jordan_decomposition(matrix)`

Jordan分解（数值近似）。

```python
P, J = MatrixDecompositions.jordan_decomposition(A)
```

---

### 双对角分解

#### `bidiagonal_decomposition(matrix)`

双对角分解。

```python
U, B, Vh = MatrixDecompositions.bidiagonal_decomposition(A)
```
