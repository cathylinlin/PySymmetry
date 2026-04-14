# Performance Module API

性能优化模块 (`tools.performance`) 提供各类性能优化功能。

## 目录

- [装饰器](#装饰器)
- [矩阵优化](#矩阵优化)
- [特征值优化](#特征值优化)
- [并行计算](#并行计算)
- [辅助函数](#辅助函数)

---

## 装饰器

### `cache_result(maxsize=128)`

缓存函数结果装饰器。

```python
from PySymmetry.tools.performance import cache_result

@cache_result(maxsize=256)
def expensive_function(x):
    # 复杂计算
    return x ** 2
```

---

### `vectorize_func()`

向量化函数装饰器。

```python
from PySymmetry.tools.performance import vectorize_func

@vectorize_func
def my_func(x):
    return x ** 2
```

---

## 矩阵优化

### `optimize_matrix_multiply(a, b)`

根据矩阵大小选择最优算法。

```python
from PySymmetry.tools.performance import optimize_matrix_multiply
import numpy as np

A = np.random.rand(100, 100)
B = np.random.rand(100, 100)
C = optimize_matrix_multiply(A, B)
```

- 小矩阵 (<64): 使用 `np.dot`
- 大稀疏矩阵: 使用 `scipy.sparse`

---

### `batch_matrix_multiply(matrices, vector)`

批量矩阵乘法（多个矩阵乘以同一个向量）。

```python
from PySymmetry.tools.performance import batch_matrix_multiply
import numpy as np

matrices = [np.random.rand(3, 3) for _ in range(10)]
vector = np.random.rand(3)
result = batch_matrix_multiply(matrices, vector)
# result shape: (10, 3)
```

---

## 特征值优化

### `optimize_eigendecomposition(matrix, k=None)`

优化的特征值分解。

```python
from PySymmetry.tools.performance import optimize_eigendecomposition
import numpy as np

A = np.random.rand(100, 100)
evals, evecs = optimize_eigendecomposition(A)

# 只计算 k 个特征值
evals, evecs = optimize_eigendecomposition(A, k=5)
```

- 小矩阵: 使用 `numpy.linalg.eigh`
- 大矩阵指定 k: 使用 `scipy.sparse.linalg.eigsh`

---

### `sparse_diagonalize(matrix, k=6, which='SA')`

稀疏矩阵特征值问题。

```python
from PySymmetry.tools.performance import sparse_diagonalize
import numpy as np

A = np.random.rand(1000, 1000)
evals, evecs = sparse_diagonalize(A, k=10)
```

---

## 并行计算

### `parallel_apply(func, args_list, n_jobs=-1)`

并行应用函数。

```python
from PySymmetry.tools.performance import parallel_apply

def process(x):
    return x ** 2

args_list = [(1,), (2,), (3,), (4,), (5,)]
results = parallel_apply(process, args_list, n_jobs=2)
# results: [1, 4, 9, 16, 25]
```

如果没有安装 joblib，将使用串行执行。

---

## 辅助函数

### `optimize_kron_sequence(matrices)`

优化的克罗内克积序列。

```python
from PySymmetry.tools.performance import optimize_kron_sequence
import numpy as np

matrices = [np.eye(2), np.eye(2), np.eye(2)]
result = optimize_kron_sequence(matrices)
```

---

### `optimize_blevit_check(matrix, eps=1e-10)`

检查矩阵是否为厄米矩阵。

```python
from PySymmetry.tools.performance import optimize_blevit_check
import numpy as np

A = np.array([[1, 2+1j], [2-1j, 1]], dtype=complex)
is_hermitian = optimize_blevit_check(A)  # True
```

---

### `optimize_trace(matrix)`

优化的矩阵迹计算（使用 einsum）。

```python
from PySymmetry.tools.performance import optimize_trace
import numpy as np

A = np.random.rand(100, 100)
tr = optimize_trace(A)
```

---

### `optimize_outer_sequence(vectors)`

优化的外积序列。

```python
from PySymmetry.tools.performance import optimize_outer_sequence
import numpy as np

vectors = [np.random.rand(3) for _ in range(5)]
result = optimize_outer_sequence(vectors)
# result shape: (5, 3, 3)
```

---

### `matrix_power_sequence(matrix, max_power=10)`

生成矩阵幂序列 A^0, A^1, ..., A^n。

```python
from PySymmetry.tools.performance import matrix_power_sequence
import numpy as np

A = np.random.rand(3, 3)
powers = matrix_power_sequence(A, max_power=5)
# powers[0] = I, powers[1] = A, powers[2] = A^2, ...
```

---

### `block_diagonalize(matrices)`

块对角化。

```python
from PySymmetry.tools.performance import block_diagonalize
import numpy as np

matrices = [np.eye(2), np.eye(3), np.eye(4)]
result = block_diagonalize(matrices)
# result shape: (9, 9)
```

---

### `optimize_commutator(a, b)`

优化的对易子 [A, B] = AB - BA。

```python
from PySymmetry.tools.performance import optimize_commutator
import numpy as np

A = np.random.rand(3, 3)
B = np.random.rand(3, 3)
comm = optimize_commutator(A, B)
```

---

### `optimize_anticommutator(a, b)`

优化的反对易子 {A, B} = AB + BA。

```python
from PySymmetry.tools.performance import optimize_anticommutator
import numpy as np

A = np.random.rand(3, 3)
B = np.random.rand(3, 3)
anticomm = optimize_anticommutator(A, B)
```

---

### `batch_evaluate(func, points, n_jobs=-1)`

批量函数评估（并行）。

```python
from PySymmetry.tools.performance import batch_evaluate
import numpy as np

def f(x):
    return np.sin(x) * np.exp(-x)

points = np.linspace(0, 10, 100)
results = batch_evaluate(f, points, n_jobs=4)
```

---

## 模块变量

### `NUMBA_AVAILABLE`

检查 numba 是否可用。

```python
from PySymmetry.tools.performance import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    print("Numba is available")
```

---

### `JOBLIB_AVAILABLE`

检查 joblib 是否可用。

```python
from PySymmetry.tools.performance import JOBLIB_AVAILABLE

if JOBLIB_AVAILABLE:
    print("Joblib is available")
```

---

## 示例

### 批量计算优化

```python
from PySymmetry.tools.performance import (
    batch_matrix_multiply,
    batch_evaluate,
    parallel_apply
)
import numpy as np

# 批量矩阵乘法
matrices = [np.random.rand(50, 50) for _ in range(20)]
vector = np.random.rand(50)
result = batch_matrix_multiply(matrices, vector)

# 批量函数评估
def wave_function(x, n):
    return np.sin(n * np.pi * x)

points = np.linspace(0, 1, 100)
results = batch_evaluate(lambda x: wave_function(x, 3), points)
```
