# PySymmetry 教程

本目录包含 PySymmetry 的详细教程和使用指南。

## 入门教程

- [快速开始](./getting_started.md) - 安装、基本概念、第一个程序

## 进阶教程

- [矩阵运算](./matrix_tutorial.md) - 矩阵创建、运算、分解及物理应用
- [量子力学计算](./quantum_tutorial.md) - 量子态、测量、纠缠、时间演化
- [群论入门](./group_theory_tutorial.md) - 有限群、李群、表示论

## 专项教程

- [氢原子对称性分析](../quantum_module.md#氢原子so4对称性) - SO(4) 对称性详解
- [可视化指南](./visualization_guide.md) - 图形绘制和动画制作

## 学习路径

```
初学者:
    1. 入门教程 (getting_started.md)
    2. 矩阵运算 (matrix_tutorial.md)
    3. 量子力学 (quantum_tutorial.md)

进阶:
    4. 群论入门 (group_theory_tutorial.md)
    5. API 参考文档 (../api/)

专家:
    - 阅读源码
    - 参与开发
    - 论文复现
```

## 快速代码示例

### 创建和操作矩阵

```python
from PySymmetry.core.matrix.factory import MatrixFactory
from PySymmetry.core.matrix.operations import MatrixOperations
import numpy as np

# 创建矩阵
A = MatrixFactory.random(3, 3)
R = MatrixFactory.rotation_2d(np.pi/4)

# 矩阵运算
B = MatrixOperations.multiply(A, R)
exp_A = MatrixOperations.matrix_exponential(A)
```

### 量子态计算

```python
from PySymmetry.phys.quantum.states import Ket, bell_state, DensityMatrix
import numpy as np

# 创建量子态
ket = Ket('0')
bell = bell_state(0)  # |Φ+⟩

# 密度矩阵
rho = DensityMatrix(ket)
exp_z = rho.expectation(np.array([[1,0],[0,-1]]))
```

### 对称性分析

```python
from PySymmetry.phys.hydrogen.so4_symmetry import HydrogenSO4Analyzer

analyzer = HydrogenSO4Analyzer()
report = analyzer.generate_report()
print(report)
```
