# 群论入门教程

本教程介绍 PySymmetry 中的群论基础概念和实现。

## 目录

1. [基础概念](#基础概念)
2. [有限群](#有限群)
3. [李群与李代数](#李群与李代数)
4. [群表示](#群表示)

---

## 基础概念

### 群的定义

群 (G, ·) 由集合 G 和二元运算 · 组成，满足：
- **封闭性**: 若 a, b ∈ G，则 a·b ∈ G
- **结合律**: (a·b)·c = a·(b·c)
- **单位元**: 存在 e ∈ G，使得 e·a = a·e = a
- **逆元**: 对每个 a ∈ G，存在 a⁻¹ ∈ G，使得 a·a⁻¹ = e

### 抽象群类

```python
from PySymmetry.core.group_theory.abstract_group import Group, GroupElement

# 自定义群元素
class IntegerModNElement(GroupElement):
    def __init__(self, n: int, mod: int):
        self.n = n % mod
        self.mod = mod
    
    def __mul__(self, other):
        return IntegerModNElement((self.n + other.n) % self.mod, self.mod)
    
    def inverse(self):
        return IntegerModNElement(-self.n % self.mod, self.mod)
    
    def __pow__(self, n: int):
        return IntegerModNElement((self.n * n) % self.mod, self.mod)
    
    def __hash__(self):
        return hash((self.n, self.mod))
    
    def is_identity(self):
        return self.n == 0
    
    def order(self):
        if self.n == 0:
            return 1
        for i in range(1, self.mod + 1):
            if (self.n * i) % self.mod == 0:
                return i
        return self.mod
```

---

## 有限群

### 循环群

```python
from PySymmetry.core.group_theory.finite_groups import CyclicGroup

# Z_n 循环群
C5 = CyclicGroup(5)  # Z_5

print(f"群阶: {C5.order()}")           # 5
print(f"是否为阿贝尔: {C5.is_abelian()}")  # True
print(f"元素: {list(C5.elements())}")
```

### 二面体群

```python
from PySymmetry.core.group_theory.finite_groups import DihedralGroup

# D_n 二面体群（正 n 边形的对称群）
D3 = DihedralGroup(3)  # 等边三角形的对称群

print(f"群阶: {D3.order()}")  # 6
print(f"元素数: {len(list(D3.elements()))}")
```

### 对称群

```python
from PySymmetry.core.group_theory.finite_groups import SymmetricGroup

# S_n 对称群（n 个元素的置换群）
S3 = SymmetricGroup(3)

print(f"群阶: {S3.order()}")  # 6 = 3!
print(f"是否为阿贝尔: {S3.is_abelian()}")  # False

# 子群
A3 = S3.alternating_subgroup()  # A_3 交错子群
print(f"A_3 阶: {A3.order()}")  # 3
```

### 四元数群

```python
from PySymmetry.core.group_theory.finite_groups import QuaternionGroup

Q8 = QuaternionGroup()

print(f"群阶: {Q8.order()}")       # 8
print(f"中心阶数: {len(list(Q8.center()))}")  # 2
```

---

## 李群与李代数

### 经典李群

```python
from PySymmetry.core.lie_theory.classical_groups import (
    GL, SL, SU, SO, U, SO_n
)

# 一般线性群 GL(n)
GL3 = GL(3)
print(f"GL(3) 维度: {GL3.dimension}")  # 9

# 特殊线性群 SL(n) - 行列式为1
SL2 = SL(2)
print(f"SL(2) 维度: {SL2.dimension}")  # 3

# 酉群 U(n)
U2 = U(2)
print(f"U(2) 维度: {U2.dimension}")  # 4

# 特殊酉群 SU(n)
SU2 = SU(2)
print(f"SU(2) 维度: {SU2.dimension}")  # 3

# 正交群 SO(n)
SO3 = SO(3)
print(f"SO(3) 维度: {SO3.dimension}")  # 3
```

### 李代数

```python
from PySymmetry.core.lie_theory.lie_algebra import LieAlgebra
import numpy as np

# SU(2) 李代数
su2 = LieAlgebra.special_unitary(2)

print(f"维度: {su2.dimension}")
print(f"基元素数: {len(su2.basis)}")

# 李括号
X = su2.basis[0]
Y = su2.basis[1]
Z = su2.lie_bracket(X, Y)
print(f"[X, Y] = {Z}")
```

### 生成元

```python
from PySymmetry.core.lie_theory.generators import (
    TranslationGenerator,
    RotationGenerator,
    BoostGenerator
)
import numpy as np

# 平移生成元
T = TranslationGenerator(dim=3)
print(f"平移生成元:\n{T.generators}")

# 旋转生成元
R = RotationGenerator()
print(f"角动量算符:\n{R.generators}")

# Lorentz 推进生成元
B = BoostGenerator()
print(f"推进生成元:\n{B.generators}")
```

---

## 群表示

### 表示论基础

```python
from PySymmetry.core.representation.group_representation import GroupRepresentation
import numpy as np

# S_3 的表示
S3_rep = GroupRepresentation(group=S3, dimension=2)

# 定义表示映射
def s3_rep_element(perm):
    if perm == ():
        return np.eye(2)
    elif perm == (0, 1):  # 对换
        return np.array([[0, 1], [1, 0]])
    # ...
```

### 特征标理论

```python
from PySymmetry.core.representation.character_theory import CharacterTable

# 构建 S_3 特征标表
chi = CharacterTable(S3)

print("S_3 的特征标表:")
print(chi.table)

# 不可约表示的维数平方和
print(f"∑ d_i² = {sum(d**2 for d in chi.irrep_dimensions)}")  # 应等于群阶
```

### 表示空间

```python
from PySymmetry.core.representation.representation_space import RepresentationSpace

# 构建表示空间
V = RepresentationSpace(dimension=4, basis=['v1', 'v2', 'v3', 'v4'])

# 张量积表示
V1 = RepresentationSpace(dimension=2)
V2 = RepresentationSpace(dimension=2)
V_tensor = V1.tensor_product(V2)
print(f"张量积维度: {V_tensor.dimension}")  # 4
```

---

## 物理应用

### 氢原子对称性

```python
from PySymmetry.phys.hydrogen.so4_symmetry import HydrogenSO4Analyzer
import numpy as np

# 分析氢原子的 SO(4) 对称性
analyzer = HydrogenSO4Analyzer()

# 检测对称性
has_so4 = analyzer.detect_so4()
print(f"是否存在 SO(4) 对称性: {has_so4}")

# 获取量子数
quantum_numbers = analyzer.quantum_numbers
print(f"主量子数 n: {quantum_numbers['n']}")
print(f"角动量 l: {quantum_numbers['l']}")
```

### 角动量耦合

```python
from PySymmetry.phys.quantum.angular_momentum import (
    ClebschGordanCoefficients,
    WignerDMatrix
)
import numpy as np

# Clebsch-Gordan 系数
cg = ClebschGordanCoefficients(j1=1/2, j2=1/2)

# j1=1/2, j2=1/2 -> j=0 或 j=1
coeff_10 = cg.get(1/2, 1/2, 0, 0)
coeff_11 = cg.get(1/2, 1/2, 1, 1)
print(f"CG(1/2, 1/2; 0, 0) = {coeff_10}")
print(f"CG(1/2, 1/2; 1, 1) = {coeff_11}")
```

### 选择规则

```python
from PySymmetry.phys.quantum.selection_rules import apply_selection_rule

# 电偶极选择规则
rule = {'Δl': ±1, 'Δm': 0, ±1}

# 检查跃迁是否允许
allowed = apply_selection_rule(
    initial={'l': 0, 'm': 0},
    final={'l': 1, 'm': 0},
    rule=rule
)
print(f"1s -> 2p 跃迁允许: {allowed}")  # True
```

---

## 示例：晶体场对称性

```python
from PySymmetry.core.group_theory.point_groups import (
    Oh, Td, D4h
)
import numpy as np

# 八面体群 Oh (立方晶体场)
oh = Oh()

print(f"群阶: {oh.order()}")     # 48
print(f"类数: {len(oh.conjugacy_classes)}")

# 不可约表示
print("不可约表示:")
for irrep in oh.irreps:
    print(f"  {irrep.name}: 维数 {irrep.dimension}")

# 分解可约表示
# t2g 轨道的分解
t2g_decomposition = oh.decompose('t2g')
print(f"t2g 分解: {t2g_decomposition}")
```
