# PySymmetry 文档

欢迎使用 PySymmetry！本项目提供对称性理论的 Python 实现。

## 文档目录

### 教程
- [快速开始](../tutorials/getting_started.md) - 安装和入门
- [矩阵运算](../tutorials/matrix_tutorial.md) - 矩阵创建、运算、分解
- [量子力学](../tutorials/quantum_tutorial.md) - 量子态、测量、纠缠
- [群论入门](../tutorials/group_theory_tutorial.md) - 有限群、李群、表示论
- [可视化](../tutorials/visualization_guide.md) - 图形和动画

### API 参考
- [core.matrix](./core_matrix.md) - 矩阵运算模块
- [quantum_states](./quantum_states.md) - 量子态 API
- [tools.performance](./tools_performance.md) - 性能优化
- [量子模块完整文档](../quantum_module.md) - 详细使用指南

---

## 模块概览

```
PySymmetry/
├── core/                    # 核心数学模块
│   ├── matrix/             # 矩阵运算
│   │   ├── factory.py     # 矩阵工厂
│   │   ├── operations.py  # 矩阵运算
│   │   └── decompositions.py # 矩阵分解
│   ├── group_theory/       # 群论
│   ├── lie_theory/         # 李代数
│   └── representation/     # 表示论
│
├── phys/                   # 物理模块
│   └── quantum/           # 量子物理
│       ├── states.py      # 量子态
│       ├── hamiltonian.py # 哈密顿算符
│       └── solver.py      # 求解器
│
├── abstract_phys/          # 抽象物理框架
│   ├── symmetry_operations/
│   ├── symmetry_environments/
│   └── physical_objects/
│
├── tools/                  # 工具模块
│   └── performance.py     # 性能优化
│
└── visual/                # 可视化模块
```
