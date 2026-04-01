"""具体矩阵李代数实现

提供 gl(n)、sl(n)、so(n)、sp(2n)、u(n)、su(n) 等经典实李代数的矩阵模型。

本模块实现了以下经典李代数：
- gl(n): 一般线性李代数（所有 n×n 矩阵）
- sl(n): 特殊线性李代数（迹为零的矩阵）
- so(n): 正交李代数（反对称矩阵）
- sp(2n): 辛李代数（保持辛形式的矩阵）
- u(n): 酉李代数（反厄米特矩阵）
- su(n): 特殊酉李代数（迹为零的反厄米特矩阵）

所有李代数都继承自 MatrixLieAlgebraBase，提供统一的矩阵表示接口。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import null_space

from .abstract_lie_algebra import LieAlgebra, LieAlgebraElement, LieAlgebraProperties


def _mat_bracket(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算矩阵李括号 [A, B] = AB - BA
    
    参数:
        a: 第一个矩阵
        b: 第二个矩阵
        
    返回:
        李括号 [A, B]
    """
    return a @ b - b @ a


class MatrixLieAlgebraElement(LieAlgebraElement):
    """
    矩阵李代数元素（实或复矩阵）
    
    这是李代数元素的具体实现，使用 NumPy 数组存储矩阵数据。
    支持基本的线性运算（加法、减法、标量乘法）和李括号运算。
    
    属性:
        matrix: 存储的 NumPy 矩阵
        lie_algebra: 所属的李代数
    """

    def __init__(self, matrix: np.ndarray, lie_algebra: "MatrixLieAlgebraBase"):
        """
        初始化矩阵李代数元素
        
        参数:
            matrix: NumPy 数组表示的矩阵
            lie_algebra: 所属的李代数对象
        """
        self.matrix = np.asarray(matrix)
        self._lie = lie_algebra

    @property
    def lie_algebra(self) -> "MatrixLieAlgebraBase":
        """返回所属的李代数"""
        return self._lie

    def __add__(self, other: LieAlgebraElement) -> "MatrixLieAlgebraElement":
        """
        矩阵加法
        
        要求两个元素属于同一个李代数
        """
        if not isinstance(other, MatrixLieAlgebraElement) or other._lie is not self._lie:
            return NotImplemented
        return MatrixLieAlgebraElement(self.matrix + other.matrix, self._lie)

    def __sub__(self, other: LieAlgebraElement) -> "MatrixLieAlgebraElement":
        """
        矩阵减法
        
        要求两个元素属于同一个李代数
        """
        if not isinstance(other, MatrixLieAlgebraElement) or other._lie is not self._lie:
            return NotImplemented
        return MatrixLieAlgebraElement(self.matrix - other.matrix, self._lie)

    def __mul__(self, scalar: float) -> "MatrixLieAlgebraElement":
        """标量乘法"""
        return MatrixLieAlgebraElement(self.matrix * scalar, self._lie)

    def bracket(self, other: LieAlgebraElement) -> "MatrixLieAlgebraElement":
        """
        计算李括号 [self, other] = self·other - other·self
        
        参数:
            other: 另一个李代数元素
            
        异常:
            TypeError: 如果 other 不属于同一个李代数
        """
        if not isinstance(other, MatrixLieAlgebraElement) or other._lie is not self._lie:
            raise TypeError("李括号仅对同一李代数中的元素定义")
        return MatrixLieAlgebraElement(_mat_bracket(self.matrix, other.matrix), self._lie)

    def __eq__(self, other: object) -> bool:
        """
        判断两个李代数元素是否相等
        
        使用 np.allclose 进行浮点数比较
        """
        if not isinstance(other, MatrixLieAlgebraElement):
            return False
        if other._lie is not self._lie:
            return False
        return np.allclose(self.matrix, other.matrix)

    def __str__(self) -> str:
        """字符串表示"""
        return str(self.matrix)


class MatrixLieAlgebraBase(LieAlgebra[MatrixLieAlgebraElement]):
    """
    矩阵李代数抽象基类
    
    为所有矩阵李代数提供公共逻辑，包括：
    - 零元素的创建
    - 基本运算（加法、李括号、标量乘法）
    - 矩阵形状和数据类型的抽象接口
    
    子类必须实现：
    - _matrix_shape(): 返回矩阵的形状
    - basis(): 返回李代数的一组基
    - from_vector(): 从向量创建元素
    - to_vector(): 将元素转换为向量
    - properties(): 返回李代数属性
    """

    def __init__(self, dimension: int):
        """
        初始化矩阵李代数
        
        参数:
            dimension: 李代数的维数（作为实向量空间）
        """
        self.dimension = dimension

    def zero(self) -> MatrixLieAlgebraElement:
        """返回李代数的零元素"""
        return MatrixLieAlgebraElement(
            np.zeros(self._matrix_shape(), dtype=self._dtype()), self
        )

    def bracket(self, x: MatrixLieAlgebraElement, y: MatrixLieAlgebraElement) -> MatrixLieAlgebraElement:
        """计算李括号 [x, y]"""
        return x.bracket(y)

    def add(self, x: MatrixLieAlgebraElement, y: MatrixLieAlgebraElement) -> MatrixLieAlgebraElement:
        """加法运算"""
        return x + y

    def scalar_multiply(self, x: MatrixLieAlgebraElement, scalar: float) -> MatrixLieAlgebraElement:
        """标量乘法"""
        return x * scalar

    def _matrix_shape(self) -> Tuple[int, ...]:
        """
        返回矩阵的形状
        
        子类必须实现此方法
        """
        raise NotImplementedError

    def _dtype(self) -> np.dtype:
        """
        返回矩阵的数据类型
        
        默认为 float64，复李代数应覆盖为 complex128
        """
        return np.float64


class GeneralLinearLieAlgebra(MatrixLieAlgebraBase):
    """
    一般线性李代数 gl(n, R)
    
    由全体 n×n 实矩阵组成，李括号定义为矩阵交换子 [A,B] = AB - BA。
    维数为 n²，不是半单李代数。
    
    属性:
        n: 矩阵维度
        dimension: 李代数维数 = n²
    """

    def __init__(self, n: int):
        """
        初始化 gl(n)
        
        参数:
            n: 矩阵维度，必须为正整数
            
        异常:
            ValueError: 如果 n < 1
        """
        if n < 1:
            raise ValueError("n 必须为正整数")
        self.n = n
        super().__init__(n * n)

    def _matrix_shape(self) -> Tuple[int, int]:
        """返回矩阵形状 (n, n)"""
        return (self.n, self.n)

    def basis(self) -> List[MatrixLieAlgebraElement]:
        """
        返回标准基 {E_ij}，其中 E_ij 在第 (i,j) 位置为 1，其余为 0
        
        基的顺序：按行优先排列，即 E_11, E_12, ..., E_1n, E_21, ...
        """
        mats = []
        for i in range(self.n):
            for j in range(self.n):
                e = np.zeros((self.n, self.n))
                e[i, j] = 1.0
                mats.append(MatrixLieAlgebraElement(e, self))
        return mats

    def from_vector(self, vector: List[float]) -> MatrixLieAlgebraElement:
        """
        从向量重建矩阵元素
        
        参数:
            vector: 长度为 n² 的向量，按行优先排列
            
        异常:
            ValueError: 如果向量长度不正确
        """
        if len(vector) != self.dimension:
            raise ValueError(f"期望长度为 {self.dimension} 的向量")
        arr = np.asarray(vector, dtype=np.float64).reshape(self.n, self.n)
        return MatrixLieAlgebraElement(arr, self)

    def to_vector(self, element: MatrixLieAlgebraElement) -> List[float]:
        """将矩阵元素展平为向量（行优先）"""
        return element.matrix.reshape(-1).tolist()

    def properties(self) -> LieAlgebraProperties:
        """
        返回李代数属性
        
        gl(n) 的性质：
        - 不是半单李代数
        - 不是单李代数
        - 当 n=1 时是阿贝尔的
        - 没有根系（不是半单的）
        """
        return LieAlgebraProperties(
            name=f"gl({self.n})",
            dimension=self.dimension,
            is_semisimple=False,
            is_simple=False,
            is_abelian=self.n == 1,
            root_system_type=None,
            rank=self.n,
        )

    def __str__(self) -> str:
        return f"gl({self.n})"


class SpecialLinearLieAlgebra(MatrixLieAlgebraBase):
    """
    特殊线性李代数 sl(n, R)
    
    由迹为零的 n×n 实矩阵组成，是 gl(n) 的子代数。
    维数为 n² - 1，是 A_{n-1} 型单李代数。
    
    属性:
        n: 矩阵维度
        dimension: 李代数维数 = n² - 1
    """

    def __init__(self, n: int):
        """
        初始化 sl(n)
        
        参数:
            n: 矩阵维度，必须 >= 2
            
        异常:
            ValueError: 如果 n < 2
        """
        if n < 2:
            raise ValueError("sl(n) 要求 n >= 2")
        self.n = n
        super().__init__(n * n - 1)

    def _matrix_shape(self) -> Tuple[int, int]:
        """返回矩阵形状 (n, n)"""
        return (self.n, self.n)

    def basis(self) -> List[MatrixLieAlgebraElement]:
        """
        返回 sl(n) 的标准基
        
        基由两部分组成：
        1. 非对角元矩阵 E_ij (i≠j)：共 n² - n 个
        2. 对角元矩阵 h_k = E_kk - E_{k+1,k+1}：共 n-1 个
        
        总维数：(n² - n) + (n - 1) = n² - 1
        """
        mats: List[MatrixLieAlgebraElement] = []
        # 非对角元基
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                e = np.zeros((self.n, self.n))
                e[i, j] = 1.0
                mats.append(MatrixLieAlgebraElement(e, self))
        # 对角元基（Cartan 子代数）
        for k in range(self.n - 1):
            h = np.zeros((self.n, self.n))
            h[k, k] = 1.0
            h[k + 1, k + 1] = -1.0
            mats.append(MatrixLieAlgebraElement(h, self))
        return mats

    def from_vector(self, vector: List[float]) -> MatrixLieAlgebraElement:
        """
        从向量重建 sl(n) 元素
        
        向量格式：
        - 前 n² - n 个分量：非对角元（按行优先）
        - 后 n - 1 个分量：对角元基 h_k 的系数
        """
        if len(vector) != self.dimension:
            raise ValueError(f"期望长度为 {self.dimension} 的向量")
        m = np.zeros((self.n, self.n), dtype=np.float64)
        idx = 0
        # 非对角元
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                m[i, j] = vector[idx]
                idx += 1
        # 对角元
        for k in range(self.n - 1):
            m[k, k] += vector[idx]
            m[k + 1, k + 1] -= vector[idx]
            idx += 1
        return MatrixLieAlgebraElement(m, self)

    def to_vector(self, element: MatrixLieAlgebraElement) -> List[float]:
        """
        将 sl(n) 元素转换为向量
        
        非对角元直接读取，对角元通过累积迹计算
        """
        m = element.matrix
        out: List[float] = []
        # 非对角元
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                out.append(float(m[i, j]))
        # 对角元系数
        for k in range(self.n - 1):
            out.append(float(sum(m[i, i] for i in range(k + 1))))
        return out

    def properties(self) -> LieAlgebraProperties:
        """
        返回 sl(n) 的属性
        
        sl(n) 的性质：
        - 半单李代数
        - 单李代数（当 n >= 2）
        - 根系类型：A_{n-1}
        - 秩：n - 1
        """
        return LieAlgebraProperties(
            name=f"sl({self.n})",
            dimension=self.dimension,
            is_semisimple=True,
            is_simple=self.n >= 2,
            is_abelian=False,
            root_system_type=f"A{self.n - 1}",
            rank=self.n - 1,
        )

    def __str__(self) -> str:
        return f"sl({self.n})"


class OrthogonalLieAlgebra(MatrixLieAlgebraBase):
    """
    正交李代数 so(n)
    
    由实反对称 n×n 矩阵组成，满足 X^T = -X。
    维数为 n(n-1)/2，是经典李代数 B 型（n 奇数）或 D 型（n 偶数）。
    
    属性:
        n: 矩阵维度
        dimension: 李代数维数 = n(n-1)/2
    """

    def __init__(self, n: int):
        """
        初始化 so(n)
        
        参数:
            n: 矩阵维度，必须 >= 2
            
        异常:
            ValueError: 如果 n < 2
        """
        if n < 2:
            raise ValueError("so(n) 要求 n >= 2")
        self.n = n
        super().__init__(n * (n - 1) // 2)

    def _matrix_shape(self) -> Tuple[int, int]:
        """返回矩阵形状 (n, n)"""
        return (self.n, self.n)

    def basis(self) -> List[MatrixLieAlgebraElement]:
        """
        返回 so(n) 的标准基
        
        基矩阵 L_ij = E_ij - E_ji (i < j)，共 n(n-1)/2 个
        这些矩阵生成 n 维空间中的旋转变换
        """
        mats = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                e = np.zeros((self.n, self.n))
                e[i, j] = 1.0
                e[j, i] = -1.0
                mats.append(MatrixLieAlgebraElement(e, self))
        return mats

    def from_vector(self, vector: List[float]) -> MatrixLieAlgebraElement:
        """
        从向量重建反对称矩阵
        
        向量包含上三角部分的元素（i < j），下三角通过对称性确定
        """
        if len(vector) != self.dimension:
            raise ValueError(f"期望长度为 {self.dimension} 的向量")
        m = np.zeros((self.n, self.n), dtype=np.float64)
        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                m[i, j] = vector[idx]
                m[j, i] = -vector[idx]  # 反对称性
                idx += 1
        return MatrixLieAlgebraElement(m, self)

    def to_vector(self, element: MatrixLieAlgebraElement) -> List[float]:
        """
        将反对称矩阵转换为向量
        
        只提取上三角部分（i < j）的元素
        """
        m = element.matrix
        out: List[float] = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                out.append(float(m[i, j]))
        return out

    def properties(self) -> LieAlgebraProperties:
        """
        返回 so(n) 的属性
        
        so(n) 的性质：
        - 当 n >= 3 时是半单李代数
        - 当 n >= 3 且 n ≠ 4 时是单李代数（so(4) ≅ so(3) ⊕ so(3)）
        - 根系类型：B_{(n-1)/2}（n 奇数）或 D_{n/2}（n 偶数）
        - 秩：⌊n/2⌋
        """
        t = f"D{self.n // 2}" if self.n % 2 == 0 else f"B{(self.n - 1) // 2}"
        return LieAlgebraProperties(
            name=f"so({self.n})",
            dimension=self.dimension,
            is_semisimple=self.n >= 3,
            is_simple=self.n >= 3 and self.n != 4,
            is_abelian=False,
            root_system_type=t,
            rank=self.n // 2,
        )

    def __str__(self) -> str:
        return f"so({self.n})"


def _symplectic_J(n: int) -> np.ndarray:
    """
    构造标准辛形式矩阵 J
    
    J = [[0,  I_n],
         [-I_n, 0]]  (2n×2n 矩阵)
    
    辛形式满足 J^T = -J（反对称）且 det(J) = 1
    
    参数:
        n: 辛空间的半维数
        
    返回:
        2n×2n 的标准辛形式矩阵
    """
    d = 2 * n
    j = np.zeros((d, d), dtype=np.float64)
    j[:n, n:] = np.eye(n)    # 右上块为单位矩阵
    j[n:, :n] = -np.eye(n)   # 左下块为负单位矩阵
    return j


def _sp2n_basis_matrices(n: int) -> List[np.ndarray]:
    """
    计算 sp(2n) 的一组基矩阵（数学优化版本）
    
    数学优化：使用 sp(2n) 的显式块结构公式，而非数值求解零空间。
    
    sp(2n) 由满足 X^T J + J X = 0 的 2n×2n 矩阵组成。
    将 X 写成分块形式 X = [[A, B], [C, D]]，其中 A,B,C,D 是 n×n 矩阵，
    则条件等价于：
    - D = -A^T（A 任意，D 由 A 决定）
    - B = B^T（B 对称）
    - C = C^T（C 对称）
    
    维数计算：
    - A 有 n² 个自由度
    - B 有 n(n+1)/2 个自由度（对称）
    - C 有 n(n+1)/2 个自由度（对称）
    - 总计：n² + n(n+1) = n(2n+1)
    
    参数:
        n: 辛空间的半维数
        
    返回:
        sp(2n) 的基矩阵列表，维数为 n(2n+1)
        
    数学参考：
        - Helgason, "Differential Geometry, Lie Groups, and Symmetric Spaces"
        - 该构造保证数值稳定性，避免 null_space 的数值误差
    """
    d = 2 * n
    basis: List[np.ndarray] = []
    
    # 类型 1: A 矩阵基（n² 个）
    # D = -A^T, B = 0, C = 0
    for i in range(n):
        for j in range(n):
            mat = np.zeros((d, d), dtype=np.float64)
            mat[i, j] = 1.0  # A 的 (i,j) 位置
            mat[n + j, n + i] = -1.0  # D = -A^T 的 (j,i) 位置
            basis.append(mat)
    
    # 类型 2: B 对称矩阵基（n(n+1)/2 个）
    # A = 0, D = 0, C = 0, B = B^T
    for i in range(n):
        for j in range(i, n):
            mat = np.zeros((d, d), dtype=np.float64)
            mat[i, n + j] = 1.0
            mat[j, n + i] = 1.0  # 对称性
            basis.append(mat)
    
    # 类型 3: C 对称矩阵基（n(n+1)/2 个）
    # A = 0, D = 0, B = 0, C = C^T
    for i in range(n):
        for j in range(i, n):
            mat = np.zeros((d, d), dtype=np.float64)
            mat[n + i, j] = 1.0
            mat[n + j, i] = 1.0  # 对称性
            basis.append(mat)
    
    return basis


class SymplecticLieAlgebra(MatrixLieAlgebraBase):
    """
    辛李代数 sp(2n, R)
    
    由满足 X^T J + J X = 0 的 2n×2n 实矩阵组成，其中 J 是标准辛形式。
    这是辛群 Sp(2n) 的李代数，保持辛形式不变。
    
    维数为 n(2n+1)，是 C_n 型单李代数。
    
    属性:
        n: 辛空间的半维数
        _d: 完整空间维数 = 2n
        dimension: 李代数维数 = n(2n+1)
    """

    def __init__(self, n: int):
        """
        初始化 sp(2n)
        
        参数:
            n: 辛空间的半维数，必须为正整数
            
        异常:
            ValueError: 如果 n < 1
        """
        if n < 1:
            raise ValueError("n 必须为正整数")
        self.n = n
        self._d = 2 * n
        # 预计算基矩阵以提高性能
        self._basis_mats = _sp2n_basis_matrices(n)
        super().__init__(len(self._basis_mats))

    def _matrix_shape(self) -> Tuple[int, int]:
        """返回矩阵形状 (2n, 2n)"""
        return (self._d, self._d)

    def basis(self) -> List[MatrixLieAlgebraElement]:
        """返回预计算的基矩阵列表"""
        return [MatrixLieAlgebraElement(np.array(m), self) for m in self._basis_mats]

    def from_vector(self, vector: List[float]) -> MatrixLieAlgebraElement:
        """
        从向量重建 sp(2n) 元素
        
        通过基矩阵的线性组合构造矩阵
        """
        if len(vector) != self.dimension:
            raise ValueError(f"期望长度为 {self.dimension} 的向量")
        m = np.zeros((self._d, self._d), dtype=np.float64)
        for c, v in enumerate(vector):
            m += v * self._basis_mats[c]
        return MatrixLieAlgebraElement(m, self)

    def to_vector(self, element: MatrixLieAlgebraElement) -> List[float]:
        """
        将 sp(2n) 元素转换为向量
        
        由于基一般不正交，使用最小二乘法在基上展开
        """
        b = element.matrix.flatten()
        a = np.column_stack([bm.flatten() for bm in self._basis_mats])
        coef, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        return coef.tolist()

    def properties(self) -> LieAlgebraProperties:
        """
        返回 sp(2n) 的属性
        
        sp(2n) 的性质：
        - 半单李代数
        - 单李代数（当 n >= 1）
        - 根系类型：C_n
        - 秩：n
        """
        return LieAlgebraProperties(
            name=f"sp({2 * self.n})",
            dimension=self.dimension,
            is_semisimple=True,
            is_simple=self.n >= 1,
            is_abelian=False,
            root_system_type=f"C{self.n}",
            rank=self.n,
        )

    def __str__(self) -> str:
        return f"sp({2 * self.n})"


def _skew_hermitian_to_vector(m: np.ndarray) -> List[float]:
    """
    将反厄米特矩阵转换为实向量
    
    向量格式（共 n² 个分量）：
    - 前 n 个：对角元的虚部（实部为零）
    - 后 n(n-1) 个：上三角非对角元的实部和虚部（交错排列）
    
    参数:
        m: n×n 反厄米特矩阵（m^† = -m）
        
    返回:
        长度为 n² 的实向量
    """
    n = m.shape[0]
    out: List[float] = []
    # 对角元：纯虚数，取虚部
    for i in range(n):
        out.append(float(np.imag(m[i, i])))
    # 非对角元：实部和虚部分开存储
    for i in range(n):
        for j in range(i + 1, n):
            out.append(float(np.real(m[i, j])))
            out.append(float(np.imag(m[i, j])))
    return out


def _vector_to_skew_hermitian(v: List[float], n: int) -> np.ndarray:
    """
    从实向量重建反厄米特矩阵（u(n) 版本）
    
    参数:
        v: 长度为 n² 的实向量
        n: 矩阵维度
        
    返回:
        n×n 反厄米特矩阵
        
    异常:
        ValueError: 如果向量长度不正确
    """
    if len(v) != n * n:
        raise ValueError("向量长度与 u(n) 维数不符")
    m = np.zeros((n, n), dtype=np.complex128)
    idx = 0
    # 对角元：纯虚数
    for i in range(n):
        m[i, i] = 1j * v[idx]
        idx += 1
    # 非对角元：利用反厄米特性 m_ji = -m_ij^*
    for i in range(n):
        for j in range(i + 1, n):
            re = v[idx]
            im = v[idx + 1]
            idx += 2
            m[i, j] = re + 1j * im
            m[j, i] = -re + 1j * im
    return m


def _vector_to_skew_hermitian_su(v: List[float], n: int) -> np.ndarray:
    """
    从实向量重建反厄米特矩阵（su(n) 版本，迹为零）
    
    与 u(n) 的区别：对角元不是独立的，而是满足迹为零的约束
    
    参数:
        v: 长度为 n²-1 的实向量
        n: 矩阵维度
        
    返回:
        n×n 迹为零的反厄米特矩阵
        
    异常:
        ValueError: 如果向量长度不正确
    """
    if len(v) != n * n - 1:
        raise ValueError("向量长度与 su(n) 维数不符")
    m = np.zeros((n, n), dtype=np.complex128)
    idx = 0
    # 对角元：前 n-1 个独立，最后一个由迹为零确定
    for k in range(n - 1):
        c = v[idx]
        idx += 1
        m[k, k] += 1j * c
        m[n - 1, n - 1] -= 1j * c  # 保持迹为零
    # 非对角元：与 u(n) 相同
    for i in range(n):
        for j in range(i + 1, n):
            re = v[idx]
            im = v[idx + 1]
            idx += 2
            m[i, j] = re + 1j * im
            m[j, i] = -re + 1j * im
    return m


def _skew_hermitian_su_to_vector(m: np.ndarray) -> List[float]:
    """
    将迹为零的反厄米特矩阵转换为实向量（su(n) 版本）
    
    向量格式（共 n²-1 个分量）：
    - 前 n-1 个：前 n-1 个对角元的虚部
    - 后 n²-n-1 个：上三角非对角元的实部和虚部
    
    参数:
        m: n×n 迹为零的反厄米特矩阵
        
    返回:
        长度为 n²-1 的实向量
    """
    n = m.shape[0]
    out: List[float] = []
    # 对角元：只取前 n-1 个（最后一个由迹为零确定）
    for i in range(n - 1):
        out.append(float(np.imag(m[i, i])))
    # 非对角元
    for i in range(n):
        for j in range(i + 1, n):
            out.append(float(np.real(m[i, j])))
            out.append(float(np.imag(m[i, j])))
    return out


class UnitaryLieAlgebra(MatrixLieAlgebraBase):
    """
    酉李代数 u(n)
    
    由 n×n 反厄米特矩阵组成（X^† = -X），实维数为 n²。
    这是酉群 U(n) 的李代数，通过指数映射 exp: u(n) → U(n) 连接。
    
    注：若文献使用厄米生成元 H，常见关系为 H = -iX（X 为本库中的反厄米元素）。
    结构常数、Casimir 算符等需与所选约定一致。
    
    属性:
        n: 矩阵维度
        dimension: 李代数维数 = n²
    """

    def __init__(self, n: int):
        """
        初始化 u(n)
        
        参数:
            n: 矩阵维度，必须为正整数
            
        异常:
            ValueError: 如果 n < 1
        """
        if n < 1:
            raise ValueError("n 必须为正整数")
        self.n = n
        super().__init__(n * n)

    def _matrix_shape(self) -> Tuple[int, int]:
        """返回矩阵形状 (n, n)"""
        return (self.n, self.n)

    def _dtype(self) -> np.dtype:
        """返回复数数据类型"""
        return np.complex128

    def basis(self) -> List[MatrixLieAlgebraElement]:
        """
        返回 u(n) 的标准基
        
        基由以下矩阵组成：
        1. 对角元：i·E_kk（n 个，纯虚对角矩阵）
        2. 非对角元实部：E_ij - E_ji（n(n-1)/2 个，实反对称）
        3. 非对角元虚部：i(E_ij + E_ji)（n(n-1)/2 个，虚对称）
        
        总维数：n + n(n-1) = n²
        """
        mats: List[MatrixLieAlgebraElement] = []
        # 对角元基
        for i in range(self.n):
            e = np.zeros((self.n, self.n), dtype=np.complex128)
            e[i, i] = 1j
            mats.append(MatrixLieAlgebraElement(e, self))
        # 非对角元基
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # 实部（反对称）
                e = np.zeros((self.n, self.n), dtype=np.complex128)
                e[i, j] = 1.0
                e[j, i] = -1.0
                mats.append(MatrixLieAlgebraElement(e, self))
                # 虚部（对称）
                f = np.zeros((self.n, self.n), dtype=np.complex128)
                f[i, j] = 1j
                f[j, i] = 1j
                mats.append(MatrixLieAlgebraElement(f, self))
        return mats

    def from_vector(self, vector: List[float]) -> MatrixLieAlgebraElement:
        """从向量重建反厄米特矩阵"""
        m = _vector_to_skew_hermitian(vector, self.n)
        return MatrixLieAlgebraElement(m, self)

    def to_vector(self, element: MatrixLieAlgebraElement) -> List[float]:
        """将反厄米特矩阵转换为向量"""
        return _skew_hermitian_to_vector(element.matrix)

    def properties(self) -> LieAlgebraProperties:
        """
        返回 u(n) 的属性
        
        u(n) 的性质：
        - 不是半单李代数（有非平凡中心）
        - 不是单李代数
        - 当 n=1 时是阿贝尔的（u(1) ≅ R）
        - 没有根系（不是半单的）
        - 可以分解为：u(n) ≅ su(n) ⊕ u(1)
        """
        return LieAlgebraProperties(
            name=f"u({self.n})",
            dimension=self.dimension,
            is_semisimple=False,
            is_simple=False,
            is_abelian=self.n == 1,
            root_system_type=None,
            rank=self.n,
        )

    def __str__(self) -> str:
        return f"u({self.n})"


class SpecialUnitaryLieAlgebra(MatrixLieAlgebraBase):
    """
    特殊酉李代数 su(n)
    
    由迹为零的 n×n 反厄米特矩阵组成，实维数为 n² - 1。
    这是特殊酉群 SU(n) 的李代数，是 u(n) 的子代数。
    
    同样采用反厄米模型；厄米生成元与反厄米李代数元素的关系与 u(n) 相同（通常 H = -iX）。
    
    属性:
        n: 矩阵维度
        dimension: 李代数维数 = n² - 1
    """

    def __init__(self, n: int):
        """
        初始化 su(n)
        
        参数:
            n: 矩阵维度，必须 >= 2
            
        异常:
            ValueError: 如果 n < 2
        """
        if n < 2:
            raise ValueError("su(n) 要求 n >= 2")
        self.n = n
        super().__init__(n * n - 1)

    def _matrix_shape(self) -> Tuple[int, int]:
        """返回矩阵形状 (n, n)"""
        return (self.n, self.n)

    def _dtype(self) -> np.dtype:
        """返回复数数据类型"""
        return np.complex128

    def basis(self) -> List[MatrixLieAlgebraElement]:
        """
        返回 su(n) 的标准基（Gell-Mann 矩阵的推广）
        
        基由以下矩阵组成：
        1. 对角元：h_k = i(E_kk - E_nn)（n-1 个，Cartan 子代数）
        2. 非对角元实部：E_ij - E_ji（n(n-1)/2 个）
        3. 非对角元虚部：i(E_ij + E_ji)（n(n-1)/2 个）
        
        总维数：(n-1) + n(n-1) = n² - 1
        """
        mats: List[MatrixLieAlgebraElement] = []
        # 对角元基（Cartan 子代数）
        for k in range(self.n - 1):
            h = np.zeros((self.n, self.n), dtype=np.complex128)
            h[k, k] = 1j
            h[self.n - 1, self.n - 1] = -1j
            mats.append(MatrixLieAlgebraElement(h, self))
        # 非对角元基
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # 实部
                e = np.zeros((self.n, self.n), dtype=np.complex128)
                e[i, j] = 1.0
                e[j, i] = -1.0
                mats.append(MatrixLieAlgebraElement(e, self))
                # 虚部
                f = np.zeros((self.n, self.n), dtype=np.complex128)
                f[i, j] = 1j
                f[j, i] = 1j
                mats.append(MatrixLieAlgebraElement(f, self))
        return mats

    def from_vector(self, vector: List[float]) -> MatrixLieAlgebraElement:
        """从向量重建迹为零的反厄米特矩阵"""
        m = _vector_to_skew_hermitian_su(vector, self.n)
        return MatrixLieAlgebraElement(m, self)

    def to_vector(self, element: MatrixLieAlgebraElement) -> List[float]:
        """将迹为零的反厄米特矩阵转换为向量"""
        return _skew_hermitian_su_to_vector(element.matrix)

    def properties(self) -> LieAlgebraProperties:
        """
        返回 su(n) 的属性
        
        su(n) 的性质：
        - 半单李代数
        - 单李代数
        - 根系类型：A_{n-1}
        - 秩：n - 1
        
        特别地：
        - su(2) ≅ so(3)（同构）
        - su(2) 的基就是泡利矩阵（乘以 i）
        """
        return LieAlgebraProperties(
            name=f"su({self.n})",
            dimension=self.dimension,
            is_semisimple=True,
            is_simple=True,
            is_abelian=False,
            root_system_type=f"A{self.n - 1}",
            rank=self.n - 1,
        )

    def __str__(self) -> str:
        return f"su({self.n})"
