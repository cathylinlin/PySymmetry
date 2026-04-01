"""抽象李代数基类

该模块定义了李代数的抽象基类，包括：
- LieAlgebraElement: 李代数元素抽象基类
- LieAlgebra: 李代数抽象基类
- LieAlgebraProperties: 李代数属性数据类

李代数是满足以下性质的向量空间：
1. 双线性李括号运算 [·, ·]: L × L → L
2. 反对称性：[x, y] = -[y, x]
3. 雅可比恒等式：[x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0

本模块提供了李代数的抽象接口，具体实现（如矩阵李代数）在子类中完成。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, TypeVar, Generic

# 类型变量，用于泛型李代数元素
T = TypeVar('T', bound='LieAlgebraElement')


class LieAlgebraElement(ABC):
    """
    李代数元素抽象基类
    
    李代数元素是李代数中的向量，支持：
    - 向量空间运算（加法、减法、标量乘法）
    - 李括号运算 [x, y]
    
    所有具体李代数元素类（如矩阵李代数元素）必须继承此类并实现抽象方法。
    """
    
    @abstractmethod
    def __add__(self, other: 'LieAlgebraElement') -> 'LieAlgebraElement':
        """
        加法运算
        
        李代数作为向量空间，支持元素间的加法运算。
        
        参数:
            other: 另一个李代数元素
            
        返回:
            两个元素的和
        """
        pass
    
    @abstractmethod
    def __sub__(self, other: 'LieAlgebraElement') -> 'LieAlgebraElement':
        """
        减法运算
        
        参数:
            other: 另一个李代数元素
            
        返回:
            两个元素的差
        """
        pass
    
    @abstractmethod
    def __mul__(self, scalar: float) -> 'LieAlgebraElement':
        """
        标量乘法
        
        李代数作为向量空间，支持实数标量乘法。
        
        参数:
            scalar: 实数标量
            
        返回:
            标量乘法结果
        """
        pass
    
    @abstractmethod
    def bracket(self, other: 'LieAlgebraElement') -> 'LieAlgebraElement':
        """
        李括号运算
        
        李代数的核心运算，满足反对称性和雅可比恒等式。
        
        参数:
            other: 另一个李代数元素
            
        返回:
            李括号 [self, other]
        """
        pass
    
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        相等性判断
        
        判断两个李代数元素是否相等。
        
        参数:
            other: 另一个对象
            
        返回:
            如果相等返回 True，否则返回 False
        """
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """
        字符串表示
        
        返回元素的可读字符串表示，便于调试和输出。
        """
        pass


class LieAlgebra(ABC, Generic[T]):
    """
    李代数抽象基类
    
    李代数是带有李括号运算的向量空间。本类定义了李代数的通用接口，
    包括向量空间结构、李括号运算以及基的操作。
    
    类型参数:
        T: 李代数元素的类型，必须是 LieAlgebraElement 的子类
    
    子类必须实现：
    - zero(): 返回零元素
    - bracket(): 李括号运算
    - add(), scalar_multiply(): 向量空间运算
    - basis(): 返回一组基
    - from_vector(), to_vector(): 向量与元素的转换
    - properties(): 返回李代数属性
    """
    
    @abstractmethod
    def __init__(self, dimension: int):
        """
        初始化李代数
        
        参数:
            dimension: 李代数作为实向量空间的维数
        """
        self.dimension = dimension
    
    @abstractmethod
    def zero(self) -> T:
        """
        返回零元素
        
        零元素 0 满足：对任意元素 x，有 [0, x] = 0
        
        返回:
            李代数的零元素
        """
        pass
    
    @abstractmethod
    def bracket(self, x: T, y: T) -> T:
        """
        李括号运算
        
        计算两个元素的李括号 [x, y]。
        
        参数:
            x: 第一个元素
            y: 第二个元素
            
        返回:
            李括号 [x, y]
        """
        pass
    
    @abstractmethod
    def add(self, x: T, y: T) -> T:
        """
        加法运算
        
        李代数作为向量空间的加法。
        
        参数:
            x: 第一个元素
            y: 第二个元素
            
        返回:
            和 x + y
        """
        pass
    
    @abstractmethod
    def scalar_multiply(self, x: T, scalar: float) -> T:
        """
        标量乘法
        
        李代数作为向量空间的标量乘法。
        
        参数:
            x: 李代数元素
            scalar: 实数标量
            
        返回:
            标量乘法结果 scalar · x
        """
        pass
    
    @abstractmethod
    def basis(self) -> List[T]:
        """
        返回李代数的一组基
        
        基是维数个线性无关的元素，可以张成整个李代数。
        
        返回:
            基元素列表，长度为 dimension
        """
        pass
    
    @abstractmethod
    def from_vector(self, vector: List[float]) -> T:
        """
        从向量创建李代数元素
        
        将实向量转换为李代数元素，使用当前基进行线性组合。
        
        参数:
            vector: 长度为 dimension 的实向量
            
        返回:
            对应的李代数元素
            
        异常:
            ValueError: 如果向量长度不正确
        """
        pass
    
    @abstractmethod
    def to_vector(self, element: T) -> List[float]:
        """
        将李代数元素转换为向量
        
        将李代数元素在当前基下的坐标表示为向量。
        
        参数:
            element: 李代数元素
            
        返回:
            长度为 dimension 的实向量
        """
        pass
    
    @abstractmethod
    def properties(self) -> 'LieAlgebraProperties':
        """
        返回李代数的属性
        
        包括名称、维数、半单性、单性、根系类型等。
        
        返回:
            LieAlgebraProperties 数据类实例
        """
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """
        字符串表示
        
        返回李代数的可读名称，如 "sl(3)"。
        """
        pass
        pass


@dataclass
class LieAlgebraProperties:
    """李代数属性数据类"""
    name: str  # 李代数名称
    dimension: int  # 李代数维度
    is_semisimple: bool  # 是否半单
    is_simple: bool  # 是否单
    is_abelian: bool  # 是否交换
    root_system_type: Optional[str] = None  # 根系类型
    rank: Optional[int] = None  # 秩
