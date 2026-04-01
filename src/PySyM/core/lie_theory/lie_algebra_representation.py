"""Lie algebra representations.

Provides:
- LieAlgebraRepresentation: abstract representation
- AdjointRepresentation: adjoint representation
- FundamentalRepresentation: defining representation (matrix elements)
- TensorProductRepresentation: tensor product of two representations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from .abstract_lie_algebra import LieAlgebra, LieAlgebraElement
from PySyM.core.matrix.factory import MatrixFactory

T = TypeVar("T", bound=LieAlgebraElement)


class LieAlgebraRepresentation(ABC, Generic[T]):
    """Abstract Lie algebra representation."""

    def __init__(self, lie_algebra: LieAlgebra[T], dimension: int):
        self.lie_algebra = lie_algebra
        self.dimension = dimension

    @abstractmethod
    def __call__(self, element: T) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def differential(self, element: T) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def is_irreducible(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def character(self, element: T) -> complex:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class AdjointRepresentation(LieAlgebraRepresentation[T]):
    """Adjoint representation: ad_X(Y) = [X, Y] in the chosen basis."""

    def __init__(self, lie_algebra: LieAlgebra[T]):
        super().__init__(lie_algebra, lie_algebra.dimension)

    def __call__(self, element: T) -> np.ndarray:
        basis = self.lie_algebra.basis()
        dim = self.dimension
        matrix = MatrixFactory.zeros(dim, dim)

        for i, y in enumerate(basis):
            ad_xy = self.lie_algebra.bracket(element, y)
            coordinates = self.lie_algebra.to_vector(ad_xy)
            for j, coord in enumerate(coordinates):
                matrix[j, i] = coord

        return matrix

    def differential(self, element: T) -> np.ndarray:
        return self(element)

    def is_irreducible(self) -> bool:
        return self.lie_algebra.properties().is_simple

    def character(self, element: T) -> complex:
        matrix = self(element)
        return complex(np.trace(matrix))

    def __str__(self) -> str:
        return f"AdjointRepresentation({self.lie_algebra})"


class FundamentalRepresentation(LieAlgebraRepresentation[T]):
    """Defining representation for elements exposing a .matrix attribute."""

    def __init__(self, lie_algebra: LieAlgebra[T], dimension: int):
        super().__init__(lie_algebra, dimension)

    def __call__(self, element: T) -> np.ndarray:
        if hasattr(element, "matrix"):
            m = getattr(element, "matrix")
            return np.asarray(m, dtype=np.complex128)
        raise NotImplementedError("Fundamental representation requires matrix Lie algebra elements")

    def differential(self, element: T) -> np.ndarray:
        return self(element)

    def is_irreducible(self) -> bool:
        return True

    def character(self, element: T) -> complex:
        matrix = self(element)
        return complex(np.trace(matrix))

    def __str__(self) -> str:
        return f"FundamentalRepresentation({self.lie_algebra}, dim={self.dimension})"


class TensorProductRepresentation(LieAlgebraRepresentation[T]):
    """Tensor product: rho1 (x) rho2 (X) = rho1(X) kronecker I + I kronecker rho2(X)."""

    def __init__(
        self,
        rep1: LieAlgebraRepresentation[T],
        rep2: LieAlgebraRepresentation[T],
    ):
        if rep1.lie_algebra is not rep2.lie_algebra:
            raise ValueError("Both representations must use the same Lie algebra instance")
        dimension = rep1.dimension * rep2.dimension
        super().__init__(rep1.lie_algebra, dimension)
        self.rep1 = rep1
        self.rep2 = rep2

    def __call__(self, element: T) -> np.ndarray:
        mat1 = np.asarray(self.rep1(element))
        mat2 = np.asarray(self.rep2(element))
        dim1 = self.rep1.dimension
        dim2 = self.rep2.dimension

        i1 = MatrixFactory.identity(dim1)
        i2 = MatrixFactory.identity(dim2)

        term1 = self._kronecker_product(mat1, i2)
        term2 = self._kronecker_product(i1, mat2)

        return term1 + term2

    def _kronecker_product(
        self, mat1: np.ndarray, mat2: np.ndarray
    ) -> np.ndarray:
        """
        计算 Kronecker 积（优化版本）
        
        数学优化：使用 NumPy 的 np.kron 函数，该函数使用优化的 BLAS 例程，
        比手动实现快 10-100 倍，且内存效率更高。
        
        对于稀疏矩阵，可考虑使用 scipy.sparse.kron 进一步优化。
        
        参数:
            mat1: 第一个矩阵
            mat2: 第二个矩阵
            
        返回:
            Kronecker 积 mat1 ⊗ mat2
        """
        # 使用 NumPy 优化的 Kronecker 积
        # 这比手动循环实现快得多，且利用 SIMD 指令
        return np.kron(mat1, mat2)

    def differential(self, element: T) -> np.ndarray:
        return self(element)

    def is_irreducible(self) -> bool:
        return False

    def character(self, element: T) -> complex:
        char1 = self.rep1.character(element)
        char2 = self.rep2.character(element)
        return char1 * char2

    def __str__(self) -> str:
        return f"TensorProductRepresentation({self.rep1} x {self.rep2})"
