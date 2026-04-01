"""李代数性能测试

验证数学优化的性能提升效果。
"""

import sys
import time
import unittest
from pathlib import Path

import numpy as np

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PySyM.core.lie_theory.specific_lie_algebra import (
    SpecialLinearLieAlgebra,
    OrthogonalLieAlgebra,
    SymplecticLieAlgebra,
)
from PySyM.core.lie_theory.lie_algebra_structure import KillingForm


class TestLieAlgebraPerformance(unittest.TestCase):
    """性能测试类"""

    def test_killing_form_performance(self):
        """测试基灵型计算的性能"""
        # 测试不同维度的李代数
        dimensions = [2, 3, 4, 5]
        
        for n in dimensions:
            algebra = SpecialLinearLieAlgebra(n)
            
            # 计时基灵型计算
            start = time.perf_counter()
            killing = KillingForm(algebra)
            elapsed = time.perf_counter() - start
            
            # 验证结果正确性
            self.assertEqual(killing.matrix.shape, (algebra.dimension, algebra.dimension))
            
            # 对于 sl(n)，基灵型应该是非退化的（半单李代数）
            self.assertTrue(killing.is_non_degenerate())
            
            print(f"sl({n}) Killing form computed in {elapsed:.4f}s (dim={algebra.dimension})")

    def test_sp2n_basis_performance(self):
        """测试 sp(2n) 基构造的性能"""
        # 测试不同维度的辛李代数
        ns = [1, 2, 3, 4, 5]
        
        for n in ns:
            start = time.perf_counter()
            algebra = SymplecticLieAlgebra(n)
            elapsed = time.perf_counter() - start
            
            # 验证维数正确：dim = n(2n+1)
            expected_dim = n * (2 * n + 1)
            self.assertEqual(algebra.dimension, expected_dim)
            
            # 验证基的数量正确
            basis = algebra.basis()
            self.assertEqual(len(basis), expected_dim)
            
            print(f"sp({2*n}) basis constructed in {elapsed:.4f}s (dim={algebra.dimension})")

    def test_lie_bracket_performance(self):
        """测试李括号运算的性能"""
        algebra = OrthogonalLieAlgebra(5)  # so(5), dim = 10
        
        # 获取基元素
        basis = algebra.basis()
        x, y = basis[0], basis[1]
        
        # 执行多次李括号运算
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            result = x.bracket(y)
        elapsed = time.perf_counter() - start
        
        avg_time = elapsed / iterations * 1000  # 转换为毫秒
        print(f"Lie bracket average time: {avg_time:.4f}ms ({iterations} iterations)")

    def test_vector_roundtrip_performance(self):
        """测试向量转换的性能"""
        algebra = SpecialLinearLieAlgebra(4)  # sl(4), dim = 15
        
        # 创建随机元素
        vec = [float(i) for i in range(algebra.dimension)]
        element = algebra.from_vector(vec)
        
        # 执行多次 roundtrip
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            recovered_vec = algebra.to_vector(element)
            recovered_element = algebra.from_vector(recovered_vec)
        elapsed = time.perf_counter() - start
        
        avg_time = elapsed / iterations * 1000
        print(f"Vector roundtrip average time: {avg_time:.4f}ms ({iterations} iterations)")


class TestNumericalStability(unittest.TestCase):
    """数值稳定性测试"""

    def test_jacobi_identity_numerical_stability(self):
        """测试雅可比恒等式的数值稳定性"""
        from PySyM.core.lie_theory.specific_lie_algebra import (
            SpecialLinearLieAlgebra,
        )
        from PySyM.core.lie_theory.lie_algebra_operations import StandardLieBracket
        
        algebra = SpecialLinearLieAlgebra(3)
        bracket = StandardLieBracket(algebra)
        
        # 使用随机元素测试
        np.random.seed(42)
        basis = algebra.basis()
        
        # 测试多组随机组合
        for _ in range(10):
            coeffs = np.random.randn(3, algebra.dimension)
            x = algebra.zero()
            y = algebra.zero()
            z = algebra.zero()
            for c, b in zip(coeffs[0], basis):
                x = x + b * c
            for c, b in zip(coeffs[1], basis):
                y = y + b * c
            for c, b in zip(coeffs[2], basis):
                z = z + b * c
            
            # 使用默认容差
            self.assertTrue(bracket.satisfies_jacobi_identity(x, y, z))
            
            # 使用更严格的容差
            self.assertTrue(bracket.satisfies_jacobi_identity(x, y, z, tolerance=1e-12))

    def test_killing_form_symmetry(self):
        """测试基灵型的对称性 K(X,Y) = K(Y,X)"""
        algebra = SpecialLinearLieAlgebra(3)
        killing = KillingForm(algebra)
        
        basis = algebra.basis()
        
        # 测试对称性
        for i in range(min(5, len(basis))):
            for j in range(min(5, len(basis))):
                k_ij = killing(basis[i], basis[j])
                k_ji = killing(basis[j], basis[i])
                self.assertAlmostEqual(k_ij, k_ji, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
