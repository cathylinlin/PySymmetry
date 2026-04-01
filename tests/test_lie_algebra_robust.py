"""李代数模块鲁棒性测试

该文件包含李代数模块的边界值、极端值和鲁棒性测试用例。
涵盖：
- 边界值测试（最小/最大维度）
- 极端值测试（大维度、特殊数值）
- 错误处理测试（无效输入）
- 数值稳定性测试
- 数学性质验证
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from PySyM.core.lie_theory.lie_algebra_factory import LieAlgebraFactory
from PySyM.core.lie_theory.specific_lie_algebra import (
    GeneralLinearLieAlgebra,
    SpecialLinearLieAlgebra,
    OrthogonalLieAlgebra,
    SymplecticLieAlgebra,
    UnitaryLieAlgebra,
    SpecialUnitaryLieAlgebra,
    MatrixLieAlgebraElement
)
from PySyM.core.lie_theory.lie_algebra_operations import StandardLieBracket
from PySyM.core.lie_theory.lie_algebra_structure import KillingForm, RootSystem
from PySyM.core.lie_theory.lie_algebra_representation import (
    AdjointRepresentation, 
    FundamentalRepresentation,
    TensorProductRepresentation
)


class TestLieAlgebraBoundaryValues(unittest.TestCase):
    """李代数边界值测试"""
    
    def test_gl_minimum_dimension(self):
        """测试 gl(1) - 最小维度"""
        gl1 = LieAlgebraFactory.create_general_linear(1)
        self.assertEqual(gl1.dimension, 1)
        self.assertEqual(gl1.n, 1)
        
        # 测试基
        basis = gl1.basis()
        self.assertEqual(len(basis), 1)
        
        # 测试零元素
        zero = gl1.zero()
        self.assertTrue(np.allclose(zero.matrix, np.zeros((1, 1))))
    
    def test_sl_minimum_dimension(self):
        """测试 sl(2) - 最小非平凡维度"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        self.assertEqual(sl2.dimension, 3)
        
        # sl(1) 应该是无效的（迹为零的1x1矩阵只有0）
        with self.assertRaises((ValueError, AssertionError)):
            LieAlgebraFactory.create_special_linear(1)
    
    def test_so_minimum_dimension(self):
        """测试 so(n) 的最小维度"""
        # so(2) 是最小的非平凡正交李代数
        so2 = LieAlgebraFactory.create_orthogonal(2)
        self.assertEqual(so2.dimension, 1)
        
        # so(1) 应该是无效的（n >= 2）
        with self.assertRaises(ValueError):
            LieAlgebraFactory.create_orthogonal(1)
    
    def test_sp_minimum_dimension(self):
        """测试 sp(2n) 的最小维度"""
        # sp(2) 是最小的辛李代数 (n=1)
        sp2 = LieAlgebraFactory.create_symplectic(1)
        self.assertEqual(sp2.dimension, 3)
        self.assertEqual(sp2.n, 1)
        
        # n=0 应该是无效的
        with self.assertRaises((ValueError, AssertionError)):
            LieAlgebraFactory.create_symplectic(0)
    
    def test_su_minimum_dimension(self):
        """测试 su(n) 的最小维度"""
        # su(2) 是最小的非平凡特殊酉李代数
        su2 = LieAlgebraFactory.create_special_unitary(2)
        self.assertEqual(su2.dimension, 3)
        
        # su(1) 应该是无效的
        with self.assertRaises((ValueError, AssertionError)):
            LieAlgebraFactory.create_special_unitary(1)
    
    def test_large_dimension_gl(self):
        """测试大维度 gl(n)"""
        # 测试 gl(10)
        gl10 = LieAlgebraFactory.create_general_linear(10)
        self.assertEqual(gl10.dimension, 100)
        
        # 验证基的数量
        basis = gl10.basis()
        self.assertEqual(len(basis), 100)
    
    def test_large_dimension_sl(self):
        """测试大维度 sl(n)"""
        # 测试 sl(10)
        sl10 = LieAlgebraFactory.create_special_linear(10)
        self.assertEqual(sl10.dimension, 99)  # n^2 - 1
        
        # 验证迹为零
        basis = sl10.basis()
        for b in basis:
            trace = np.trace(b.matrix)
            self.assertAlmostEqual(trace, 0.0, places=10)
    
    def test_large_dimension_so(self):
        """测试大维度 so(n)"""
        # 测试 so(10)
        so10 = LieAlgebraFactory.create_orthogonal(10)
        self.assertEqual(so10.dimension, 45)  # n(n-1)/2
        
        # 验证反对称性
        basis = so10.basis()
        for b in basis:
            m = b.matrix
            self.assertTrue(np.allclose(m, -m.T))


class TestLieAlgebraExtremeValues(unittest.TestCase):
    """李代数极端值测试"""
    
    def test_zero_element_operations(self):
        """测试零元素的各种运算"""
        sl3 = LieAlgebraFactory.create_special_linear(3)
        zero = sl3.zero()
        
        # 获取非零元素
        basis = sl3.basis()
        x = basis[0]
        
        # 零元素加任何元素等于该元素
        self.assertEqual(zero + x, x)
        self.assertEqual(x + zero, x)
        
        # 零元素的李括号
        bracket_zero_x = sl3.bracket(zero, x)
        self.assertEqual(bracket_zero_x, zero)
        
        bracket_x_zero = sl3.bracket(x, zero)
        self.assertEqual(bracket_x_zero, zero)
    
    def test_scalar_multiplication_extremes(self):
        """测试标量乘法的极端值"""
        so3 = LieAlgebraFactory.create_orthogonal(3)
        basis = so3.basis()
        x = basis[0]
        
        # 测试零标量
        zero_scaled = x * 0
        self.assertTrue(np.allclose(zero_scaled.matrix, np.zeros((3, 3))))
        
        # 测试单位标量
        identity_scaled = x * 1
        self.assertTrue(np.allclose(identity_scaled.matrix, x.matrix))
        
        # 测试负标量
        negative_scaled = x * (-1)
        self.assertTrue(np.allclose(negative_scaled.matrix, -x.matrix))
        
        # 测试大标量
        large_scaled = x * 1e6
        self.assertTrue(np.allclose(large_scaled.matrix, x.matrix * 1e6))
        
        # 测试极小标量
        small_scaled = x * 1e-10
        self.assertTrue(np.allclose(small_scaled.matrix, x.matrix * 1e-10))
    
    def test_linear_combinations(self):
        """测试线性组合"""
        su2 = LieAlgebraFactory.create_special_unitary(2)
        basis = su2.basis()
        
        # 测试线性无关性：基不能互相表示
        for i, b1 in enumerate(basis):
            for j, b2 in enumerate(basis):
                if i != j:
                    # 检查是否线性无关（不能相等）
                    self.assertNotEqual(b1, b2)
        
        # 测试线性组合
        x = basis[0] * 2 + basis[1] * 3 + basis[2] * (-1)
        vec = su2.to_vector(x)
        self.assertEqual(len(vec), 3)
        self.assertAlmostEqual(vec[0], 2.0)
        self.assertAlmostEqual(vec[1], 3.0)
        self.assertAlmostEqual(vec[2], -1.0)
    
    def test_vector_roundtrip(self):
        """测试向量转换的往返一致性"""
        sl4 = LieAlgebraFactory.create_special_linear(4)
        
        # 测试各种向量
        test_vectors = [
            [0.0] * sl4.dimension,
            [1.0] * sl4.dimension,
            list(range(sl4.dimension)),
            [float(i) / sl4.dimension for i in range(sl4.dimension)],
            np.random.randn(sl4.dimension).tolist(),
        ]
        
        for vec in test_vectors:
            element = sl4.from_vector(vec)
            recovered_vec = sl4.to_vector(element)
            for v1, v2 in zip(vec, recovered_vec):
                self.assertAlmostEqual(v1, v2, places=10)


class TestLieAlgebraErrorHandling(unittest.TestCase):
    """李代数错误处理测试"""
    
    def test_invalid_dimension_zero(self):
        """测试维度为0的情况"""
        # gl(0) 应该是无效的
        with self.assertRaises((ValueError, AssertionError)):
            LieAlgebraFactory.create_general_linear(0)
    
    def test_invalid_dimension_negative(self):
        """测试负维度"""
        with self.assertRaises((ValueError, AssertionError)):
            LieAlgebraFactory.create_general_linear(-1)
        
        with self.assertRaises((ValueError, AssertionError)):
            LieAlgebraFactory.create_special_linear(-5)
    
    def test_mismatched_lie_algebra_bracket(self):
        """测试不同李代数元素的李括号"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        so3 = LieAlgebraFactory.create_orthogonal(3)
        
        x = sl2.basis()[0]
        y = so3.basis()[0]
        
        # 应该抛出类型错误
        with self.assertRaises(TypeError):
            x.bracket(y)
    
    def test_mismatched_addition(self):
        """测试不同李代数元素的加法"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        so3 = LieAlgebraFactory.create_orthogonal(3)
        
        x = sl2.basis()[0]
        y = so3.basis()[0]
        
        # 应该抛出 TypeError 或返回 NotImplemented
        with self.assertRaises(TypeError):
            x + y
    
    def test_invalid_vector_length(self):
        """测试无效向量长度"""
        sl3 = LieAlgebraFactory.create_special_linear(3)
        
        # 向量长度不匹配
        with self.assertRaises((ValueError, AssertionError)):
            sl3.from_vector([1, 2])  # 太短
        
        with self.assertRaises((ValueError, AssertionError)):
            sl3.from_vector([1] * 100)  # 太长
    
    def test_invalid_factory_name(self):
        """测试工厂无效名称"""
        with self.assertRaises(ValueError):
            LieAlgebraFactory.create_lie_algebra('invalid', 2)
        
        with self.assertRaises(ValueError):
            LieAlgebraFactory.get_lie_algebra_class('unknown')


class TestLieAlgebraMathematicalProperties(unittest.TestCase):
    """李代数数学性质测试"""
    
    def test_jacobi_identity_comprehensive(self):
        """全面测试雅可比恒等式"""
        algebras = [
            ('sl', 3),
            ('so', 4),
            ('su', 3),
        ]
        
        for name, n in algebras:
            algebra = LieAlgebraFactory.create_lie_algebra(name, n)
            bracket = StandardLieBracket(algebra)
            basis = algebra.basis()
            
            # 测试所有基元素的三元组
            for i, x in enumerate(basis):
                for j, y in enumerate(basis):
                    for k, z in enumerate(basis):
                        self.assertTrue(
                            bracket.satisfies_jacobi_identity(x, y, z),
                            f"雅可比恒等式在 {name}({n}) 的基元素 ({i},{j},{k}) 上失败"
                        )
    
    def test_anticommutativity(self):
        """测试反对称性 [x,y] = -[y,x]"""
        sl3 = LieAlgebraFactory.create_special_linear(3)
        basis = sl3.basis()
        
        for i, x in enumerate(basis):
            for j, y in enumerate(basis):
                bracket_xy = sl3.bracket(x, y)
                bracket_yx = sl3.bracket(y, x)
                
                # [x,y] = -[y,x]
                self.assertEqual(bracket_xy, sl3.scalar_multiply(bracket_yx, -1))
    
    def test_bilinearity(self):
        """测试双线性"""
        sl3 = LieAlgebraFactory.create_special_linear(3)
        basis = sl3.basis()
        x, y, z = basis[0], basis[1], basis[2]
        
        # 测试第一参数的线性
        # [ax + by, z] = a[x,z] + b[y,z]
        a, b = 2.0, 3.0
        left = sl3.bracket(sl3.scalar_multiply(x, a) + sl3.scalar_multiply(y, b), z)
        right = sl3.scalar_multiply(sl3.bracket(x, z), a) + sl3.scalar_multiply(sl3.bracket(y, z), b)
        self.assertEqual(left, right)
        
        # 测试第二参数的线性
        # [z, ax + by] = a[z,x] + b[z,y]
        left2 = sl3.bracket(z, sl3.scalar_multiply(x, a) + sl3.scalar_multiply(y, b))
        right2 = sl3.scalar_multiply(sl3.bracket(z, x), a) + sl3.scalar_multiply(sl3.bracket(z, y), b)
        self.assertEqual(left2, right2)
    
    def test_killing_form_properties(self):
        """测试基灵型性质"""
        sl3 = LieAlgebraFactory.create_special_linear(3)
        killing = KillingForm(sl3)
        
        # 对称性 K(X,Y) = K(Y,X)
        basis = sl3.basis()
        for i, x in enumerate(basis):
            for j, y in enumerate(basis):
                k_xy = killing(x, y)
                k_yx = killing(y, x)
                self.assertAlmostEqual(k_xy, k_yx, places=10)
        
        # 对于半单李代数，基灵型应该非退化
        self.assertTrue(killing.is_non_degenerate())
        
        # 基灵型矩阵应该对称
        matrix = killing.matrix
        self.assertTrue(np.allclose(matrix, matrix.T))
    
    def test_adjoint_representation_homomorphism(self):
        """测试伴随表示是李代数同态"""
        sl3 = LieAlgebraFactory.create_special_linear(3)
        adjoint = AdjointRepresentation(sl3)
        
        basis = sl3.basis()
        x, y = basis[0], basis[1]
        
        # ad([x,y]) = [ad(x), ad(y)]
        bracket_xy = sl3.bracket(x, y)
        ad_bracket = adjoint(bracket_xy)
        
        ad_x = adjoint(x)
        ad_y = adjoint(y)
        commutator = ad_x @ ad_y - ad_y @ ad_x
        
        self.assertTrue(np.allclose(ad_bracket, commutator))


class TestLieAlgebraNumericalStability(unittest.TestCase):
    """李代数数值稳定性测试"""
    
    def test_near_zero_operations(self):
        """测试接近零的运算"""
        so3 = LieAlgebraFactory.create_orthogonal(3)
        basis = so3.basis()
        x = basis[0]
        
        # 极小标量乘法
        epsilon = 1e-15
        tiny = x * epsilon
        
        # 应该仍然保持反对称性
        self.assertTrue(np.allclose(tiny.matrix, -tiny.matrix.T))
    
    def test_large_number_operations(self):
        """测试大数运算"""
        so3 = LieAlgebraFactory.create_orthogonal(3)
        basis = so3.basis()
        x = basis[0]
        
        # 大数乘法
        big = x * 1e10
        
        # 应该仍然保持反对称性
        self.assertTrue(np.allclose(big.matrix, -big.matrix.T, rtol=1e-5))
    
    def test_floating_point_accumulation(self):
        """测试浮点数累积误差"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        basis = sl2.basis()
        x = basis[0]
        
        # 多次加法累积
        result = sl2.zero()
        for _ in range(1000):
            result = result + x * 0.001
        
        # 应该接近 x
        expected = x
        diff = result - expected
        # 允许一定的累积误差
        vec_diff = sl2.to_vector(diff)
        self.assertTrue(all(abs(v) < 1e-10 for v in vec_diff))


class TestLieAlgebraRepresentationRobustness(unittest.TestCase):
    """李代数表示鲁棒性测试"""
    
    def test_adjoint_representation_dimension(self):
        """测试伴随表示维度"""
        for n in [2, 3, 4, 5]:
            sln = LieAlgebraFactory.create_special_linear(n)
            adjoint = AdjointRepresentation(sln)
            
            # 维度应该等于李代数的维数
            self.assertEqual(adjoint.dimension, sln.dimension)
            
            # 表示矩阵应该是方阵
            x = sln.basis()[0]
            matrix = adjoint(x)
            self.assertEqual(matrix.shape, (sln.dimension, sln.dimension))
    
    def test_fundamental_representation_consistency(self):
        """测试基本表示的一致性"""
        su3 = LieAlgebraFactory.create_special_unitary(3)
        fundamental = FundamentalRepresentation(su3, 3)
        
        # 零元素应该映射到零矩阵
        zero = su3.zero()
        zero_repr = fundamental(zero)
        self.assertTrue(np.allclose(zero_repr, np.zeros((3, 3))))
        
        # 特征标应该满足线性性
        basis = su3.basis()
        x, y = basis[0], basis[1]
        
        char_x = fundamental.character(x)
        char_y = fundamental.character(y)
        char_sum = fundamental.character(x + y)
        
        # tr(X + Y) = tr(X) + tr(Y)
        self.assertAlmostEqual(char_sum, char_x + char_y, places=10)
    
    def test_tensor_product_representation(self):
        """测试张量积表示"""
        su2 = LieAlgebraFactory.create_special_unitary(2)
        
        fund1 = FundamentalRepresentation(su2, 2)
        fund2 = FundamentalRepresentation(su2, 2)
        
        tensor = TensorProductRepresentation(fund1, fund2)
        
        # 维度应该是乘积
        self.assertEqual(tensor.dimension, 4)
        
        # 测试同态性质
        basis = su2.basis()
        x, y = basis[0], basis[1]
        
        # ρ([x,y]) = [ρ(x), ρ(y)]
        bracket_xy = su2.bracket(x, y)
        repr_bracket = tensor(bracket_xy)
        
        repr_x = tensor(x)
        repr_y = tensor(y)
        commutator = repr_x @ repr_y - repr_y @ repr_x
        
        self.assertTrue(np.allclose(repr_bracket, commutator))


class TestLieAlgebraStructureRobustness(unittest.TestCase):
    """李代数结构鲁棒性测试"""
    
    def test_root_system_basic(self):
        """测试根系基本性质"""
        sl3 = LieAlgebraFactory.create_special_linear(3)
        
        # 创建根系
        roots = [
            np.array([1, -1, 0]),
            np.array([1, 0, -1]),
            np.array([0, 1, -1]),
            np.array([-1, 1, 0]),
            np.array([-1, 0, 1]),
            np.array([0, -1, 1]),
        ]
        root_system = RootSystem(roots)
        
        # 秩应该等于根向量的维度
        self.assertEqual(root_system.rank, 3)
        
        # 根的数量
        self.assertEqual(len(root_system.roots), 6)
    
    def test_different_lie_algebra_types(self):
        """测试不同类型的李代数"""
        test_cases = [
            ('gl', 3, 9),      # gl(3) 维数 = 9
            ('sl', 3, 8),      # sl(3) 维数 = 8
            ('so', 4, 6),      # so(4) 维数 = 6
            ('sp', 2, 10),     # sp(4) 维数 = 10 (n=2, 矩阵4x4)
            ('u', 2, 4),       # u(2) 维数 = 4
            ('su', 3, 8),      # su(3) 维数 = 8
        ]
        
        for name, n, expected_dim in test_cases:
            algebra = LieAlgebraFactory.create_lie_algebra(name, n)
            self.assertEqual(
                algebra.dimension, 
                expected_dim,
                f"{name}({n}) 的维数应该是 {expected_dim}，但得到 {algebra.dimension}"
            )


if __name__ == '__main__':
    unittest.main()
