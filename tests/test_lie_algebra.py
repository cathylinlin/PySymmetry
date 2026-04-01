"""李代数模块测试

该文件包含李代数模块的测试用例。
"""
import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from PySyM.core.lie_theory.lie_algebra_factory import LieAlgebraFactory
from PySyM.core.lie_theory.lie_algebra_representation import AdjointRepresentation, FundamentalRepresentation
from PySyM.core.lie_theory.lie_algebra_structure import KillingForm, RootSystem, WeylGroup
from PySyM.core.lie_theory.lie_algebra_operations import StandardLieBracket


class TestLieAlgebra(unittest.TestCase):
    """李代数测试类"""
    
    def test_general_linear_lie_algebra(self):
        """测试一般线性李代数 gl(2)"""
        gl2 = LieAlgebraFactory.create_general_linear(2)
        self.assertEqual(gl2.dimension, 4)
        self.assertEqual(gl2.n, 2)
        
        # 测试零元素
        zero = gl2.zero()
        self.assertIsNotNone(zero)
        
        # 测试基
        basis = gl2.basis()
        self.assertEqual(len(basis), 4)
        
        # 测试李括号
        x = gl2.from_vector([1, 0, 0, 0])  # 单位矩阵的 (0,0) 元素
        y = gl2.from_vector([0, 1, 0, 0])  # 单位矩阵的 (0,1) 元素
        bracket = gl2.bracket(x, y)
        self.assertIsNotNone(bracket)
        
        # 测试向量转换
        vec = gl2.to_vector(x)
        self.assertEqual(len(vec), 4)
    
    def test_special_linear_lie_algebra(self):
        """测试特殊线性李代数 sl(2)"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        self.assertEqual(sl2.dimension, 3)
        self.assertEqual(sl2.n, 2)
        
        # 测试零元素
        zero = sl2.zero()
        self.assertIsNotNone(zero)
        
        # 测试基
        basis = sl2.basis()
        self.assertEqual(len(basis), 3)
        
        # 测试李括号
        x = sl2.from_vector([1, 0, 0])  # 非对角元素
        y = sl2.from_vector([0, 1, 0])  # 非对角元素
        bracket = sl2.bracket(x, y)
        self.assertIsNotNone(bracket)
        
        # 测试迹为零
        for b in basis:
            trace = np.trace(b.matrix)
            self.assertAlmostEqual(trace, 0.0, places=10)
    
    def test_orthogonal_lie_algebra(self):
        """测试正交李代数 so(3)"""
        so3 = LieAlgebraFactory.create_orthogonal(3)
        self.assertEqual(so3.dimension, 3)
        self.assertEqual(so3.n, 3)
        
        # 测试零元素
        zero = so3.zero()
        self.assertIsNotNone(zero)
        
        # 测试基
        basis = so3.basis()
        self.assertEqual(len(basis), 3)
        
        # 测试反对称性
        for b in basis:
            m = b.matrix
            self.assertTrue(np.allclose(m, -m.T))
    
    def test_symplectic_lie_algebra(self):
        """测试辛李代数 sp(2)（n=1 时矩阵为 2×2）"""
        sp1 = LieAlgebraFactory.create_symplectic(1)
        self.assertEqual(sp1.dimension, 3)
        self.assertEqual(sp1.n, 1)
        
        # 测试零元素
        zero = sp1.zero()
        self.assertIsNotNone(zero)
        
        # 测试基
        basis = sp1.basis()
        self.assertEqual(len(basis), 3)
        
        # 测试辛条件 X^T J + J X = 0
        from PySyM.core.lie_theory.specific_lie_algebra import _symplectic_J
        J = _symplectic_J(1)
        for b in basis:
            m = b.matrix
            condition = m.T @ J + J @ m
            self.assertTrue(np.allclose(condition, 0.0))
    
    def test_unitary_lie_algebra(self):
        """测试酉李代数 u(2)"""
        u2 = LieAlgebraFactory.create_unitary(2)
        self.assertEqual(u2.dimension, 4)
        self.assertEqual(u2.n, 2)
        
        # 测试零元素
        zero = u2.zero()
        self.assertIsNotNone(zero)
        
        # 测试基
        basis = u2.basis()
        self.assertEqual(len(basis), 4)
        
        # 测试反厄米性 X^† = -X
        for b in basis:
            m = b.matrix
            self.assertTrue(np.allclose(m, -m.conj().T))
    
    def test_special_unitary_lie_algebra(self):
        """测试特殊酉李代数 su(2)"""
        su2 = LieAlgebraFactory.create_special_unitary(2)
        self.assertEqual(su2.dimension, 3)
        self.assertEqual(su2.n, 2)
        
        # 测试零元素
        zero = su2.zero()
        self.assertIsNotNone(zero)
        
        # 测试基
        basis = su2.basis()
        self.assertEqual(len(basis), 3)
        
        # 测试反厄米性和迹为零
        for b in basis:
            m = b.matrix
            self.assertTrue(np.allclose(m, -m.conj().T))
            self.assertAlmostEqual(np.trace(m), 0.0, places=10)

        props = su2.properties()
        self.assertFalse(props.is_abelian)
        self.assertTrue(props.is_semisimple)
    
    def test_adjoint_representation(self):
        """测试伴随表示"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        adjoint = AdjointRepresentation(sl2)
        self.assertEqual(adjoint.dimension, 3)
        
        # 测试表示
        x = sl2.from_vector([1, 0, 0])
        matrix = adjoint(x)
        self.assertIsNotNone(matrix)
        self.assertEqual(matrix.shape, (3, 3))
        
        # 测试特征标
        char = adjoint.character(x)
        self.assertIsInstance(char, complex)
    
    def test_fundamental_representation(self):
        """测试基本表示"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        fundamental = FundamentalRepresentation(sl2, 2)
        self.assertEqual(fundamental.dimension, 2)
        
        # 测试表示
        x = sl2.from_vector([1, 0, 0])
        matrix = fundamental(x)
        self.assertIsNotNone(matrix)
        self.assertEqual(matrix.shape, (2, 2))
    
    def test_killing_form(self):
        """测试基灵型"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        killing_form = KillingForm(sl2)
        
        # 测试基灵型矩阵
        self.assertIsNotNone(killing_form.matrix)
        self.assertEqual(killing_form.matrix.shape, (3, 3))
        
        # 测试非退化性
        self.assertTrue(killing_form.is_non_degenerate())
        
        # 测试基灵型的对称性
        matrix = killing_form.matrix
        self.assertTrue(np.allclose(matrix, matrix.T))
        
        # 测试基灵型计算
        x = sl2.from_vector([1, 0, 0])
        y = sl2.from_vector([0, 1, 0])
        k = killing_form(x, y)
        self.assertIsInstance(k, float)
    
    def test_lie_bracket(self):
        """测试李括号"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        lie_bracket = StandardLieBracket(sl2)
        
        # 测试反对称性
        self.assertTrue(lie_bracket.is_anticommutative())
        
        # 测试雅可比恒等式
        x = sl2.from_vector([1, 0, 0])
        y = sl2.from_vector([0, 1, 0])
        z = sl2.from_vector([0, 0, 1])
        self.assertTrue(lie_bracket.satisfies_jacobi_identity(x, y, z))
        
        # 测试李括号的反对称性 [x, y] = -[y, x]
        bracket_xy = lie_bracket(x, y)
        bracket_yx = lie_bracket(y, x)
        self.assertEqual(bracket_xy, sl2.scalar_multiply(bracket_yx, -1))
    
    def test_lie_algebra_factory(self):
        """测试李代数工厂"""
        # 测试创建一般线性李代数
        gl2 = LieAlgebraFactory.create_general_linear(2)
        self.assertIsNotNone(gl2)
        
        # 测试创建特殊线性李代数
        sl2 = LieAlgebraFactory.create_special_linear(2)
        self.assertIsNotNone(sl2)
        
        # 测试根据名称创建李代数
        so3 = LieAlgebraFactory.create_lie_algebra('so', 3)
        self.assertIsNotNone(so3)
        
        # 测试获取李代数类
        gl_class = LieAlgebraFactory.get_lie_algebra_class('gl')
        self.assertIsNotNone(gl_class)
        
        # 测试错误处理
        with self.assertRaises(ValueError):
            LieAlgebraFactory.create_lie_algebra('unknown', 2)


class TestLieAlgebraProperties(unittest.TestCase):
    """测试李代数属性"""
    
    def test_sl2_properties(self):
        """测试 sl(2) 的属性"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        props = sl2.properties()
        
        self.assertEqual(props.name, "sl(2)")
        self.assertEqual(props.dimension, 3)
        self.assertTrue(props.is_semisimple)
        self.assertTrue(props.is_simple)
        self.assertFalse(props.is_abelian)
        self.assertEqual(props.root_system_type, "A1")
        self.assertEqual(props.rank, 1)
    
    def test_so3_properties(self):
        """测试 so(3) 的属性"""
        so3 = LieAlgebraFactory.create_orthogonal(3)
        props = so3.properties()
        
        self.assertEqual(props.name, "so(3)")
        self.assertEqual(props.dimension, 3)
        self.assertTrue(props.is_semisimple)
        self.assertTrue(props.is_simple)
        self.assertFalse(props.is_abelian)
        self.assertEqual(props.root_system_type, "B1")
        self.assertEqual(props.rank, 1)
    
    def test_su2_properties(self):
        """测试 su(2) 的属性"""
        su2 = LieAlgebraFactory.create_special_unitary(2)
        props = su2.properties()
        
        self.assertEqual(props.name, "su(2)")
        self.assertEqual(props.dimension, 3)
        self.assertTrue(props.is_semisimple)
        self.assertTrue(props.is_simple)
        self.assertFalse(props.is_abelian)
        self.assertEqual(props.root_system_type, "A1")
        self.assertEqual(props.rank, 1)
    
    def test_gl1_properties(self):
        """测试 gl(1) 的属性（阿贝尔）"""
        gl1 = LieAlgebraFactory.create_general_linear(1)
        props = gl1.properties()
        
        self.assertEqual(props.name, "gl(1)")
        self.assertEqual(props.dimension, 1)
        self.assertFalse(props.is_semisimple)
        self.assertFalse(props.is_simple)
        self.assertTrue(props.is_abelian)


class TestRootSystem(unittest.TestCase):
    """测试根系"""
    
    def test_root_system_creation(self):
        """测试根系创建"""
        roots = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
        root_system = RootSystem(roots)
        
        self.assertEqual(len(root_system.roots), 4)
        self.assertEqual(root_system.rank, 2)
    
    def test_positive_roots(self):
        """测试正根计算"""
        roots = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
        root_system = RootSystem(roots)
        
        positive = root_system.positive_roots()
        self.assertEqual(len(positive), 2)
    
    def test_simple_roots(self):
        """测试单根计算"""
        roots = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
        root_system = RootSystem(roots)
        
        simple = root_system.simple_roots()
        self.assertGreaterEqual(len(simple), 1)
    
    def test_root_system_type(self):
        """测试根系类型识别"""
        # A1 类型
        roots_a1 = [np.array([1]), np.array([-1])]
        rs_a1 = RootSystem(roots_a1)
        self.assertEqual(rs_a1.root_system_type(), "A1")
        
        # A2 类型
        roots_a2 = [
            np.array([1, 0]), np.array([0, 1]), np.array([1, 1]),
            np.array([-1, 0]), np.array([0, -1]), np.array([-1, -1])
        ]
        rs_a2 = RootSystem(roots_a2)
        self.assertEqual(rs_a2.root_system_type(), "A2")


class TestWeylGroup(unittest.TestCase):
    """测试外尔群"""
    
    def test_weyl_group_creation(self):
        """测试外尔群创建"""
        roots = [np.array([1]), np.array([-1])]
        root_system = RootSystem(roots)
        weyl = WeylGroup(root_system)
        
        self.assertIsNotNone(weyl.generators)
        self.assertEqual(weyl.order(), 2)  # A1 的外尔群阶为 2
    
    def test_weyl_group_order(self):
        """测试外尔群的阶"""
        # A2 类型
        roots_a2 = [
            np.array([1, 0]), np.array([0, 1]), np.array([1, 1]),
            np.array([-1, 0]), np.array([0, -1]), np.array([-1, -1])
        ]
        rs_a2 = RootSystem(roots_a2)
        weyl_a2 = WeylGroup(rs_a2)
        self.assertEqual(weyl_a2.order(), 6)
    
    def test_weyl_group_action(self):
        """测试外尔群作用"""
        roots = [np.array([1]), np.array([-1])]
        root_system = RootSystem(roots)
        weyl = WeylGroup(root_system)
        
        vector = np.array([2.0])
        for generator in weyl.generators:
            result = weyl.act(generator, vector)
            self.assertIsNotNone(result)


class TestLieAlgebraOperations(unittest.TestCase):
    """测试李代数运算"""
    
    def test_addition(self):
        """测试加法运算"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        x = sl2.from_vector([1, 0, 0])
        y = sl2.from_vector([0, 1, 0])
        
        z = sl2.add(x, y)
        vec = sl2.to_vector(z)
        self.assertEqual(vec, [1, 1, 0])
    
    def test_scalar_multiplication(self):
        """测试标量乘法"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        x = sl2.from_vector([1, 0, 0])
        
        y = sl2.scalar_multiply(x, 2.0)
        vec = sl2.to_vector(y)
        self.assertEqual(vec, [2, 0, 0])
    
    def test_vector_roundtrip(self):
        """测试向量转换的往返一致性"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        original_vec = [1.0, 2.0, 3.0]
        
        element = sl2.from_vector(original_vec)
        recovered_vec = sl2.to_vector(element)
        
        for a, b in zip(original_vec, recovered_vec):
            self.assertAlmostEqual(a, b, places=10)


class TestLieAlgebraEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_sl_n_minimum(self):
        """测试 sl(n) 的最小 n 值"""
        with self.assertRaises(ValueError):
            LieAlgebraFactory.create_special_linear(1)
    
    def test_so_n_minimum(self):
        """测试 so(n) 的最小 n 值"""
        with self.assertRaises(ValueError):
            LieAlgebraFactory.create_orthogonal(1)
    
    def test_su_n_minimum(self):
        """测试 su(n) 的最小 n 值"""
        with self.assertRaises(ValueError):
            LieAlgebraFactory.create_special_unitary(1)
    
    def test_gl_n_minimum(self):
        """测试 gl(n) 的最小 n 值"""
        with self.assertRaises(ValueError):
            LieAlgebraFactory.create_general_linear(0)
    
    def test_sp_n_minimum(self):
        """测试 sp(n) 的最小 n 值"""
        with self.assertRaises(ValueError):
            LieAlgebraFactory.create_symplectic(0)
    
    def test_lie_bracket_same_algebra(self):
        """测试不同李代数元素的李括号"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        so3 = LieAlgebraFactory.create_orthogonal(3)
        
        x = sl2.from_vector([1, 0, 0])
        y = so3.from_vector([1, 0, 0])
        
        # 应该抛出类型错误
        with self.assertRaises(TypeError):
            x.bracket(y)


class TestHigherDimensionalLieAlgebras(unittest.TestCase):
    """测试高维李代数"""
    
    def test_sl3(self):
        """测试 sl(3)"""
        sl3 = LieAlgebraFactory.create_special_linear(3)
        self.assertEqual(sl3.dimension, 8)
        
        basis = sl3.basis()
        self.assertEqual(len(basis), 8)
        
        props = sl3.properties()
        self.assertEqual(props.root_system_type, "A2")
        self.assertEqual(props.rank, 2)
    
    def test_so4(self):
        """测试 so(4)"""
        so4 = LieAlgebraFactory.create_orthogonal(4)
        self.assertEqual(so4.dimension, 6)
        
        basis = so4.basis()
        self.assertEqual(len(basis), 6)
    
    def test_so5(self):
        """测试 so(5)"""
        so5 = LieAlgebraFactory.create_orthogonal(5)
        self.assertEqual(so5.dimension, 10)
        
        props = so5.properties()
        self.assertEqual(props.root_system_type, "B2")
        self.assertEqual(props.rank, 2)
    
    def test_su3(self):
        """测试 su(3)"""
        su3 = LieAlgebraFactory.create_special_unitary(3)
        self.assertEqual(su3.dimension, 8)
        
        props = su3.properties()
        self.assertEqual(props.root_system_type, "A2")
        self.assertEqual(props.rank, 2)
    
    def test_sp4(self):
        """测试 sp(4)（n=2 时的辛李代数）"""
        sp2 = LieAlgebraFactory.create_symplectic(2)
        self.assertEqual(sp2.dimension, 10)
        
        props = sp2.properties()
        self.assertEqual(props.root_system_type, "C2")
        self.assertEqual(props.rank, 2)


if __name__ == '__main__':
    unittest.main()
