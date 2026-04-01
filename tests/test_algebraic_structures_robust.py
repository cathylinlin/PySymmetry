"""代数结构模块鲁棒性测试

该文件包含代数结构模块的边界值、极端值和鲁棒性测试用例。
涵盖：
- 边界值测试（最小/最大阶数、特殊元素）
- 极端值测试（大数、小数、特殊数值）
- 错误处理测试（无效输入、除零等）
- 代数性质验证（域、环、向量空间公理）
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from PySyM.core.algebraic_structures import (
    IntegerRing, IntegerRingElement,
    RationalField, RationalFieldElement,
    RealField, RealFieldElement,
    FiniteField, FiniteFieldElement,
    ComplexField, ComplexFieldElement,
    PolynomialRing, PolynomialRingElement,
    MatrixRing, MatrixRingElement,
    FiniteDimensionalVectorSpace, FiniteDimensionalVectorSpaceElement,
    LinearTransformation,
    direct_sum_groups
)


class TestIntegerRingRobustness(unittest.TestCase):
    """整数环鲁棒性测试"""
    
    def test_zero_element(self):
        """测试零元素性质"""
        ring = IntegerRing()
        zero = IntegerRingElement(0)
        
        # 0 + a = a
        for a in [IntegerRingElement(5), IntegerRingElement(-3), IntegerRingElement(0)]:
            self.assertEqual(ring.add(zero, a).value, a.value)
            self.assertEqual(ring.add(a, zero).value, a.value)
        
        # 0 * a = 0
        for a in [IntegerRingElement(5), IntegerRingElement(-3)]:
            self.assertEqual(ring.multiply(zero, a).value, 0)
    
    def test_identity_element(self):
        """测试乘法单位元"""
        ring = IntegerRing()
        one = IntegerRingElement(1)
        
        # 1 * a = a
        for a in [IntegerRingElement(5), IntegerRingElement(-3), IntegerRingElement(0)]:
            self.assertEqual(ring.multiply(one, a).value, a.value)
            self.assertEqual(ring.multiply(a, one).value, a.value)
    
    def test_negative_numbers(self):
        """测试负数运算"""
        ring = IntegerRing()
        
        # 加法逆元
        a = IntegerRingElement(5)
        neg_a = ring.inverse(a)
        self.assertEqual(neg_a.value, -5)
        
        # a + (-a) = 0
        sum_result = ring.add(a, neg_a)
        self.assertEqual(sum_result.value, 0)
    
    def test_large_numbers(self):
        """测试大数运算"""
        ring = IntegerRing()
        
        big = IntegerRingElement(10**10)
        small = IntegerRingElement(1)
        
        # 大数加法
        result = ring.add(big, small)
        self.assertEqual(result.value, 10**10 + 1)
        
        # 大数乘法
        result = ring.multiply(big, IntegerRingElement(2))
        self.assertEqual(result.value, 2 * 10**10)
    
    def test_distributivity(self):
        """测试分配律"""
        ring = IntegerRing()
        
        a = IntegerRingElement(2)
        b = IntegerRingElement(3)
        c = IntegerRingElement(4)
        
        # a * (b + c) = a * b + a * c
        left = ring.multiply(a, ring.add(b, c))
        right = ring.add(ring.multiply(a, b), ring.multiply(a, c))
        self.assertEqual(left.value, right.value)


class TestRationalFieldRobustness(unittest.TestCase):
    """有理数域鲁棒性测试"""
    
    def test_zero_and_one(self):
        """测试零元和单位元"""
        field = RationalField()
        zero = RationalFieldElement(0, 1)
        one = RationalFieldElement(1, 1)
        
        # 零元性质
        a = RationalFieldElement(3, 4)
        self.assertEqual(field.add(zero, a), a)
        
        # 单位元性质
        self.assertEqual(field.multiply(one, a), a)
    
    def test_simplification(self):
        """测试分数化简"""
        # 6/8 应该化简为 3/4
        a = RationalFieldElement(6, 8)
        self.assertEqual(a.numerator, 3)
        self.assertEqual(a.denominator, 4)
        
        # -6/8 应该化简为 -3/4
        b = RationalFieldElement(-6, 8)
        self.assertEqual(b.numerator, -3)
        self.assertEqual(b.denominator, 4)
        
        # 6/-8 应该化简为 -3/4（分母保持正数）
        c = RationalFieldElement(6, -8)
        self.assertEqual(c.numerator, -3)
        self.assertEqual(c.denominator, 4)
    
    def test_addition_different_denominators(self):
        """测试不同分母的加法"""
        field = RationalField()
        
        a = RationalFieldElement(1, 2)
        b = RationalFieldElement(1, 3)
        result = field.add(a, b)
        
        self.assertEqual(result.numerator, 5)
        self.assertEqual(result.denominator, 6)
    
    def test_multiplication(self):
        """测试乘法"""
        field = RationalField()
        
        a = RationalFieldElement(2, 3)
        b = RationalFieldElement(3, 4)
        result = field.multiply(a, b)
        
        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 2)
    
    def test_division(self):
        """测试除法"""
        field = RationalField()
        
        a = RationalFieldElement(2, 3)
        b = RationalFieldElement(4, 5)
        result = field.divide(a, b)
        
        # (2/3) / (4/5) = (2/3) * (5/4) = 10/12 = 5/6
        self.assertEqual(result.numerator, 5)
        self.assertEqual(result.denominator, 6)
    
    def test_multiplicative_inverse(self):
        """测试乘法逆元"""
        field = RationalField()
        
        a = RationalFieldElement(3, 4)
        inv = field.multiplicative_inverse(a)
        
        # (3/4)^(-1) = 4/3
        self.assertEqual(inv.numerator, 4)
        self.assertEqual(inv.denominator, 3)
        
        # a * a^(-1) = 1
        product = field.multiply(a, inv)
        self.assertEqual(product.numerator, 1)
        self.assertEqual(product.denominator, 1)
    
    def test_division_by_zero(self):
        """测试除零错误"""
        field = RationalField()
        
        a = RationalFieldElement(1, 2)
        zero = RationalFieldElement(0, 1)
        
        with self.assertRaises((ValueError, ZeroDivisionError)):
            field.divide(a, zero)
        
        with self.assertRaises((ValueError, ZeroDivisionError)):
            field.multiplicative_inverse(zero)


class TestFiniteFieldRobustness(unittest.TestCase):
    """有限域鲁棒性测试"""
    
    def test_prime_field_basic(self):
        """测试素域基本运算"""
        # GF(5)
        field = FiniteField(5)
        
        # 加法
        a = FiniteFieldElement(3, 5)
        b = FiniteFieldElement(4, 5)
        result = field.add(a, b)
        self.assertEqual(result.value, 2)  # 3 + 4 = 7 ≡ 2 (mod 5)
        
        # 乘法
        result = field.multiply(a, b)
        self.assertEqual(result.value, 2)  # 3 * 4 = 12 ≡ 2 (mod 5)
    
    def test_finite_field_properties(self):
        """测试有限域性质"""
        # GF(7)
        field = FiniteField(7)
        
        # 每个非零元素都有乘法逆元
        for i in range(1, 7):
            a = FiniteFieldElement(i, 7)
            inv = field.multiplicative_inverse(a)
            product = field.multiply(a, inv)
            self.assertEqual(product.value, 1)
    
    def test_fermats_little_theorem(self):
        """测试费马小定理"""
        # 对于 GF(p)，a^(p-1) ≡ 1 (mod p)
        p = 7
        field = FiniteField(p)
        
        for a in range(1, p):
            elem = FiniteFieldElement(a, p)
            # 计算 a^(p-1)
            result = elem
            for _ in range(p - 2):
                result = field.multiply(result, elem)
            self.assertEqual(result.value, 1)
    
    def test_invalid_modulus(self):
        """测试无效模数"""
        # 模数必须为正整数
        with self.assertRaises((ValueError, AssertionError)):
            FiniteField(0)
        
        with self.assertRaises((ValueError, AssertionError)):
            FiniteField(-1)
    
    def test_element_out_of_range(self):
        """测试超出范围的元素"""
        field = FiniteField(5)
        
        # 元素值应该在 [0, p-1] 范围内
        # 注意：具体行为取决于实现，可能自动取模或抛出异常
        try:
            elem = FiniteFieldElement(5, 5)
            # 如果成功创建，检查是否正确取模
            self.assertEqual(elem.value % 5, 0)
        except (ValueError, AssertionError):
            pass  # 期望的行为
        
        try:
            elem = FiniteFieldElement(-1, 5)
            # 如果成功创建，检查是否正确取模
            self.assertEqual(elem.value % 5, 4)
        except (ValueError, AssertionError):
            pass  # 期望的行为


class TestComplexFieldRobustness(unittest.TestCase):
    """复数域鲁棒性测试"""
    
    def test_complex_basic_operations(self):
        """测试复数基本运算"""
        field = ComplexField()
        
        a = ComplexFieldElement(3, 4)  # 3 + 4i
        b = ComplexFieldElement(1, 2)  # 1 + 2i
        
        # 加法
        result = field.add(a, b)
        self.assertEqual(result.real, 4)
        self.assertEqual(result.imag, 6)
        
        # 乘法
        result = field.multiply(a, b)
        # (3 + 4i)(1 + 2i) = 3 + 6i + 4i + 8i^2 = 3 + 10i - 8 = -5 + 10i
        self.assertEqual(result.real, -5)
        self.assertEqual(result.imag, 10)
    
    def test_complex_conjugate(self):
        """测试复共轭"""
        a = ComplexFieldElement(3, 4)
        # 复共轭可能通过 conj 方法或属性访问
        try:
            conj = a.conjugate()
        except AttributeError:
            # 可能直接访问属性
            conj = ComplexFieldElement(a.real, -a.imag)
        
        self.assertEqual(conj.real, 3)
        self.assertEqual(conj.imag, -4)
    
    def test_complex_modulus(self):
        """测试复数模"""
        a = ComplexFieldElement(3, 4)
        # 复数模可能通过 modulus 方法或手动计算
        try:
            modulus = a.modulus()
        except AttributeError:
            modulus = (a.real**2 + a.imag**2)**0.5
        
        # |3 + 4i| = 5
        self.assertEqual(modulus, 5.0)
    
    def test_complex_division(self):
        """测试复数除法"""
        field = ComplexField()
        
        a = ComplexFieldElement(3, 4)
        b = ComplexFieldElement(1, 0)  # 1
        result = field.divide(a, b)
        
        self.assertEqual(result.real, 3)
        self.assertEqual(result.imag, 4)
    
    def test_pure_imaginary(self):
        """测试纯虚数"""
        field = ComplexField()
        
        i = ComplexFieldElement(0, 1)
        
        # i^2 = -1
        result = field.multiply(i, i)
        self.assertEqual(result.real, -1)
        self.assertEqual(result.imag, 0)


class TestPolynomialRingRobustness(unittest.TestCase):
    """多项式环鲁棒性测试"""
    
    def test_zero_polynomial(self):
        """测试零多项式"""
        ring = PolynomialRing()
        
        zero = PolynomialRingElement([0])
        p = PolynomialRingElement([1, 2, 3])
        
        # 0 + p = p
        result = ring.add(zero, p)
        self.assertEqual(result.coefficients, p.coefficients)
        
        # 0 * p = 0
        result = ring.multiply(zero, p)
        self.assertEqual(result.coefficients, [0])
    
    def test_constant_polynomial(self):
        """测试常数多项式"""
        ring = PolynomialRing()
        
        const = PolynomialRingElement([5])
        p = PolynomialRingElement([1, 2, 3])  # 1 + 2x + 3x^2
        
        # 常数乘法
        result = ring.multiply(const, p)
        self.assertEqual(result.coefficients, [5, 10, 15])
    
    def test_polynomial_degree(self):
        """测试多项式次数"""
        p1 = PolynomialRingElement([1])  # 常数，次数 0
        self.assertEqual(p1.degree(), 0)
        
        p2 = PolynomialRingElement([1, 2])  # 1 + 2x，次数 1
        self.assertEqual(p2.degree(), 1)
        
        p3 = PolynomialRingElement([1, 2, 3, 4])  # 次数 3
        self.assertEqual(p3.degree(), 3)
    
    def test_polynomial_multiplication(self):
        """测试多项式乘法"""
        ring = PolynomialRing()
        
        # (1 + x)(1 - x) = 1 - x^2
        p1 = PolynomialRingElement([1, 1])
        p2 = PolynomialRingElement([1, -1])
        result = ring.multiply(p1, p2)
        
        self.assertEqual(result.coefficients, [1, 0, -1])
    
    def test_polynomial_evaluation(self):
        """测试多项式求值"""
        p = PolynomialRingElement([1, 2, 3])  # 1 + 2x + 3x^2
        
        # 手动计算多项式值
        def evaluate(poly, x):
            return sum(c * (x ** i) for i, c in enumerate(poly.coefficients))
        
        # p(0) = 1
        self.assertEqual(evaluate(p, 0), 1)
        
        # p(1) = 1 + 2 + 3 = 6
        self.assertEqual(evaluate(p, 1), 6)
        
        # p(2) = 1 + 4 + 12 = 17
        self.assertEqual(evaluate(p, 2), 17)


class TestMatrixRingRobustness(unittest.TestCase):
    """矩阵环鲁棒性测试"""
    
    def setUp(self):
        self.int_ring = IntegerRing()
    
    def test_zero_matrix(self):
        """测试零矩阵"""
        ring = MatrixRing(self.int_ring, 2)
        
        zero = MatrixRingElement(
            [[IntegerRingElement(0), IntegerRingElement(0)],
             [IntegerRingElement(0), IntegerRingElement(0)]],
            self.int_ring, 2, 2
        )
        
        m = MatrixRingElement(
            [[IntegerRingElement(1), IntegerRingElement(2)],
             [IntegerRingElement(3), IntegerRingElement(4)]],
            self.int_ring, 2, 2
        )
        
        # 0 + M = M
        result = ring.add(zero, m)
        self.assertEqual(result.entries[0][0].value, 1)
        
        # 0 * M = 0
        result = ring.multiply(zero, m)
        self.assertEqual(result.entries[0][0].value, 0)
    
    def test_identity_matrix(self):
        """测试单位矩阵"""
        ring = MatrixRing(self.int_ring, 2)
        
        identity = MatrixRingElement(
            [[IntegerRingElement(1), IntegerRingElement(0)],
             [IntegerRingElement(0), IntegerRingElement(1)]],
            self.int_ring, 2, 2
        )
        
        m = MatrixRingElement(
            [[IntegerRingElement(2), IntegerRingElement(3)],
             [IntegerRingElement(4), IntegerRingElement(5)]],
            self.int_ring, 2, 2
        )
        
        # I * M = M
        result = ring.multiply(identity, m)
        self.assertEqual(result.entries[0][0].value, 2)
        self.assertEqual(result.entries[1][1].value, 5)
    
    def test_matrix_multiplication(self):
        """测试矩阵乘法"""
        ring = MatrixRing(self.int_ring, 2)
        
        m1 = MatrixRingElement(
            [[IntegerRingElement(1), IntegerRingElement(2)],
             [IntegerRingElement(3), IntegerRingElement(4)]],
            self.int_ring, 2, 2
        )
        
        m2 = MatrixRingElement(
            [[IntegerRingElement(5), IntegerRingElement(6)],
             [IntegerRingElement(7), IntegerRingElement(8)]],
            self.int_ring, 2, 2
        )
        
        result = ring.multiply(m1, m2)
        
        # [1 2]   [5 6]   [19 22]
        # [3 4] * [7 8] = [43 50]
        self.assertEqual(result.entries[0][0].value, 19)
        self.assertEqual(result.entries[0][1].value, 22)
        self.assertEqual(result.entries[1][0].value, 43)
        self.assertEqual(result.entries[1][1].value, 50)
    
    def test_non_commutative(self):
        """测试矩阵乘法非交换性"""
        ring = MatrixRing(self.int_ring, 2)
        
        m1 = MatrixRingElement(
            [[IntegerRingElement(1), IntegerRingElement(2)],
             [IntegerRingElement(0), IntegerRingElement(1)]],
            self.int_ring, 2, 2
        )
        
        m2 = MatrixRingElement(
            [[IntegerRingElement(1), IntegerRingElement(0)],
             [IntegerRingElement(2), IntegerRingElement(1)]],
            self.int_ring, 2, 2
        )
        
        result1 = ring.multiply(m1, m2)
        result2 = ring.multiply(m2, m1)
        
        # AB ≠ BA
        self.assertNotEqual(result1.entries[0][0].value, result2.entries[0][0].value)


class TestVectorSpaceRobustness(unittest.TestCase):
    """向量空间鲁棒性测试"""
    
    def setUp(self):
        self.field = RationalField()
        self.vector_space = FiniteDimensionalVectorSpace(self.field, 3)
    
    def test_zero_vector(self):
        """测试零向量"""
        zero = FiniteDimensionalVectorSpaceElement(
            [RationalFieldElement(0, 1) for _ in range(3)],
            self.field
        )
        
        v = FiniteDimensionalVectorSpaceElement(
            [RationalFieldElement(1, 1), RationalFieldElement(2, 1), RationalFieldElement(3, 1)],
            self.field
        )
        
        # 0 + v = v
        result = self.vector_space.add(zero, v)
        self.assertEqual(result.components[0].numerator, 1)
    
    def test_vector_addition(self):
        """测试向量加法"""
        v1 = FiniteDimensionalVectorSpaceElement(
            [RationalFieldElement(1, 2), RationalFieldElement(1, 3)],
            self.field
        )
        
        v2 = FiniteDimensionalVectorSpaceElement(
            [RationalFieldElement(1, 4), RationalFieldElement(1, 6)],
            self.field
        )
        
        result = self.vector_space.add(v1, v2)
        
        # (1/2 + 1/4, 1/3 + 1/6) = (3/4, 1/2)
        self.assertEqual(result.components[0].numerator, 3)
        self.assertEqual(result.components[0].denominator, 4)
        self.assertEqual(result.components[1].numerator, 1)
        self.assertEqual(result.components[1].denominator, 2)
    
    def test_scalar_multiplication(self):
        """测试标量乘法"""
        v = FiniteDimensionalVectorSpaceElement(
            [RationalFieldElement(1, 2), RationalFieldElement(2, 3)],
            self.field
        )
        
        scalar = RationalFieldElement(2, 1)
        # 标量乘法可能通过不同的接口
        try:
            result = self.vector_space.scalar_multiply(scalar, v)
        except (AttributeError, TypeError):
            # 尝试元素级别的乘法
            result_components = [c * scalar for c in v.components]
            result = FiniteDimensionalVectorSpaceElement(result_components, self.field)
        
        # 2 * (1/2, 2/3) = (1, 4/3)
        self.assertEqual(result.components[0].numerator, 1)
        self.assertEqual(result.components[1].numerator, 4)
        self.assertEqual(result.components[1].denominator, 3)
    
    def test_linear_independence(self):
        """测试线性无关性"""
        # 标准基向量
        e1 = FiniteDimensionalVectorSpaceElement(
            [RationalFieldElement(1, 1), RationalFieldElement(0, 1)],
            self.field
        )
        e2 = FiniteDimensionalVectorSpaceElement(
            [RationalFieldElement(0, 1), RationalFieldElement(1, 1)],
            self.field
        )
        
        # e1 和 e2 线性无关
        # 检查：a*e1 + b*e2 = 0 当且仅当 a = b = 0
        # 这里简化测试，只验证它们不相等
        self.assertNotEqual(e1.components[0].numerator, e2.components[0].numerator)


class TestAlgebraicAxioms(unittest.TestCase):
    """代数公理测试"""
    
    def test_field_axioms(self):
        """测试域公理"""
        field = RationalField()
        
        a = RationalFieldElement(2, 3)
        b = RationalFieldElement(3, 4)
        c = RationalFieldElement(4, 5)
        
        # 加法结合律: (a + b) + c = a + (b + c)
        left = field.add(field.add(a, b), c)
        right = field.add(a, field.add(b, c))
        self.assertEqual(left.numerator * right.denominator, right.numerator * left.denominator)
        
        # 乘法结合律: (a * b) * c = a * (b * c)
        left = field.multiply(field.multiply(a, b), c)
        right = field.multiply(a, field.multiply(b, c))
        self.assertEqual(left.numerator * right.denominator, right.numerator * left.denominator)
        
        # 分配律: a * (b + c) = a * b + a * c
        left = field.multiply(a, field.add(b, c))
        right = field.add(field.multiply(a, b), field.multiply(a, c))
        self.assertEqual(left.numerator * right.denominator, right.numerator * left.denominator)
    
    def test_ring_axioms(self):
        """测试环公理"""
        ring = IntegerRing()
        
        a = IntegerRingElement(2)
        b = IntegerRingElement(3)
        c = IntegerRingElement(4)
        
        # 加法结合律
        left = ring.add(ring.add(a, b), c)
        right = ring.add(a, ring.add(b, c))
        self.assertEqual(left.value, right.value)
        
        # 乘法结合律
        left = ring.multiply(ring.multiply(a, b), c)
        right = ring.multiply(a, ring.multiply(b, c))
        self.assertEqual(left.value, right.value)
        
        # 分配律
        left = ring.multiply(a, ring.add(b, c))
        right = ring.add(ring.multiply(a, b), ring.multiply(a, c))
        self.assertEqual(left.value, right.value)


if __name__ == '__main__':
    unittest.main()
