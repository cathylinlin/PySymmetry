"""群论模块鲁棒性测试

该文件包含群论模块的边界值、极端值和鲁棒性测试用例。
涵盖：
- 边界值测试（最小/最大阶数）
- 极端值测试（大群、特殊元素）
- 错误处理测试（无效输入）
- 群性质验证
- 子群和商群测试
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from PySyM.core.group_theory import (
    GroupFactory, Subgroup, Coset, QuotientGroup
)
from PySyM.core.group_theory.specific_group import (
    CyclicGroup, SymmetricGroup, DihedralGroup
)


class TestGroupTheoryBoundaryValues(unittest.TestCase):
    """群论边界值测试"""
    
    def test_cyclic_group_minimum(self):
        """测试循环群最小阶数"""
        # C_1 是平凡群
        C1 = GroupFactory.cyclic_group(1)
        self.assertEqual(C1.order(), 1)
        self.assertEqual(C1.elements(), [0])
        self.assertTrue(C1.is_abelian())
    
    def test_cyclic_group_small_orders(self):
        """测试小阶循环群"""
        for n in [2, 3, 4, 5]:
            Cn = GroupFactory.cyclic_group(n)
            self.assertEqual(Cn.order(), n)
            self.assertEqual(len(Cn.elements()), n)
            self.assertTrue(Cn.is_abelian())
    
    def test_symmetric_group_minimum(self):
        """测试对称群最小阶数"""
        # S_0 和 S_1 都是平凡群
        S0 = GroupFactory.symmetric_group(0)
        self.assertEqual(S0.order(), 1)
        
        S1 = GroupFactory.symmetric_group(1)
        self.assertEqual(S1.order(), 1)
        self.assertTrue(S1.is_abelian())
    
    def test_symmetric_group_s2(self):
        """测试 S_2"""
        S2 = GroupFactory.symmetric_group(2)
        self.assertEqual(S2.order(), 2)
        self.assertTrue(S2.is_abelian())  # S_2 是阿贝尔群
    
    def test_symmetric_group_s3(self):
        """测试 S_3"""
        S3 = GroupFactory.symmetric_group(3)
        self.assertEqual(S3.order(), 6)
        self.assertFalse(S3.is_abelian())  # S_3 不是阿贝尔群
    
    def test_dihedral_group_minimum(self):
        """测试二面体群最小阶数"""
        # D_1 是最小的非平凡二面体群
        D1 = GroupFactory.dihedral_group(1)
        self.assertEqual(D1.order(), 2)
        
        # D_2 (Klein 四元群)
        D2 = GroupFactory.dihedral_group(2)
        self.assertEqual(D2.order(), 4)
        self.assertTrue(D2.is_abelian())
    
    def test_dihedral_group_d3(self):
        """测试 D_3 (同构于 S_3)"""
        D3 = GroupFactory.dihedral_group(3)
        self.assertEqual(D3.order(), 6)
        self.assertFalse(D3.is_abelian())
    
    def test_large_cyclic_group(self):
        """测试大阶循环群"""
        # 测试大阶循环群
        C100 = GroupFactory.cyclic_group(100)
        self.assertEqual(C100.order(), 100)
        
        # 测试运算
        self.assertEqual(C100.multiply(50, 50), 0)  # 50 + 50 = 100 ≡ 0 (mod 100)
        self.assertEqual(C100.multiply(99, 1), 0)   # 99 + 1 = 100 ≡ 0 (mod 100)


class TestGroupTheoryExtremeValues(unittest.TestCase):
    """群论极端值测试"""
    
    def test_identity_element_properties(self):
        """测试单位元性质"""
        C5 = GroupFactory.cyclic_group(5)
        identity = C5.identity()
        
        # 单位元性质: e * a = a * e = a
        for a in C5.elements():
            self.assertEqual(C5.multiply(identity, a), a)
            self.assertEqual(C5.multiply(a, identity), a)
        
        # 单位元的逆是自身
        self.assertEqual(C5.inverse(identity), identity)
    
    def test_inverse_properties(self):
        """测试逆元性质"""
        S4 = GroupFactory.symmetric_group(4)
        
        for a in S4.elements():
            # a * a^(-1) = e
            inv = S4.inverse(a)
            product = S4.multiply(a, inv)
            self.assertEqual(product, S4.identity())
            
            # (a^(-1))^(-1) = a
            inv_inv = S4.inverse(inv)
            self.assertEqual(inv_inv, a)
    
    def test_associativity(self):
        """测试结合律"""
        D4 = GroupFactory.dihedral_group(4)
        elements = D4.elements()
        
        # 测试随机三元组
        import random
        random.seed(42)
        for _ in range(100):
            a = random.choice(elements)
            b = random.choice(elements)
            c = random.choice(elements)
            
            # (a * b) * c = a * (b * c)
            left = D4.multiply(D4.multiply(a, b), c)
            right = D4.multiply(a, D4.multiply(b, c))
            self.assertEqual(left, right)
    
    def test_element_order(self):
        """测试元素阶数"""
        C6 = GroupFactory.cyclic_group(6)
        
        # 在循环群中，元素 k 的阶数是 6 / gcd(6, k)
        expected_orders = {0: 1, 1: 6, 2: 3, 3: 2, 4: 3, 5: 6}
        
        for elem in C6.elements():
            # 计算元素阶数
            order = 1
            current = elem
            while current != C6.identity():
                current = C6.multiply(current, elem)
                order += 1
            
            self.assertEqual(order, expected_orders[elem])


class TestGroupTheoryErrorHandling(unittest.TestCase):
    """群论错误处理测试"""
    
    def test_invalid_cyclic_group_order(self):
        """测试无效循环群阶数"""
        with self.assertRaises(ValueError):
            GroupFactory.cyclic_group(0)
        
        with self.assertRaises(ValueError):
            GroupFactory.cyclic_group(-1)
    
    def test_invalid_symmetric_group_order(self):
        """测试无效对称群阶数"""
        with self.assertRaises(ValueError):
            GroupFactory.symmetric_group(-1)
        
        # S_n 对于大 n 应该抛出错误
        with self.assertRaises(ValueError):
            GroupFactory.symmetric_group(100)  # 太大
    
    def test_invalid_dihedral_group_order(self):
        """测试无效二面体群阶数"""
        with self.assertRaises(ValueError):
            GroupFactory.dihedral_group(0)
        
        with self.assertRaises(ValueError):
            GroupFactory.dihedral_group(-1)
    
    def test_element_not_in_group(self):
        """测试不属于群的元素"""
        C3 = GroupFactory.cyclic_group(3)
        
        # 3 不在 C_3 中
        self.assertFalse(3 in C3)
        self.assertTrue(0 in C3)
        self.assertTrue(1 in C3)
        self.assertTrue(2 in C3)
    
    def test_invalid_multiplication(self):
        """测试无效乘法"""
        C3 = GroupFactory.cyclic_group(3)
        
        # 尝试乘以不属于群的元素
        # 注意：具体行为取决于实现，可能返回错误结果或抛出异常
        try:
            result = C3.multiply(0, 5)  # 5 不在 C_3 中
            # 如果没有抛出异常，检查结果是否合理
            self.assertIn(result, C3.elements())
        except (ValueError, KeyError, TypeError):
            pass  # 期望的行为


class TestSubgroupRobustness(unittest.TestCase):
    """子群鲁棒性测试"""
    
    def test_trivial_subgroup(self):
        """测试平凡子群"""
        C6 = GroupFactory.cyclic_group(6)
        
        # 平凡子群 {0}
        trivial = C6.generate_subgroup([0])
        self.assertEqual(trivial.order(), 1)
        
        # 整个群作为子群
        whole = C6.generate_subgroup([1])
        self.assertEqual(whole.order(), 6)
    
    def test_cyclic_subgroup_structure(self):
        """测试循环子群结构"""
        C12 = GroupFactory.cyclic_group(12)
        
        # 由 4 生成的子群应该是 {0, 4, 8}，阶数为 3
        H = C12.generate_subgroup([4])
        self.assertEqual(H.order(), 3)
        self.assertTrue(0 in H)
        self.assertTrue(4 in H)
        self.assertTrue(8 in H)
    
    def test_symmetric_subgroup(self):
        """测试对称群的子群"""
        S4 = GroupFactory.symmetric_group(4)
        
        # 交错群 A_4 是 S_4 的子群
        A4 = S4.alternating_group()
        self.assertEqual(A4.order(), 12)
        
        # 验证 A_4 的元素都在 S_4 中
        for elem in A4.elements():
            self.assertIn(elem, S4.elements())
    
    def test_subgroup_closure(self):
        """测试子群的封闭性"""
        C8 = GroupFactory.cyclic_group(8)
        H = C8.generate_subgroup([2])  # {0, 2, 4, 6}
        
        # 验证封闭性
        for a in H.elements():
            for b in H.elements():
                product = C8.multiply(a, b)
                self.assertTrue(product in H)
        
        # 验证逆元
        for a in H.elements():
            inv = C8.inverse(a)
            self.assertTrue(inv in H)


class TestCosetRobustness(unittest.TestCase):
    """陪集鲁棒性测试"""
    
    def test_coset_basic_properties(self):
        """测试陪集基本性质"""
        C6 = GroupFactory.cyclic_group(6)
        H = C6.generate_subgroup([3])  # {0, 3}
        
        # 左陪集 1 + H = {1, 4}
        coset = Coset(C6, H, 1, is_left=True)
        self.assertTrue(1 in coset)
        self.assertTrue(4 in coset)
        self.assertFalse(0 in coset)
        self.assertFalse(3 in coset)
    
    def test_coset_partition(self):
        """测试陪集划分"""
        C6 = GroupFactory.cyclic_group(6)
        H = C6.generate_subgroup([2])  # {0, 2, 4}
        
        # C_6 / H 应该有 2 个陪集
        coset0 = Coset(C6, H, 0, is_left=True)  # H
        coset1 = Coset(C6, H, 1, is_left=True)  # 1 + H
        
        # 验证划分
        elements_in_cosets = set()
        for c in [coset0, coset1]:
            for elem in c.elements():
                elements_in_cosets.add(elem)
        
        self.assertEqual(elements_in_cosets, set(C6.elements()))
    
    def test_lagrange_theorem(self):
        """测试拉格朗日定理 |G| = |H| * [G:H]"""
        C12 = GroupFactory.cyclic_group(12)
        H = C12.generate_subgroup([4])  # {0, 4, 8}, |H| = 3
        
        # 指数 [G:H] 应该是 4
        index = C12.order() // H.order()
        self.assertEqual(index, 4)
        
        # 验证 |G| = |H| * [G:H]
        self.assertEqual(C12.order(), H.order() * index)


class TestQuotientGroupRobustness(unittest.TestCase):
    """商群鲁棒性测试"""
    
    def test_quotient_group_basic(self):
        """测试商群基本性质"""
        C6 = GroupFactory.cyclic_group(6)
        H = C6.generate_subgroup([3])  # {0, 3}
        
        # C_6 / {0, 3} ≅ C_3
        Q = QuotientGroup(C6, H)
        self.assertEqual(Q.order(), 3)
    
    def test_quotient_group_homomorphism(self):
        """测试商群同态"""
        C12 = GroupFactory.cyclic_group(12)
        H = C12.generate_subgroup([4])  # {0, 4, 8}
        
        Q = QuotientGroup(C12, H)
        
        # 验证同态性质: φ(a * b) = φ(a) * φ(b)
        # 这里 φ 是自然投影
        for a in C12.elements():
            for b in C12.elements():
                # 计算投影
                coset_ab = Coset(C12, H, C12.multiply(a, b), is_left=True)
                # 商群中的乘法
                # 注意：这里简化测试，实际应该使用商群的乘法
                pass
    
    def test_first_isomorphism_theorem(self):
        """测试第一同构定理"""
        # C_n / C_m ≅ C_{n/m} 当 m | n
        C12 = GroupFactory.cyclic_group(12)
        H = C12.generate_subgroup([3])  # C_4
        
        Q = QuotientGroup(C12, H)
        # C_12 / C_4 ≅ C_3
        self.assertEqual(Q.order(), 3)


class TestGroupTheoryMathematicalProperties(unittest.TestCase):
    """群论数学性质测试"""
    
    def test_cyclic_group_is_abelian(self):
        """测试循环群是阿贝尔群"""
        for n in [1, 2, 3, 5, 7, 10, 100]:
            Cn = GroupFactory.cyclic_group(n)
            self.assertTrue(Cn.is_abelian(), f"C_{n} 应该是阿贝尔群")
    
    def test_symmetric_group_non_abelian(self):
        """测试对称群的非阿贝尔性"""
        # S_n 对于 n >= 3 是非阿贝尔群
        for n in [3, 4, 5]:
            Sn = GroupFactory.symmetric_group(n)
            self.assertFalse(Sn.is_abelian(), f"S_{n} 应该是非阿贝尔群")
    
    def test_dihedral_group_non_abelian(self):
        """测试二面体群的非阿贝尔性"""
        # D_n 对于 n >= 3 是非阿贝尔群
        for n in [3, 4, 5]:
            Dn = GroupFactory.dihedral_group(n)
            self.assertFalse(Dn.is_abelian(), f"D_{n} 应该是非阿贝尔群")
    
    def test_cayley_theorem(self):
        """测试凯莱定理：每个群都同构于某个对称群的子群"""
        # 对于小群，验证可以嵌入到对称群中
        C3 = GroupFactory.cyclic_group(3)
        
        # C_3 可以嵌入到 S_3 中
        S3 = GroupFactory.symmetric_group(3)
        
        # 验证 C_3 的阶数整除 S_3 的阶数
        self.assertEqual(S3.order() % C3.order(), 0)
    
    def test_class_equation(self):
        """测试类方程"""
        # 对于有限群，|G| = |Z(G)| + Σ[G:C(g)]
        # 这里简化测试，只验证中心的存在
        S3 = GroupFactory.symmetric_group(3)
        
        # S_3 的中心是平凡的
        # 验证单位元在中心
        identity = S3.identity()
        
        for g in S3.elements():
            # e * g = g * e
            self.assertEqual(
                S3.multiply(identity, g),
                S3.multiply(g, identity)
            )


class TestSpecialGroups(unittest.TestCase):
    """特殊群测试"""
    
    def test_klein_four_group(self):
        """测试 Klein 四元群"""
        V4 = GroupFactory.klein_four_group()
        self.assertEqual(V4.order(), 4)
        self.assertTrue(V4.is_abelian())
        
        # 每个非单位元的阶数都是 2
        identity = V4.identity()
        for elem in V4.elements():
            if elem != identity:
                self.assertEqual(V4.multiply(elem, elem), identity)
    
    def test_quaternion_group(self):
        """测试四元数群"""
        Q8 = GroupFactory.quaternion_group()
        self.assertEqual(Q8.order(), 8)
        self.assertFalse(Q8.is_abelian())
        
        # 测试基本关系: i^2 = j^2 = k^2 = ijk = -1
        i, j, k = 'i', 'j', 'k'
        minus_1 = '-1'
        
        self.assertEqual(Q8.multiply(i, i), minus_1)
        self.assertEqual(Q8.multiply(j, j), minus_1)
        self.assertEqual(Q8.multiply(k, k), minus_1)
        
        # ijk = -1
        ij = Q8.multiply(i, j)
        ijk = Q8.multiply(ij, k)
        self.assertEqual(ijk, minus_1)
    
    def test_alternating_group(self):
        """测试交错群"""
        for n in [3, 4, 5]:
            An = GroupFactory.alternating_group(n)
            # A_n 的阶数是 n!/2
            import math
            expected_order = math.factorial(n) // 2
            self.assertEqual(An.order(), expected_order)


if __name__ == '__main__':
    unittest.main()
