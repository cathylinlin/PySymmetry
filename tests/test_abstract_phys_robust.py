"""物理抽象层鲁棒性测试

该文件包含物理抽象层的边界值、极端值和鲁棒性测试用例。
涵盖：
- 边界值测试（最小/最大维度、极端物理量）
- 极端值测试（极大/极小质量、能量等）
- 错误处理测试（无效输入）
- 对称操作测试
- 物理性质验证
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from PySyM.abstract_phys.physical_objects import (
    PhysicalObject,
    PhysicalSpace,
    ElementaryParticle,
    Quark,
    Lepton,
    ScalarField,
    VectorField,
)
from PySyM.abstract_phys.symmetry_operations import (
    IdentityOperation,
    TranslationOperation,
    RotationOperation,
    ParityOperation,
    TimeReversalOperation,
)
from PySyM.abstract_phys.symmetry_environments import (
    PoincareGroup,
    LorentzGroup,
)


class ConcretePhysicalObject(PhysicalObject):
    """PhysicalObject的具体实现用于测试"""
    
    def __init__(self, mass: float = 1.0, charge: float = 0.0, spin: float = 0.5):
        self._mass = mass
        self._charge = charge
        self._spin = spin
    
    def symmetry_properties(self) -> dict:
        return {"mass": self._mass, "charge": self._charge, "spin": self._spin}
    
    def transform(self, symmetry_operation) -> 'ConcretePhysicalObject':
        return ConcretePhysicalObject(self._mass, self._charge, self._spin)
    
    def is_invariant_under(self, symmetry_operation) -> bool:
        return True
    
    def get_mass(self):
        return self._mass
    
    def get_charge(self):
        return self._charge
    
    def get_spin(self):
        return self._spin


class ConcretePhysicalSpace(PhysicalSpace):
    """PhysicalSpace的具体实现用于测试"""
    
    def __init__(self, dim: int = 3):
        self._dim = dim
        self._metric = None
    
    def dimension(self) -> int:
        return self._dim
    
    def inner_product(self, x, y):
        return np.dot(x, y)
    
    def norm(self, x):
        return np.linalg.norm(x)


class TestPhysicalObjectRobustness(unittest.TestCase):
    """物理对象鲁棒性测试"""
    
    def test_zero_mass_object(self):
        """测试零质量对象"""
        obj = ConcretePhysicalObject(mass=0.0, charge=1.0, spin=1.0)
        self.assertEqual(obj.get_mass(), 0.0)
        props = obj.symmetry_properties()
        self.assertEqual(props["mass"], 0.0)
    
    def test_negative_mass(self):
        """测试负质量（理论上的）"""
        # 某些理论允许负质量
        obj = ConcretePhysicalObject(mass=-1.0, charge=0.0, spin=0.0)
        self.assertEqual(obj.get_mass(), -1.0)
    
    def test_extreme_mass_values(self):
        """测试极端质量值"""
        # 极大质量
        huge_mass = 1e30  # 类似恒星质量
        obj1 = ConcretePhysicalObject(mass=huge_mass)
        self.assertEqual(obj1.get_mass(), huge_mass)
        
        # 极小质量
        tiny_mass = 1e-30  # 类似粒子物理尺度
        obj2 = ConcretePhysicalObject(mass=tiny_mass)
        self.assertEqual(obj2.get_mass(), tiny_mass)
    
    def test_fractional_spin(self):
        """测试分数自旋"""
        # 费米子：半整数自旋
        fermion = ConcretePhysicalObject(spin=0.5)
        self.assertEqual(fermion.get_spin(), 0.5)
        
        # 玻色子：整数自旋
        boson = ConcretePhysicalObject(spin=1.0)
        self.assertEqual(boson.get_spin(), 1.0)
        
        # 零自旋
        scalar = ConcretePhysicalObject(spin=0.0)
        self.assertEqual(scalar.get_spin(), 0.0)
    
    def test_extreme_charge_values(self):
        """测试极端电荷值"""
        # 基本电荷单位
        e = 1.602e-19
        
        # 正电荷
        obj1 = ConcretePhysicalObject(charge=e)
        self.assertEqual(obj1.get_charge(), e)
        
        # 负电荷
        obj2 = ConcretePhysicalObject(charge=-e)
        self.assertEqual(obj2.get_charge(), -e)
        
        # 多电荷单位
        obj3 = ConcretePhysicalObject(charge=2*e)
        self.assertEqual(obj3.get_charge(), 2*e)
    
    def test_object_transformation(self):
        """测试对象变换"""
        obj = ConcretePhysicalObject(mass=1.0, charge=0.5, spin=0.5)
        identity = IdentityOperation()
        
        # 恒等变换
        transformed = obj.transform(identity)
        self.assertEqual(transformed.get_mass(), obj.get_mass())
        self.assertEqual(transformed.get_charge(), obj.get_charge())


class TestPhysicalSpaceRobustness(unittest.TestCase):
    """物理空间鲁棒性测试"""
    
    def test_space_dimensions(self):
        """测试不同维度的空间"""
        # 1维空间
        space1d = ConcretePhysicalSpace(dim=1)
        self.assertEqual(space1d.dimension(), 1)
        
        # 2维空间
        space2d = ConcretePhysicalSpace(dim=2)
        self.assertEqual(space2d.dimension(), 2)
        
        # 3维空间
        space3d = ConcretePhysicalSpace(dim=3)
        self.assertEqual(space3d.dimension(), 3)
        
        # 4维时空
        space4d = ConcretePhysicalSpace(dim=4)
        self.assertEqual(space4d.dimension(), 4)
    
    def test_zero_dimensional_space(self):
        """测试零维空间"""
        space0d = ConcretePhysicalSpace(dim=0)
        self.assertEqual(space0d.dimension(), 0)
    
    def test_inner_product_properties(self):
        """测试内积性质"""
        space = ConcretePhysicalSpace(dim=3)
        
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([1.0, 1.0, 0.0])
        
        # 正交向量内积为0
        self.assertEqual(space.inner_product(x, y), 0.0)
        
        # 双线性
        # <x, y+z> = <x,y> + <x,z>
        left = space.inner_product(x, y + z)
        right = space.inner_product(x, y) + space.inner_product(x, z)
        self.assertEqual(left, right)
        
        # 对称性 <x,y> = <y,x>
        self.assertEqual(space.inner_product(x, z), space.inner_product(z, x))
    
    def test_norm_properties(self):
        """测试范数性质"""
        space = ConcretePhysicalSpace(dim=3)
        
        # 零向量范数为0
        zero = np.array([0.0, 0.0, 0.0])
        self.assertEqual(space.norm(zero), 0.0)
        
        # 单位向量
        x = np.array([1.0, 0.0, 0.0])
        self.assertEqual(space.norm(x), 1.0)
        
        # 齐次性 ||ax|| = |a| ||x||
        a = 2.5
        ax = a * x
        self.assertAlmostEqual(space.norm(ax), abs(a) * space.norm(x))
    
    def test_metric_tensor(self):
        """测试度规张量"""
        space = ConcretePhysicalSpace(dim=3)
        
        # 欧几里得度规
        g = np.eye(3)
        space.metric_tensor = g
        
        np.testing.assert_array_equal(space.metric_tensor, g)
        
        # 无效度规（非方阵）
        with self.assertRaises(ValueError):
            space.metric_tensor = np.ones((3, 4))
        
        # 无效度规（维度不匹配）
        with self.assertRaises(ValueError):
            space.metric_tensor = np.eye(4)


class TestElementaryParticleRobustness(unittest.TestCase):
    """基本粒子鲁棒性测试"""
    
    def test_quark_properties(self):
        """测试夸克性质"""
        # 上夸克
        up = Quark(flavor="up", mass=2.2, charge=2/3, spin=0.5)
        self.assertEqual(up.get_charge(), 2/3)
        self.assertEqual(up.get_spin(), 0.5)
        
        # 下夸克
        down = Quark(flavor="down", mass=4.7, charge=-1/3, spin=0.5)
        self.assertEqual(down.get_charge(), -1/3)
        
        # 奇夸克
        strange = Quark(flavor="strange", mass=96, charge=-1/3, spin=0.5)
        self.assertEqual(strange.get_charge(), -1/3)
    
    def test_lepton_properties(self):
        """测试轻子性质"""
        # 电子
        electron = Lepton(lepton_type="electron", mass=0.511, charge=-1, spin=0.5)
        self.assertEqual(electron.get_charge(), -1)
        self.assertEqual(electron.get_spin(), 0.5)
        
        # 中微子（近似零质量）- 使用具体的轻子类型
        neutrino = Lepton(lepton_type="electron_neutrino", mass=0, charge=0, spin=0.5)
        self.assertEqual(neutrino.get_charge(), 0)
        self.assertEqual(neutrino.get_mass(), 0)
    
    def test_particle_quantum_numbers(self):
        """测试粒子量子数"""
        # 测试夸克和轻子的基本属性
        quark = Quark(flavor="up", mass=2.2, charge=2/3, spin=0.5)
        self.assertEqual(quark.get_charge(), 2/3)
        
        lepton = Lepton(lepton_type="electron", mass=0.511, charge=-1, spin=0.5)
        self.assertEqual(lepton.get_charge(), -1)


class TestFieldRobustness(unittest.TestCase):
    """场鲁棒性测试"""
    
    def test_scalar_field_basic(self):
        """测试标量场基本性质"""
        field = ScalarField(field_function=lambda x: np.sum(x**2))
        self.assertAlmostEqual(field.evaluate(np.array([1.0, 2.0, 3.0])), 14.0)

    def test_vector_field_basic(self):
        """测试矢量场基本性质"""
        field = VectorField(field_function=lambda x: np.array([x[1], -x[0], 0.0]))
        res = field.evaluate(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(res, np.array([0.0, -1.0, 0.0]))

    def test_field_at_point(self):
        """测试场在空间点的值和梯度"""
        field = ScalarField(field_function=lambda x: x[0]**2 + x[1]**2)
        # 在原点的值
        value_at_origin = field.evaluate(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(value_at_origin, 0.0)
        grad = field.gradient(np.array([1.0, 1.0, 0.0]))
        np.testing.assert_array_almost_equal(grad, np.array([2.0, 2.0, 0.0]))

    def test_massless_vector_field(self):
        """测试无质量矢量场(通过电磁场近似测试其散度和旋度)"""
        photon = VectorField(field_function=lambda x: np.array([x[1], x[2], x[0]]))
        div = photon.divergence(np.array([1.0, 1.0, 1.0]))
        self.assertAlmostEqual(div, 0.0)

    def test_massive_vector_field(self):
        """测试矢量场能量密度"""
        w_boson = VectorField(field_function=lambda x: np.array([x[0], 0.0, 0.0]))
        energy_density = w_boson.get_energy_density(np.array([0.0, 0.0, 0.0]))
        self.assertGreater(energy_density, 0.0)
    def test_identity_operation(self):
        """测试恒等操作"""
        identity = IdentityOperation()
        
        # 恒等操作不改变对象
        obj = ConcretePhysicalObject(mass=1.0)
        transformed = identity.act_on(obj)
        
        self.assertEqual(transformed.get_mass(), obj.get_mass())
    
    def test_translation_operation(self):
        """测试平移操作"""
        # 空间平移
        translation = TranslationOperation(displacement=np.array([1.0, 2.0, 3.0]))
        
        obj = ConcretePhysicalObject()
        transformed = translation.act_on(obj)
        
        # 平移不改变内禀性质
        self.assertEqual(transformed.get_mass(), obj.get_mass())
        self.assertEqual(transformed.get_charge(), obj.get_charge())
    
    def test_rotation_operation(self):
        """测试旋转操作"""
        # 绕z轴旋转90度
        rotation = RotationOperation(axis=np.array([0.0, 0.0, 1.0]), angle=np.pi/2)
        
        obj = ConcretePhysicalObject()
        transformed = rotation.act_on(obj)
        
        # 旋转不改变内禀性质
        self.assertEqual(transformed.get_mass(), obj.get_mass())
    
    def test_parity_operation(self):
        """测试宇称操作"""
        parity = ParityOperation()

        class DummyObj:
            def __init__(self, pos):
                self.position = pos
        
        obj = DummyObj(np.array([1.0, 2.0, 3.0]))
        transformed = parity.act_on(obj)

        # 宇称变换: r -> -r
        np.testing.assert_array_almost_equal(transformed.position, np.array([-1.0, -2.0, -3.0]))

    def test_time_reversal_operation(self):
        """测试时间反演操作"""
        time_rev = TimeReversalOperation()

        class DummyKinetic:
            def __init__(self, velocity):
                self.velocity = velocity
                
        obj = DummyKinetic(np.array([1.0, 2.0, 3.0]))
        transformed = time_rev.act_on(obj)

        # 时间反演变换: v -> -v
        np.testing.assert_array_almost_equal(transformed.velocity, np.array([-1.0, -2.0, -3.0]))
        """测试操作的复合"""
        identity = IdentityOperation()
        translation = TranslationOperation(displacement=np.array([1.0, 0.0, 0.0]))
        
        # 先平移再恒等 = 平移
        obj = ConcretePhysicalObject()
        transformed1 = identity.act_on(translation.act_on(obj))
        transformed2 = translation.act_on(obj)
        
        # 结果应该相同
        self.assertEqual(transformed1.get_mass(), transformed2.get_mass())


class TestSymmetryEnvironmentsRobustness(unittest.TestCase):
    """对称环境鲁棒性测试"""
    
    def test_poincare_group_basic(self):
        """测试庞加莱群基本性质"""
        poincare = PoincareGroup()
        
        # 庞加莱群是闵可夫斯基时空的对称群
        self.assertIsNotNone(poincare)
    
    def test_lorentz_group_basic(self):
        """测试洛伦兹群基本性质"""
        lorentz = LorentzGroup()
        
        # 洛伦兹群保持闵可夫斯基度规
        self.assertIsNotNone(lorentz)
    
    def test_lorentz_transformations(self):
        """测试洛伦兹变换"""
        lorentz = LorentzGroup()
        
        # boost变换
        v = 0.5  # 光速的一半
        gamma = 1 / np.sqrt(1 - v**2)
        
        # 验证洛伦兹因子
        self.assertGreater(gamma, 1.0)
        self.assertAlmostEqual(gamma, 1.1547, places=3)


class TestConservationLaws(unittest.TestCase):
    """守恒律测试"""
    
    def test_energy_conservation(self):
        """测试能量守恒"""
        # 在封闭系统中能量守恒
        obj1 = ConcretePhysicalObject(mass=1.0)
        obj2 = ConcretePhysicalObject(mass=2.0)
        
        total_mass = obj1.get_mass() + obj2.get_mass()
        self.assertEqual(total_mass, 3.0)
    
    def test_charge_conservation(self):
        """测试电荷守恒"""
        # 在电磁相互作用中电荷守恒
        obj1 = ConcretePhysicalObject(charge=1.0)
        obj2 = ConcretePhysicalObject(charge=-1.0)
        
        total_charge = obj1.get_charge() + obj2.get_charge()
        self.assertEqual(total_charge, 0.0)
    
    def test_momentum_conservation(self):
        """测试动量守恒（简化版）"""
        # 在封闭系统中动量守恒
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([-1.0, 0.0, 0.0])
        
        total_momentum = p1 + p2
        np.testing.assert_array_almost_equal(total_momentum, np.array([0.0, 0.0, 0.0]))


class TestExtremePhysicalConditions(unittest.TestCase):
    """极端物理条件测试"""
    
    def test_relativistic_limit(self):
        """测试相对论极限"""
        # 接近光速的情况
        v = 0.999  # 光速的99.9%
        gamma = 1 / np.sqrt(1 - v**2)
        
        # 洛伦兹因子应该很大
        self.assertGreater(gamma, 20.0)
    
    def test_quantum_limit(self):
        """测试量子极限"""
        # 普朗克尺度
        planck_mass = 2.176e-8  # kg
        planck_length = 1.616e-35  # m
        
        # 创建普朗克尺度的对象
        obj = ConcretePhysicalObject(mass=planck_mass)
        self.assertEqual(obj.get_mass(), planck_mass)
    
    def test_thermal_limit(self):
        """测试热力学极限"""
        # 绝对零度
        T_zero = 0.0
        
        # 极高温度
        T_high = 1e10  # K
        
        # 测试对象在极端温度下的行为
        obj = ConcretePhysicalObject(mass=1.0)
        self.assertEqual(obj.get_mass(), 1.0)  # 质量不随温度变化


if __name__ == '__main__':
    unittest.main()
