# 可视化教程

本教程介绍如何使用 PySymmetry 进行科学可视化。

## 目录

1. [基础图形](#基础图形)
2. [量子态可视化](#量子态可视化)
3. [能级图](#能级图)
4. [动画制作](#动画制作)

---

## 基础图形

### 设置

```python
import matplotlib.pyplot as plt
from PySymmetry.visual.quantum_visual import QuantumVisualizer
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建可视化器
viz = QuantumVisualizer()
```

---

## 量子态可视化

### Bloch 球

```python
from PySymmetry.visual.quantum_visual import QuantumVisualizer
from PySymmetry.phys.quantum.states import Ket
import numpy as np

viz = QuantumVisualizer()

# |+⟩ 态
plus = (Ket('0') + Ket('1')) / np.sqrt(2)
fig = viz.bloch_sphere([plus], labels=['|+⟩'])
fig.savefig('bloch_plus.png', dpi=150)
plt.close()

# 多个态
states = [
    Ket('0'),
    (Ket('0') + Ket('1')) / np.sqrt(2),
    (Ket('0') + 1j*Ket('1')) / np.sqrt(2),
]
fig = viz.bloch_sphere(states, labels=['|0⟩', '|+⟩', '|+ᵢ⟩'])
```

### 密度矩阵热图

```python
from PySymmetry.visual.quantum_visual import QuantumVisualizer
from PySymmetry.phys.quantum.states import DensityMatrix
import numpy as np

viz = QuantumVisualizer()

# 创建密度矩阵
rho = DensityMatrix((Ket('0') + Ket('1')) / np.sqrt(2))
fig = viz.density_matrix_heatmap(rho.matrix, title='|+⟩ 态的密度矩阵')
fig.savefig('density_matrix.png', dpi=150)
```

### 概率分布

```python
# 1D 概率分布
probs = np.array([0.5, 0.5])
fig = viz.probability_distribution(probs, labels=['|0⟩', '|1⟩'])
fig.savefig('prob_1d.png')

# 2D 概率分布（量子行走等）
x = np.linspace(-5, 5, 100)
y = np.exp(-x**2)
fig = viz.probability_2d(x, y)
```

---

## 能级图

### 氢原子能级

```python
from PySymmetry.visual.hydrogen_visual import HydrogenVisualizer

viz = HydrogenVisualizer()

# 能量本征值
n_levels = [1, 2, 3, 4, 5]
energies = [-1/(2*n**2) for n in n_levels]

fig = viz.energy_diagram(n_levels, energies, 
                         labels=['1s', '2s/2p', '3s/3p/3d', '4s/4p/4d/4f', '5s/...'])
fig.savefig('hydrogen_levels.png')
```

### 光谱图

```python
# 吸收/发射光谱
wavelengths = np.linspace(400, 700, 1000)  # nm
spectrum = np.exp(-(wavelengths - 500)**2 / 100)

fig = viz.spectrum(wavelengths, spectrum, 
                   title='发射光谱', 
                   xlabel='波长 (nm)',
                   ylabel='强度')
fig.savefig('spectrum.png')
```

---

## 动画制作

### 量子态演化动画

```python
from PySymmetry.visual.animation import QuantumAnimator
import numpy as np

animator = QuantumAnimator()

# 创建动画：Bloch 球上的旋转
def state_generator(t):
    theta = t * np.pi / 10
    return (Ket('0') * np.cos(theta/2) + 
            Ket('1') * np.sin(theta/2) * np.exp(1j*t))

anim = animator.bloch_rotation(state_generator, 
                               frames=100, 
                               interval=50)
anim.save('rotation.mp4', writer='ffmpeg')
```

### 时间演化动画

```python
# 波函数时间演化
x = np.linspace(-10, 10, 200)
t_values = np.linspace(0, 10, 50)

def wavefunction(x, t):
    return np.exp(-x**2/4) * np.cos(x - t)

fig = animator.wavefunction_evolution(x, t_values, wavefunction,
                                      title='波包传播')
fig.save('wavepacket.gif', writer='pillow')
```

### 纠缠态演化

```python
# 纠缠熵随时间变化
times = np.linspace(0, 10, 100)
entropies = 1 - np.exp(-times/2)  # 示例数据

fig = animator.entanglement_evolution(times, entropies,
                                      ylabel='纠缠熵 S')
fig.savefig('entanglement.png')
```

---

## 图形导出

### 导出格式

```python
fig = viz.bloch_sphere([Ket('0')])

# PNG (高分辨率)
fig.savefig('bloch.png', dpi=300, bbox_inches='tight')

# PDF (矢量图)
fig.savefig('bloch.pdf', bbox_inches='tight')

# SVG (Web 可用)
fig.savefig('bloch.svg')
```

### 子图布局

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 左上：Bloch 球
ax1 = axes[0, 0]
viz.plot_bloch_on_axis(ax1, Ket('0'))

# 右上：概率分布
ax2 = axes[0, 1]
viz.plot_probability(ax2, [0.5, 0.5])

# 左下：密度矩阵
ax3 = axes[1, 0]
viz.plot_density_matrix(ax3, rho.matrix)

# 右下：能级图
ax4 = axes[1, 1]
viz.plot_energy_levels(ax4, [1, 2, 3], [-0.5, -0.125, -0.055])

plt.tight_layout()
fig.savefig('comprehensive.png', dpi=150)
```

---

## 样式自定义

```python
# 全局样式
plt.style.use('seaborn-v0_8-darkgrid')

# 颜色方案
viz.set_color_scheme('viridis')  # 或 'plasma', 'inferno', 'coolwarm'

# 字体大小
viz.set_fontsize(title=16, labels=12, ticks=10)

# 线宽
viz.set_linewidth(curve=2, errorbar=1.5)
```
