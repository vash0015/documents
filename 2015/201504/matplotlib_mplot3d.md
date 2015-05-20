mplot3d tutorial

[Top Page](http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html)


# Getting started

An Axes3D object is created just like any other axes using the projection=‘3d’ keyword. Create a new matplotlib.figure.Figure and add a new axes to it of type Axes3D:

```py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
```

New in version 1.0.0: This approach is the preferred method of (方式) creating a 3D axes ().

** Note: ** Prior (～より前の) to version 1.0.0, the method of creating a 3D axes was different. For those using older versions of matplotlib, change `ax = fig.add_subplot(111, projection='3d')` to `ax = Axes3D(fig)`.


# Line Plots

```py
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# 描画領域定義
# gca : Get the current axes, creating one if necessary.
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')

# 入力値準備
# θとzの値の行列を作り、そこからx,yの行列を計算する
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

# 描画実施
ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()
```

![](http://matplotlib.org/_images/lines3d_demo1.png)


# Scatter plots

```py
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 指定範囲内で乱数値を生成する
def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

# 乱数値を生成し順にグラフに設定する
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    # (c, m, zl, zh)の組が2つ
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zl, zh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
```

![](http://matplotlib.org/_images/scatter3d_demo1.png)


# Wireframe plots

```py
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# 描画処理
# add_subplot(colrownum) → col列row行num番目のsubplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()
```

![](http://matplotlib.org/_images/wire3d_demo1.png)


# Surface plots

```py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# 0.25刻みの数値行列を作る
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)

# Z軸の計算用にXとYの組み合わせ表を作る
X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False
)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
```

![](http://matplotlib.org/_images/surface3d_demo1.png)


# Tri-Surface plots

```py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

n_angles = 36
n_radii = 8

# An array of radii
# Does not include radius r=0, this is to eliminate duplicate points
radii = np.linspace(0.125, 1.0, n_radii)

# An array of angles
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

# Repeat all angles for each radius
angles = np.repeat(angles[...,np.newaxis], n_radii, axis=1)

# Convert polar (radii, angles) coords to cartesian (x, y) coords
# (0, 0) is added here. There are no duplicate points in the (x, y) plane
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())

# Pringle surface
z = np.sin(-x*y)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)

plt.show()
```

![](http://matplotlib.org/_images/trisurf3d_demo1.png)


# Contour plots


























a
