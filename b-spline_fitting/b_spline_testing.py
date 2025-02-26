# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:43:00 2025

@author: Danny Struijk

This file is to play with control points and fitting b-splines. No further purposes, 
besides learning to work with b-splines and visualizing them.
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import make_interp_spline

# Handmatig punten instellen in 3D
points = np.array([[0, 0, 0], [1.5, 3, 3], [3, 0, 0]])

# Handmatige parameterwaarden t instellen
t = [0, 0.6, 1]

# B-spline fitten met make_interp_spline
spl = make_interp_spline(t, points, k=2)

# Fijne parameterwaarden voor de curve
u_fine = np.linspace(0, 1, 100)
x_fine, y_fine, z_fine = spl(u_fine).T

# 3D-visualisatie
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[:, 0], points[:, 1], points[:, 2], 'ro', label='Handmatige punten')
ax.plot(x_fine, y_fine, z_fine, 'b-', label='B-spline fit')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D B-spline Fit met Handmatige Knopen en t')
plt.show()
