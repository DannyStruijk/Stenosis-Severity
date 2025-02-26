" Code om te testen hoe b-splines werken."
" Daarnaast probeer ik de reconstructie van een leaflet te maken dmv Nadir point, commisuren en hinge points" 
" Op basis van zelfgekozen punten. Dit resultaat wordt vergeleken met de Philips segmentatie. "


import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import make_interp_spline
from mpl_toolkits.mplot3d import Axes3D, art3d

# Handmatig punten instellen in 3D
points = np.array([[0, 0, 1], [1.5, 3, 3], [3, 0, 1]])

#Nadir punt 
nadir = [1.5, 1.5, 0]

# Handmatige parameterwaarden t instellen
t = [0, 0.5, 1]

# B-spline fitten met make_interp_spline
spl = make_interp_spline(t, points, k=2)

# Fijne parameterwaarden voor de curve
u_fine = np.linspace(0, 1, 100)
x_fine, y_fine, z_fine = spl(u_fine).T

# Defineren van de lijn tussen de eindpunten van de curven
x_close = [x_fine[0], x_fine[-1]]
y_close = [y_fine[0], y_fine[-1]]
z_close = [z_fine[0], z_fine[-1]]

# 3D-visualisatie
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[:, 0], points[:, 1], points[:, 2], 'ro', label='Handmatige punten')
ax.plot(x_fine, y_fine, z_fine, 'b-', label='B-spline fit')
ax.plot(x_close, y_close, z_close, 'b-', label='Sluitlijn')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D B-spline Fit met Handmatige Knopen en t')
plt.show()

