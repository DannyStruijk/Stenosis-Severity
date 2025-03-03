" Code om te testen hoe b-splines werken."
" Daarnaast probeer ik de reconstructie van een leaflet te maken dmv Nadir point, commisuren en hinge points" 
" Op basis van zelfgekozen punten. Dit resultaat wordt vergeleken met de Philips segmentatie. "


# Change working directory in order to import functions from other file
os.chdir("H:/DATA/Afstuderen/2. Code/Stenosis-Severity/b-spline_fitting")

import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import make_interp_spline
from mpl_toolkits.mplot3d import Axes3D, art3d
import functions

# User input: selecting the necessary points to reconstruct the leaflet
# Commisuren & hinge point
point_1 = [0,0,1]
point_2 = [2,0,1]
hinge_point = [1,0,0]

# Automatische leaflet_tip herkenning
leaflet_tip=[(point_1[0]+point_2[0])/2,  1 , (point_1[2]+hinge_point[2])/3]

# Arch punten om te helpen met lijn vormen tussen tip en commisuren
arch_control_1=[(leaflet_tip[0]+point_1[0])/2-0.5, (leaflet_tip[1]+hinge_point[1])/2+0.5, (leaflet_tip[2]+hinge_point[2])/2]
arch_control_2=[(leaflet_tip[0]+point_2[0])/2+0.5, (leaflet_tip[1]+hinge_point[1])/2+0.5, (leaflet_tip[2]+hinge_point[2])/2]

# Handmatig punten instellen in 3D
points = np.array([point_1, hinge_point, point_2])

# Punten voor de arches creëeren
arch_spline_1=[point_1, arch_control_1, leaflet_tip]
arch_spline_2=[point_2, arch_control_2, leaflet_tip]

# Handmatige parameterwaarden t instellen
t = [0, 0.5, 1]

# B-spline fitten met make_interp_spline
spl = make_interp_spline(t, points, k=2)
spl_arch_1 = make_interp_spline(t, arch_spline_1, k=2)
spl_arch_2 = make_interp_spline(t, arch_spline_2, k=2)

# Fijne parameterwaarden voor de curve
u_fine = np.linspace(0, 1, 100)
x_fine, y_fine, z_fine = spl(u_fine).T
arch_1_x, arch_1_y, arch_1_z = spl_arch_1(u_fine).T
arch_2_x, arch_2_y, arch_2_z = spl_arch_2(u_fine).T

# 3D-visualisatie
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[:, 0], points[:, 1], points[:, 2], 'ro', label='Handmatige punten')
ax.plot(x_fine, y_fine, z_fine, 'b-')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D B-spline Fit met Handmatige Knopen en t')

# Plot the arch splines
ax.plot(arch_1_x, arch_1_y, arch_1_z, 'b-')
ax.plot(arch_2_x, arch_2_y, arch_2_z, 'b-')

#Plot leaflet tip
ax.scatter(*leaflet_tip, color='g', s=100, label='Leaflet tip')  # Groen punt voor leaflet tip
ax.legend()

# Plot arch points
ax.scatter(*arch_control_1, color='r', s=50, label="Arch point")
ax.scatter(*arch_control_2, color='r', s=50, label="Arch point")

# Plot annulus midpoints
center = [leaflet_tip[0], leaflet_tip[1], (point_1[2]+point_2[2])/2]
annulus_midpoint = functions.midpoint_on_annulus(point_1, point_2, center)
ax.scatter(*annulus_midpoint, color= 'b', s=50, label="Annulus Midpoint")

plt.show()

