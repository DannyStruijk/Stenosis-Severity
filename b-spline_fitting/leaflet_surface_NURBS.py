# Script to use the boundaries created in leaflet_boundaries.py to reconstruct
# the leaflet based on a NURBS surface. Consequently, export the file and view 
# surface to a vtk file to view it in paraview.

from geomdl import fitting, BSpline
import numpy as np

# Define control points based on user's input
point_1 = [0, 0, 1]  # Commissure 1
point_2 = [2, 0, 1]  # Commissure 2
hinge_point = [1, 0, 0]  # Hinge point
leaflet_tip = [(point_1[0] + point_2[0]) / 2, 1, (point_1[2] + hinge_point[2]) / 3]
arch_control_1 = [(leaflet_tip[0] + point_1[0]) / 2, (leaflet_tip[1] + hinge_point[1]) / 2, (leaflet_tip[2] + hinge_point[2]) / 1.5]
arch_control_2 = [(leaflet_tip[0] + point_2[0]) / 2, (leaflet_tip[1] + hinge_point[1]) / 2, (leaflet_tip[2] + hinge_point[2]) / 1.5]

# Create boundary curves
curve_1 = [point_1, arch_control_1, leaflet_tip]
curve_2 = [leaflet_tip, arch_control_2, point_2]
curve_3 = [point_1, hinge_point, point_2]  # curve from point_1 to point_2 via hinge_point

# Define the control grid (3x3 control points)
control_points = np.array([
    [point_1, arch_control_1, leaflet_tip],
    [hinge_point, hinge_point, hinge_point],
    [point_2, arch_control_2, leaflet_tip]
])

# Set degree to 3 (cubic)
degree_u = 3
degree_v = 3

# Set the size of the surface in u and v directions
size_u = 2
size_v = 2
 
# Fit NURBS surface through the control points
surf = fitting.interpolate_surface(control_points, size_u, size_v, degree_u, degree_v)

# Set evaluation delta for resolution
surf.delta = 0.05
