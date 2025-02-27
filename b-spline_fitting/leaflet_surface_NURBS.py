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

# Define the control grid (3x3 control points)
control_points = ([
    [point_1, arch_control_1, leaflet_tip],
    [hinge_point, hinge_point, hinge_point],
    [point_2, arch_control_2, leaflet_tip]
])

# Define surface
surf = BSpline.Surface()
surf.degree_u = 2
surf.degree_v = 2
surf.ctrlpts2d = control_points # Set control points

# Define knot vectors
surf.knotvector_u = [0, 0, 0, 1, 1, 1]
surf.knotvector_v = [0, 0, 0, 1, 1, 1]

# Set resolution
surf.delta = 0.05 # Resolution

# Fit NURBS surface through the control points
surf.evaluate()

