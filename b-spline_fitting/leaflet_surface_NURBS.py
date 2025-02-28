# Script to use the boundaries created in leaflet_boundaries.py to reconstruct
# the leaflet based on a NURBS surface. Consequently, export the file and view 
# surface to a vtk file to view it in paraview.

from geomdl import BSpline
import numpy as np
import pyvista as pv
import functions

import os 

os.chdir("H:/DATA/Afstuderen/2. Code/Stenosis-Severity/b-spline_fitting")

# Define control points based on user's input
commissure_1 = [0, 0, 1]  # Commissure 1
commissure_2 = [2, 0, 1]  # Commissure 2
hinge_point = [1, 0, 0]  # Hinge point
leaflet_tip = [(commissure_1[0] + commissure_2[0]) / 2, 1, (commissure_1[2] + hinge_point[2]) / 3]

# Calculate the control poitns for the surface reconstructions
control_points = functions.calc_boundary_ctrlpts(commissure_1, commissure_2, leaflet_tip, hinge_point)

# Define the NURBS surface
surf = BSpline.Surface()
surf.degree_u = 2
surf.degree_v = 2
surf.ctrlpts2d = control_points  # Set control points

# Define knot vectors
surf.knotvector_u = [0, 0, 0, 1, 1, 1]
surf.knotvector_v = [0, 0, 0, 1, 1, 1]

# Set resolution
surf.delta = 0.05  # Resolution

surf.evaluate()

functions.export_vtk(surf, "H:/DATA/Afstuderen/2. Code/Stenosis-Severity/reconstructions/leaflet_surface.vtk")

