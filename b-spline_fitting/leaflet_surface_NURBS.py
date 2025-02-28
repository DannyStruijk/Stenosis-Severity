# Script to use the boundaries created in leaflet_boundaries.py to reconstruct
# the leaflet based on a NURBS surface. Consequently, export the file and view 
# surface to a vtk file to view it in paraview.

import numpy as np
import functions
import os 
from geomdl.visualization import VisMPL

# Change working directory in order to import functions from other file
os.chdir("H:/DATA/Afstuderen/2. Code/Stenosis-Severity/b-spline_fitting")

# Define control points based on user's input
commissure_1 = [0, 0, 1]  # Commissure 1
commissure_2 = [2, 0, 1]  # Commissure 2
hinge_point = [1, 0, 0]  # Hinge point
leaflet_tip = [(commissure_1[0] + commissure_2[0]) / 2, 1, (commissure_1[2] + hinge_point[2]) / 3]

# Calculate the control poitns for the surface reconstructions
control_points = functions.calc_surface_ctrlpts(commissure_1, commissure_2, leaflet_tip, hinge_point)

# Define the knot vectors for the surface
knotvector_u = [0, 0.01, 0.4, 0.6, 0.8, 1]
knotvector_v = [0, 0.01, 0.4, 0.6, 0.8, 1]

#knotvector_u = [0, 0.25, 0.5, 0.75, 0.9, 1]
#knotvector_v = [0, 0.25, 0.5, 0.75, 0.9, 1]

# Reconstruct the surface using the defined control points, parameters are currently hard-coded
surf = functions.reconstruct_surface(control_points, knotvector_u = knotvector_u, knotvector_v = knotvector_v)

# Save the surface as a VTK file
functions.export_vtk(surf, "H:/DATA/Afstuderen/2. Code/Stenosis-Severity/reconstructions/leaflet_surface.vtk")
