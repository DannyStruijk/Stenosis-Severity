import sys
import numpy as np
import pyvista as pv
from geomdl import BSpline, fitting
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
import os 

os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/b-spline_fitting")

import functions

# Import leaflet to compare with the fitting B-spline
mesh = pv.read(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\reconstructions\leaflet_surface_1_interpolated.vtk")

# Loading in the annotated data from the STL object
landmarks_file = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\gui\commissures.txt"
landmarks = np.loadtxt(landmarks_file)
commissure_1, commissure_2, commissure_3, leaflet_tip, hinge_1, hinge_2, hinge_3 = landmarks
cusp_landmarks = functions.calc_leaflet_landmarks(commissure_1, commissure_2, commissure_3, hinge_1, hinge_2, hinge_3)

# Step 2: Create the list of points to interpolate through
points = [cusp_landmarks[0][0], cusp_landmarks[0][2], cusp_landmarks[0][1]]

# Step 4: Use the interpolation method to fit the B-spline curve through the points
# The interpolate function creates a B-spline curve that fits the points
curve = fitting.interpolate_curve(points, degree=2)

# Step 6: Evaluate the curve at different parameter values (u values from 0 to 1)
u_vals = np.linspace(0, 1, 100)  # Evaluate at 100 points along the curve
eval_pts = [curve.evaluate_single(u) for u in u_vals]

# Convert eval_pts into a NumPy array for visualization
eval_pts = np.array(eval_pts)

# Extra control points (just an example, you can modify as needed)
extra_ctrlpts = [[eval_pts[25], eval_pts[75]]]  # Specific points on the B-spline curve

# Step 7: Visualize the curve with PyVista
# Create a PyVista point cloud from the evaluated points
point_cloud = pv.PolyData(eval_pts)

# Add the input points (to show them in the visualization as well)
input_points = np.array(points)

# Create a PyVista plotter
plotter = pv.Plotter()

# Add the curve points to the plotter
plotter.add_mesh(point_cloud, color='blue', point_size=10, render_points_as_spheres=True)

# Add the input points to the plotter (with a different color)
input_points_mesh = pv.PolyData(input_points)
plotter.add_mesh(input_points_mesh, color='red', point_size=15, render_points_as_spheres=True)

# Add the extra control points (to show them in the visualization as well)
extra_ctrlpts_mesh = pv.PolyData(np.array(extra_ctrlpts).reshape(-1, 3))  # Reshape for correct format
plotter.add_mesh(extra_ctrlpts_mesh, color='green', point_size=15, render_points_as_spheres=True)

# Set up labels (optional)
plotter.add_point_labels(input_points, ['Commissure 1', 'Hinge 1', 'Commissure 2'], font_size=12)
plotter.add_point_labels(extra_ctrlpts_mesh.points, ['Extra Control Point 1', 'Extra Control Point 2'], font_size=12)

# Also add the leaflet surface to the plotter
#plotter.add_mesh(mesh, color="lightgray", opacity=0.7)

# Show the plot
plotter.show()
