import os 
os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/b-spline_fitting")

import numpy as np
import pyvista as pv
import functions

# Load data
landmarks_file = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\gui\commissures.txt"
landmarks = np.loadtxt(landmarks_file)

# Assign commissures and leaflet tip from file
commissure_1, commissure_2, commissure_3, leaflet_tip, hinge_1, hinge_2, hinge_3 = landmarks
cusp_landmarks = functions.calc_leaflet_landmarks(commissure_1, commissure_2, commissure_3, hinge_1, hinge_2, hinge_3)

# First cusp
leaf_1_ctrl = functions.calc_additional_ctrlpoints(cusp_landmarks[0], leaflet_tip)
interpolated_leaf_1_ctrl = functions.interpolate_surface(leaf_1_ctrl)
functions.export_vtk(interpolated_leaf_1_ctrl, "H:/DATA/Afstuderen/2.Code/temporary/test_mesh.vtk")

# Load mesh
mesh = pv.read(r"H:/DATA/Afstuderen/2.Code/temporary/test_mesh.vtk")

# Create plotter
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightgray", opacity=0.7)

# Flatten the control points (if leaf_1_ctrl is a list of numpy arrays, flatten it)
leaf_1_ctrl_flat = np.array(leaf_1_ctrl).reshape(-1, 3)  # Flattening into a 2D array (Nx3)

# Plot the control points
control_points_actor = plotter.add_points(leaf_1_ctrl_flat, color="red", point_size=10, render_points_as_spheres=True)

# Initialize variables to track the state
selected_point_idx = None  # This will store the index of the selected control point
click_count = 0  # This will keep track of the number of clicks

# Define the callback function for click events
def on_click_callback(click_pos):
    global selected_point_idx, click_count, leaf_1_ctrl_flat, control_points_actor

    if click_count == 0:
        # First click: find the closest point in the control points
        distances = np.linalg.norm(leaf_1_ctrl_flat - click_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        # Highlight the selected point (could change color, etc.)
        print(f"First click - closest control point index: {closest_idx}, coordinates: {leaf_1_ctrl_flat[closest_idx]}")
        
        selected_point_idx = closest_idx  # Store the index of the selected control point
        click_count += 1  # Increment click count
        
    elif click_count == 1:
        # Second click: update the selected point with the new clicked position
        print(f"Second click - new coordinates: {click_pos}")

        # Update the selected control point with the new clicked coordinates
        leaf_1_ctrl_flat[selected_point_idx] = click_pos  # Modify the control point

        # Update the points in the plotter without clearing the whole plot
        control_points_actor.points = leaf_1_ctrl_flat  # Update the points' coordinates

        print(f"Updated control point {selected_point_idx} to new coordinates: {click_pos}")

        # Reset for next interaction
        selected_point_idx = None
        click_count = 0

# Enable point picking with the callback function
plotter.enable_surface_point_picking(on_click_callback, show_message=True)

# Show the plot
plotter.show()
