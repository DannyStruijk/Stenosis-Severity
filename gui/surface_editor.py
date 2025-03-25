import os 
import numpy as np
import pyvista as pv


# Set working directory
os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/b-spline_fitting")

import functions  # Custom functions

# Load initial data
landmarks_file = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\gui\commissures.txt"
landmarks = np.loadtxt(landmarks_file)

# Assign commissures and leaflet tip from file
commissure_1, commissure_2, commissure_3, leaflet_tip, hinge_1, hinge_2, hinge_3 = landmarks
cusp_landmarks = functions.calc_leaflet_landmarks(commissure_1, commissure_2, commissure_3, hinge_1, hinge_2, hinge_3)

# First cusp control points
leaf_1_ctrl = functions.calc_additional_ctrlpoints(cusp_landmarks[0], leaflet_tip)
leaf_1_ctrl_flat = np.array(leaf_1_ctrl).reshape(-1, 3)  # Flatten to Nx3 array

# Initialize variables
selected_point_idx = None
click_count = 0

# Create PyVista plotter
plotter = pv.Plotter()

mesh_valve = pv.read(r"H:\DATA\Afstuderen\3.Data\Harde Schijf Koen\AoS Stress\Afstudeerproject Koen Janssens\aos14\Mimics\aos_14.stl")
plotter.add_mesh(mesh_valve, color="lightgray", opacity=0.7)


# Function to reconstruct and update visualization
def reconstruct_and_update():
    """ Updates the visualization dynamically without restarting PyVista. """
    global leaf_1_ctrl, leaf_1_ctrl_flat

    # Reconstruct surface using the updated control points
    interpolated_leaf_1_ctrl = functions.interpolate_surface(leaf_1_ctrl)
    functions.export_vtk(interpolated_leaf_1_ctrl, "H:/DATA/Afstuderen/2.Code/temporary/test_mesh.vtk")

    # Load updated mesh
    mesh = pv.read("H:/DATA/Afstuderen/2.Code/temporary/test_mesh.vtk")

    # Clear previous elements but keep window open
    plotter.clear()
    plotter.add_mesh(mesh, color="lightgray", opacity=0.7)

    # Re-add control points
    plotter.add_points(leaf_1_ctrl_flat, color="red", point_size=10, render_points_as_spheres=True)

    # Re-enable point picking
    plotter.enable_surface_point_picking(on_click_callback, show_message=True, pickable_window=True)

    plotter.render()  # Redraw window without closing it

# Mouse click callback function
def on_click_callback(click_pos):
    """ Handles user interaction for moving control points. """
    global selected_point_idx, click_count, leaf_1_ctrl, leaf_1_ctrl_flat

    if click_count == 0:
        # Find the closest control point to the clicked position
        distances = np.linalg.norm(leaf_1_ctrl_flat - click_pos, axis=1)
        selected_point_idx = np.argmin(distances)
        print(f"Selected control point {selected_point_idx} at {leaf_1_ctrl_flat[selected_point_idx]}")

        click_count = 1  # Wait for the second click

    elif click_count == 1:
        # Move the selected control point
        print(f"Moving control point {selected_point_idx} to {click_pos}")

        # Update both the flat array and the original structured list
        leaf_1_ctrl_flat[selected_point_idx] = click_pos  

        # Update the original leaf_1_ctrl structure
        num_rows, num_cols, _ = np.shape(leaf_1_ctrl)
        row_idx, col_idx = divmod(selected_point_idx, num_cols)
        leaf_1_ctrl[row_idx][col_idx] = click_pos  # Update original structure

        # Reset selection
        selected_point_idx = None
        click_count = 0

        # Update visualization dynamically
        reconstruct_and_update()

# Initial reconstruction and display
reconstruct_and_update()
plotter.show()
