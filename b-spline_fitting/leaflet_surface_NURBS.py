# Script to use the boundaries created in leaflet_boundaries.py to reconstruct
# the leaflet based on a BSpline surface. Consequently, export the file and view 
# surface to a vtk file to view it in paraview.
import os 
# Change working directory in order to import functions from other file
os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/b-spline_fitting")

# Import modules
import functions
import numpy as np

# Read landmarks from file, which were annotated using the GUI
# Landmarks file = landmarks from CT images
# Commissures file = landmarks from STL object
landmarks_file = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\gui\commissures.txt"
#landmarks_file = r"H:/DATA/Afstuderen/2.Code/Stenosis-Severity/b-spline_fitting/landmarks.txt"
landmarks = np.loadtxt(landmarks_file)

# Assign commissures and leaflet tip from file
commissure_1, commissure_2, commissure_3, leaflet_tip, hinge_1, hinge_2, hinge_3 = landmarks
cusp_landmarks = functions.calc_leaflet_landmarks(commissure_1, commissure_2, commissure_3, hinge_1, hinge_2, hinge_3)

# %% RECONSTRUCTION WTIH ADDITIONAL CONTROL POINTS

# First cusp
leaf_1_ctrl = functions.calc_additional_ctrlpoints(cusp_landmarks[0], leaflet_tip)
interpolated_leaf_1_ctrl = functions.interpolate_surface(leaf_1_ctrl)
functions.export_vtk(interpolated_leaf_1_ctrl, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/added_leaflet_surface_1.vtk")

# Second cusp
leaf_2_ctrl = functions.calc_additional_ctrlpoints(cusp_landmarks[1], leaflet_tip)
interpolated_leaf_2_ctrl = functions.interpolate_surface(leaf_2_ctrl)
functions.export_vtk(interpolated_leaf_2_ctrl, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/added_leaflet_surface_2.vtk")

# Third cusp
leaf_3_ctrl = functions.calc_additional_ctrlpoints(cusp_landmarks[2], leaflet_tip)
interpolated_leaf_3_ctrl = functions.interpolate_surface(leaf_3_ctrl)
functions.export_vtk(interpolated_leaf_3_ctrl, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/added_leaflet_surface_3.vtk")


#%% RECONSTRUCTING LEAVES WITH INTEGRATED HINGE POINTS

# Calculate the control points for the surface reconstructions
leaf_1 = functions.calc_surface_ctrlpts_hinge(cusp_landmarks[0], leaflet_tip)
leaf_2 = functions.calc_surface_ctrlpts_hinge(cusp_landmarks[1], leaflet_tip)
leaf_3 = functions.calc_surface_ctrlpts_hinge(cusp_landmarks[2], leaflet_tip)

# Calculate the control poitns for the INTERPOLATED surface reconstructions where the HINGE is known!!
interpolated_leaf_1 = functions.interpolate_surface(leaf_1)
interpolated_leaf_2 = functions.interpolate_surface(leaf_2)
interpolated_leaf_3 = functions.interpolate_surface(leaf_3)

functions.export_vtk(interpolated_leaf_1, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_1_interpolated.vtk")
functions.export_vtk(interpolated_leaf_2, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_2_interpolated.vtk")
functions.export_vtk(interpolated_leaf_3, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_3_interpolated.vtk")

#%% RECONSTRUCTING THE LEAVES WITH 5x3 GRID OF CONTROL POINTS
