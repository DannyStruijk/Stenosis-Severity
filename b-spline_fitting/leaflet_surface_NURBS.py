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

leaf_1_ctrl = functions.calc_additional_ctrlpoints(cusp_landmarks[0], leaflet_tip)
interpolated_leaf_1_ctrl = functions.interpolate_surface(leaf_1_ctrl)
functions.export_vtk(interpolated_leaf_1_ctrl, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/added_leaflet_surface_1.vtk")



#%% RECONSTRUCTING LEAVES WITH INTEGRATED HINGE POINTS

# Calculate the control points for the surface reconstructions
leaf_1 = functions.calc_surface_ctrlpts_hinge(cusp_landmarks[0], leaflet_tip)
leaf_2 = functions.calc_surface_ctrlpts_hinge(cusp_landmarks[1], leaflet_tip)
leaf_3 = functions.calc_surface_ctrlpts_hinge(cusp_landmarks[2], leaflet_tip)

# Calculate the control poitns for the INTERPOLATED surface reconstructions where the HINGE is known!!
interpolated_leaf_1 = functions.interpolate_surface(leaf_1)
interpolated_leaf_2 = functions.interpolate_surface(leaf_2)
interpolated_leaf_3 = functions.interpolate_surface(leaf_3)

# Define the knot vectors for the surface
knotvector_u = [0, 0, 0, 0.33, 0.66, 1, 1, 1]  # 5 control points → 8 knots for degree 2
knotvector_v = [0, 0, 0, 1, 1, 1]  # Still 3 control points → remains the same

# # Reconstruct the surface using the defined control points, parameters are currently hard-coded
leaflet_1_hinge = functions.reconstruct_surface(leaf_1, knotvector_u = knotvector_u, knotvector_v = knotvector_v)
leaflet_2_hinge = functions.reconstruct_surface(leaf_2, knotvector_u = knotvector_u, knotvector_v = knotvector_v)
leaflet_3_hinge = functions.reconstruct_surface(leaf_3, knotvector_u = knotvector_u, knotvector_v = knotvector_v)

# # Save the surface as a VTK file
functions.export_vtk(leaflet_1_hinge, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_1_hinge.vtk")
functions.export_vtk(leaflet_2_hinge, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_2_hinge.vtk")
functions.export_vtk(leaflet_3_hinge, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_3_hinge.vtk")

functions.export_vtk(interpolated_leaf_1, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_1_interpolated.vtk")
functions.export_vtk(interpolated_leaf_2, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_2_interpolated.vtk")
functions.export_vtk(interpolated_leaf_3, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_3_interpolated.vtk")

#%% RECONSTRUCTING THE LEAVES WITH 5x3 GRID OF CONTROL POINTS
