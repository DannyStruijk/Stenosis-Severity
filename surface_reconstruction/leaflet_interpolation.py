import os 
import functions
import numpy as np

# Read landmarks from file, which were annotated using the GUI
landmarks_file = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\annotations\commissures.txt"
landmarks = np.loadtxt(landmarks_file)

# Assign commissures and leaflet tip from file
commissure_1, commissure_2, commissure_3, center, hinge_1, hinge_2, hinge_3 = landmarks
cusp_landmarks = functions.calc_leaflet_landmarks(commissure_1, commissure_2, commissure_3, hinge_1, hinge_2, hinge_3)

# %% RECONSTRUCTION WTIH ADDITIONAL CONTROL POINTS

# Firs the control points for the interpolation are calculated. Note that several control points
# are calculated on the basis of the other control points.
# Then, each cusp is reconstructed on the basis of control point interpolation.
# THe cusp is then individually exported as VTK file. 

# Define reconstruction save path
recon_path = "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/"

# First cusp
leaf_1 = functions.calc_ctrlpoints(cusp_landmarks[0], center)
interpolated_leaf_1 = functions.interpolate_surface(leaf_1)
functions.save_surface_evalpts(interpolated_leaf_1, recon_path + "leaflet_1_points.txt") # Save the points seperately
functions.export_vtk(interpolated_leaf_1, recon_path + "reconstructed_leaflet_1.vtk") # Save as VTK

# Second cusp
leaf_2 = functions.calc_ctrlpoints(cusp_landmarks[1], center)
interpolated_leaf_2 = functions.interpolate_surface(leaf_2)
functions.save_surface_evalpts(interpolated_leaf_2, recon_path + "leaflet_2_points.txt") #Save the points individually
functions.export_vtk(interpolated_leaf_2, recon_path + "reconstructed_leaflet_2.vtk") # Save as VTK

# Third cusp
leaf_3 = functions.calc_ctrlpoints(cusp_landmarks[2], center)
interpolated_leaf_3 = functions.interpolate_surface(leaf_3)
functions.save_surface_evalpts(interpolated_leaf_3, recon_path + "leaflet_3_points.txt") # Save the points seperately
functions.export_vtk(interpolated_leaf_3, recon_path + "reconstructed_leaflet_3.vtk") # Save as VTK
