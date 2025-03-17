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


# %% SURFACE CALCULATION & RECONSTRUCTION

# Calculate the control poitns for the surface reconstructions
control_points_1 = functions.calc_surface_ctrlpts(commissure_1, commissure_2, leaflet_tip)
control_points_2 = functions.calc_surface_ctrlpts(commissure_1, commissure_3, leaflet_tip)
control_points_3 = functions.calc_surface_ctrlpts(commissure_2, commissure_3, leaflet_tip)

# Define the knot vectors for the surface
knotvector_u = [0, 0, 0,1,1,1]
knotvector_v = [0, 0, 0, 1, 1, 1]

print("\nCalculating the leaflet surfaces... \n")

# # Reconstruct the surface using the defined control points, parameters are currently hard-coded
leaflet_1 = functions.reconstruct_surface(control_points_1, knotvector_u = knotvector_u, knotvector_v = knotvector_v)
leaflet_2 = functions.reconstruct_surface(control_points_2, knotvector_u = knotvector_u, knotvector_v = knotvector_v)
leaflet_3 = functions.reconstruct_surface(control_points_3, knotvector_u = knotvector_u, knotvector_v = knotvector_v)

# # Save the surface as a VTK file
functions.export_vtk(leaflet_1, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_1.vtk")
functions.export_vtk(leaflet_2, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_2.vtk")
functions.export_vtk(leaflet_3, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_3.vtk")

#%% RECONSTRUCTING THE LEAVES WITH KNOWN HINGE POINTS

# Calculate the control points for the surface reconstructions
leaf_1 = functions.calc_surface_ctrlpts_hinge(commissure_1, commissure_2, leaflet_tip, hinge_1)
leaf_2 = functions.calc_surface_ctrlpts_hinge(commissure_1, commissure_3, leaflet_tip, hinge_2)
leaf_3 = functions.calc_surface_ctrlpts_hinge(commissure_2, commissure_3, leaflet_tip, hinge_3)

# Define the knot vectors for the surface
knotvector_u = [0, 0, 0,1,1,1]
knotvector_v = [0, 0, 0, 1, 1, 1]

# # Reconstruct the surface using the defined control points, parameters are currently hard-coded
leaflet_1_hinge = functions.reconstruct_surface(control_points_1, knotvector_u = knotvector_u, knotvector_v = knotvector_v)
leaflet_2_hinge = functions.reconstruct_surface(control_points_2, knotvector_u = knotvector_u, knotvector_v = knotvector_v)
leaflet_3_hinge = functions.reconstruct_surface(control_points_3, knotvector_u = knotvector_u, knotvector_v = knotvector_v)

# # Save the surface as a VTK file
functions.export_vtk(leaflet_1, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_1_hinge.vtk")
functions.export_vtk(leaflet_2, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_2_hinge.vtk")
functions.export_vtk(leaflet_3, "H:/DATA/Afstuderen/2.Code/Stenosis-Severity/reconstructions/leaflet_surface_3_hinge.vtk")

#%% RECONSTRUCTING LEAFLET WALLS
 
# # Reconstruct the leaflet wall
# control_points_wall_1 = functions.calc_wall_ctrlpts(commissure_1 = commissure_1, commissure_2 = commissure_2, leaflet_tip = leaflet_tip)
# control_points_wall_2 = functions.calc_wall_ctrlpts(commissure_2, commissure_3, leaflet_tip)
# control_points_wall_3 = functions.calc_wall_ctrlpts(commissure_1, commissure_3, leaflet_tip)

# print("\nCalculating the leaflets wall... \n")

# # Reconstruct the surface for each wall
# leaflet_wall_1 = functions.reconstruct_surface(control_points_wall_1, knotvector_u=knotvector_u, knotvector_v=knotvector_v)
# leaflet_wall_2 = functions.reconstruct_surface(control_points_wall_2, knotvector_u=knotvector_u, knotvector_v=knotvector_v)
# leaflet_wall_3 = functions.reconstruct_surface(control_points_wall_3, knotvector_u=knotvector_u, knotvector_v=knotvector_v)

# # Save the surfaces as VTK files
# functions.export_vtk(leaflet_wall_1, "H:/DATA/Afstuderen/2. Code/Stenosis-Severity/reconstructions/leaflet_wall_1.vtk")
# functions.export_vtk(leaflet_wall_2, "H:/DATA/Afstuderen/2. Code/Stenosis-Severity/reconstructions/leaflet_wall_2.vtk")
# functions.export_vtk(leaflet_wall_3, "H:/DATA/Afstuderen/2. Code/Stenosis-Severity/reconstructions/leaflet_wall_3.vtk")

