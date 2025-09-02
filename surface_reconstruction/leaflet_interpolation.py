import functions
import numpy as np
import os
from scipy.ndimage import gaussian_filter

# Set patient ID
patient_id = "aos14"

# Read landmarks from file, which were annotated using the GUI
landmarks_file = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\annotations\ras_coordinates.txt"
landmarks = np.loadtxt(landmarks_file)

# The landmarks are loaded from the file. Then, sets of commissures and hinge points are made
commissure_1, commissure_2, commissure_3, center, hinge_1, hinge_2, hinge_3 = landmarks
cusp_landmarks = functions.calc_leaflet_landmarks(commissure_1, commissure_2, commissure_3, hinge_1, hinge_2, hinge_3)

# Save the landmarks with consistent commissure ordering
output_path = fr"H:\DATA\Afstuderen\3.Data\SSM\patient_database\{patient_id}\landmarks"
functions.save_ordered_landmarks(cusp_landmarks, center, output_path)

# %% RECONSTRUCTION WTIH ADDITIONAL CONTROL POINTS

# Base output path for reconstructions
base_recon_path = r"H:\DATA\Afstuderen\3.Data\SSM"

# Non-coronary cusp reconstruction
ncc_path = fr"H:\DATA\Afstuderen\3.Data\SSM\patient_database\{patient_id}\landmarks\ncc_template_landmarks_test.txt"
ncc_ctrlpts = functions.load_leaflet_landmarks(ncc_path)

print("Leaflet Landmarks: ", ncc_ctrlpts)

leaf_1 = functions.calc_ctrlpoints(ncc_ctrlpts, center)
# functions.plot_control_points(leaf_1)

print("Control points of NCC:", leaf_1)

interpolated_leaf_1 = functions.interpolate_surface(leaf_1)
ncc_recon_path = os.path.join(base_recon_path, "ncc", "input_patients", patient_id)
os.makedirs(ncc_recon_path, exist_ok=True)
functions.save_surface_evalpts(interpolated_leaf_1, os.path.join(ncc_recon_path, "ncc_points.txt"))
functions.export_vtk(interpolated_leaf_1, os.path.join(ncc_recon_path, "reconstructed_ncc.vtk"))

## TESTING OTHER COORDINATES HARD CODED


# Left-coronary cusp reconstruction
lcc_path = fr"H:\DATA\Afstuderen\3.Data\SSM\patient_database\{patient_id}\landmarks\lcc_template_landmarks_test.txt"
lcc_ctrlpts = functions.load_leaflet_landmarks(lcc_path)
leaf_2 = functions.calc_ctrlpoints(lcc_ctrlpts, center)
interpolated_leaf_2 = functions.interpolate_surface(leaf_2)
print("Control points of LCC:", leaf_2)
lcc_recon_path = os.path.join(base_recon_path, "lcc", "input_patients", patient_id)
os.makedirs(lcc_recon_path, exist_ok=True)
functions.save_surface_evalpts(interpolated_leaf_2, os.path.join(lcc_recon_path, "lcc_points.txt"))
functions.export_vtk(interpolated_leaf_2, os.path.join(lcc_recon_path, "reconstructed_lcc.vtk"))

# # Right-coronary cusp reconstruction
rcc_path = fr"H:\DATA\Afstuderen\3.Data\SSM\patient_database\{patient_id}\landmarks\rcc_template_landmarks_test.txt"
rcc_ctrlpts = functions.load_leaflet_landmarks(rcc_path)
leaf_3 = functions.calc_ctrlpoints(rcc_ctrlpts, center)
interpolated_leaf_3 = functions.interpolate_surface(leaf_3)
print("Control points of RCC:", leaf_3)
rcc_recon_path = os.path.join(base_recon_path, "rcc", "input_patients", patient_id)
os.makedirs(rcc_recon_path, exist_ok=True)
functions.save_surface_evalpts(interpolated_leaf_3, os.path.join(rcc_recon_path, "rcc_points.txt"))
functions.export_vtk(interpolated_leaf_3, os.path.join(rcc_recon_path, "reconstructed_rcc.vtk"))

# functions.plot_control_points(leaf_1)

