import sys
sys.path.append(r"H:\\DATA\Afstuderen\\2.Code\\Stenosis-Severity\\gui")

import gui_functions as gf
import functions
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import vtk


# %% PREPROCESSING
# READING IN THE DICOM
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"
dicom_reversed = gf.get_sorted_dicom_files(dicom_dir)
dicom = dicom_reversed[::-1]

# Convert DICOM to matrix
volume = gf.dicom_to_matrix(dicom)
dicom_template = pydicom.dcmread(dicom[0][0])

# Extract voxel spacing
slice_thickness = float(dicom_template.SliceThickness)
pixel_spacing_y = float(dicom_template.PixelSpacing[0])
pixel_spacing_x = float(dicom_template.PixelSpacing[1])
pixel_spacing = (slice_thickness, pixel_spacing_y, pixel_spacing_x)
dicom_origin= np.array(dicom_template.ImagePositionPatient, dtype=float)


# %% REORIENTATION
# Calculating the vector perpendicualr to the annulus
patient_nr= "aos14"  # insert here which patient you would like to analyze
annular_normal = functions.get_annular_normal(patient_nr)

rescaled_volume = functions.zoom(volume, pixel_spacing)
rescaled_volume = rescaled_volume[:, :, :]
reoriented_volume, rotation_matrix = functions.reorient_volume(rescaled_volume, annular_normal, dicom_origin, pixel_spacing)
# volume_xyz = np.transpose(reoriented_volume, (2, 1, 0))


# %%% PLOTTING SINGLE SLICE

# Slice index (X)
slice_idx = 10

# Extract the transversal slice
transversal_slice = reoriented_volume[slice_idx, :, :]

# Plot the slice
plt.figure(figsize=(6,6))
plt.imshow(transversal_slice, cmap="gray")
plt.title(f"Transversal slice at x={slice_idx}")
plt.axis("off")

# Overlay landmarks that lie on this slice
# for x, y, z in landmark_voxels:
#     if z == slice_idx:  # Only plot landmarks on this slice
#         plt.scatter(x, y, c='r', s=20)  # z → horizontal, y → vertical

plt.show()




# %% PLOTTING THE ANNOTATED LANDMARKS

# In this code the landmarks will be laid over the dicom to assure proper overlay
lps_coords = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\annotations\ras_coordinates.txt"

# Calculate the voxel positions of the landmarks 
# Uses the DICOM origin and spacing in order to convert
landmark_voxels = functions.landmarks_to_voxel(lps_coords, dicom_origin, pixel_spacing)

# Rescale the landmarks 
# landmarks_voxel: N x 3 array in (z, y, x)
landmarks_zyx = landmark_voxels[:, [2,1,0]].astype(float)
landmarks_scaled = landmarks_zyx.astype(float)
print("Old: ", landmarks_scaled)

# # zoom factors used for the volume
# Landmarks need to be zoomed accordingly as the original volume is also zooemd
landmarks_scaled[:, 0] *= slice_thickness  # z-axis
landmarks_scaled[:, 1] *= pixel_spacing_y  # y-axis
landmarks_scaled[:, 2] *= pixel_spacing_x  # x-axis
landmarks_scaled_int = np.round(landmarks_scaled).astype(int)
print("New: ", landmarks_scaled)

# # Rotate the landmarks
output_shape = reoriented_volume.shape
landmarks_rotated = functions.reorient_landmarks(landmarks_scaled_int, rotation_matrix, dicom_origin, pixel_spacing, output_shape)

#%% PLOTTING ENTIRE FIGURE

# Loop over slice indices
for slice_idx in range(40,70):  # or any range you want
    # Extract the transversal slice
    transversal_slice = reoriented_volume[slice_idx, :, :]
    
    # Clear the current figure
    plt.clf()
    
    # Show the slice
    plt.imshow(transversal_slice, cmap="gray")
    plt.title(f"Transversal slice at x={slice_idx}")
    plt.axis("off")
    
    # Overlay landmarks that lie on this slice
    for z, y, x in landmarks_rotated:
        if z == slice_idx:  # Only plot landmarks on this slice
            plt.scatter(x, y, c='r', s=20)  
            print("Gevonden")
    
    # Draw and pause
    plt.draw()
    plt.pause(0.5)  # pause 2 seconds
    
plt.close()