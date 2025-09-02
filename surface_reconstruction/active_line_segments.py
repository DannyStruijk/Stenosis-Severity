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
dicom = gf.get_sorted_dicom_files(dicom_dir)
# dicom = dicom_reversed[::-1]

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
rescaled_volume = rescaled_volume[::-1, :, :]
reoriented_volume, rotation_matrix = functions.reorient_volume(rescaled_volume, annular_normal, dicom_origin, pixel_spacing)
# volume_xyz = np.transpose(reoriented_volume, (2, 1, 0))


# %% PLOTTING THE ANNOTATED LANDMARKS

# In this code the landmarks will be laid over the dicom to assure proper overlay

lps_coords = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\annotations\ras_coordinates.txt"
spacing=  (pixel_spacing_x, pixel_spacing_y, slice_thickness)

# Calculate the voxel positions of the landmarks 
landmark_voxels = functions.landmarks_to_voxel(lps_coords, dicom_origin, spacing)

# Rescale the landmarks 
# landmarks_voxel: N x 3 array in (z, y, x)
landmarks_zyx = landmark_voxels[:, [2,1,0]].astype(float)
landmarks_scaled = landmarks_zyx.astype(float)

# zoom factors used for the volume
landmarks_scaled[:, 0] *= slice_thickness  # z-axis
landmarks_scaled[:, 1] *= pixel_spacing_y  # y-axis
landmarks_scaled[:, 2] *= pixel_spacing_x  # x-axis

landmarks_scaled_int = np.round(landmarks_scaled).astype(int)

# Rotate the landmarks
volume_shape = volume.shape
output_shape = reoriented_volume.shape
landmark_rotated = functions.reorient_landmarks(landmarks_scaled_int, rotation_matrix, dicom_origin, pixel_spacing, volume_shape, output_shape)


# %%% PLOTTING SINGLE SLICE

# Slice index (X)
slice_idx = 40

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

#%% PLOTTING ENTIRE FIGURE

# Loop over slice indices
for slice_idx in range(0, 134):  # or any range you want
    # Extract the transversal slice
    transversal_slice = reoriented_volume[slice_idx, :, :]
    
    # Clear the current figure
    plt.clf()
    
    # Show the slice
    plt.imshow(transversal_slice, cmap="gray")
    plt.title(f"Transversal slice at x={slice_idx}")
    plt.axis("off")
    
    # Overlay landmarks that lie on this slice
    # for x, y, z in landmarks_scaled_int:
    #     if z == slice_idx:  # Only plot landmarks on this slice
    #         plt.scatter(x, y, c='r', s=20)  
    
    # Draw and pause
    plt.draw()
    plt.pause(0.1)  # pause 2 seconds
    
plt.close()

# %% VTK REORIENTATION

# Reading the reconstructed VTK surface
reader= vtk.vtkPolyDataReader()
reader.SetFileName(r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos14\cusps\lcc\lcc.vtk")
reader.Update()
surface = reader.GetOutput()

# Retrieving the necessary metadata to convert the surface data into dicom space
spacing = np.array([slice_thickness, pixel_spacing_y, pixel_spacing_x])
dicom_origin= np.array(dicom_template.ImagePositionPatient, dtype=float)
vtk_in_dicom = functions.vtk_to_volume_space(surface, dicom_origin, spacing)

# Writing the VTK
writer = vtk.vtkPolyDataWriter()
writer.SetFileName(r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos14\cusps\lcc\lcc_in_dicom.vtk")
writer.SetInputData(vtk_in_dicom)
writer.Write()
print("VTK surface correctly reformatted to DICOM space and saved")