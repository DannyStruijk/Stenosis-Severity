import numpy as np
import os 

os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/gui")

import gui_functions as gf 
from scipy.ndimage import map_coordinates
import scipy.ndimage
import pydicom
import matplotlib.pyplot as plt

# Set working directory


# %%%%% Loading the data
# Directory containing the DICOM files and converting them into a volume
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"

# Get a list of all .dcm files (you can filter as needed)
dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)]

# Load the first DICOM file
ds = pydicom.dcmread(dicom_files[0])

# Print pixel spacing and slice thickness to understand the spacing
print("Pixel Spacing:", ds.PixelSpacing)
print("Slice Thickness:", ds.SliceThickness)

# Assuming you have the necessary function to load and sort the DICOM files
sorted_dicom_files = gf.get_sorted_dicom_files(dicom_dir)
volume = np.stack([gf.load_dicom(file[0]) for file in sorted_dicom_files], axis=0)

#%%%%%%%% ANALYSIS 
# Get the pixel spacing (assuming it's the same for all slices)
pixel_spacing = ds.PixelSpacing  # [mm/pixel] for (x, y)
slice_thickness = ds.SliceThickness  # [mm] for z-axis

# Define the voxel spacing in X, Y, and Z directions
voxel_spacing = np.array([pixel_spacing[0], pixel_spacing[1], slice_thickness])

# Now perform the rotation while taking the voxel spacing into account
# First, adjust the scaling according to the voxel spacing
scaling_factors = np.array([1 / voxel_spacing[0], 1 / voxel_spacing[1], 1 / voxel_spacing[2]])

# Scale the volume along each axis separately
scaled_volume_x = scipy.ndimage.zoom(volume, (scaling_factors[0], 1, 1), order=1)
scaled_volume_y = scipy.ndimage.zoom(scaled_volume_x, (1, scaling_factors[1], 1), order=1)
scaled_volume_z = scipy.ndimage.zoom(scaled_volume_y, (1, 1, scaling_factors[2]), order=1)


#%%%%%%%% Now you can rotate the volume if needed
rotated_scaled_volume = scipy.ndimage.rotate(scaled_volume_z, angle=-1, axes=(0, 2), reshape=False, order=1)

# Create figure to display a slice
fig, ax = plt.subplots(figsize=(10, 8))

# Display the slice from the rotated volume
ax.imshow(rotated_scaled_volume[25, :, :], cmap="gray")

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Inclined Slice with Mapped Coordinates")

# Show the plot
plt.legend()
plt.show()
