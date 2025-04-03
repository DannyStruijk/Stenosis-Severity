import os
import numpy as np
import pyvista as pv
import pydicom
import matplotlib.pyplot as plt

# Set working directory
os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/gui")

import gui_functions as gf

annular_normal = np.array([ 0.755, 0, 0.655])  # Example

# Directory containing the DICOM files and converting them into a volume
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"
sorted_dicom_files = gf.get_sorted_dicom_files(dicom_dir)
volume = np.stack([gf.load_dicom(file[0]) for file in sorted_dicom_files], axis=0)

# Calcualte the rotation needed to look perpendicular to the annular plane
rotated_volume = gf.reslice_numpy_volume(volume, annular_normal)

#rotated_volume = gf.apply_rotation(volume, rotation_axis, rotation_angle)

# Create subplots for original and rotated images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original Image (No Rotation)
axes[0].imshow(volume[25], cmap="gray")
axes[0].set_title("Original Slice")
axes[0].axis("off")

# Rotated Image (Z-axis rotated by 180 degrees)
axes[1].imshow(rotated_volume[11], cmap="gray")
axes[1].set_title("Z-axis Rotated Slice (180°)")
axes[1].axis("off")

# Show the plots
plt.tight_layout()
plt.show()
