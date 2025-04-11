# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:34:27 2025

@author: u840707
"""

# PLOTTING THE import numpy as np
import os 

os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/gui")

import gui_functions as gf 
from scipy.ndimage import map_coordinates
import scipy.ndimage
import pydicom
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.ndimage import affine_transform
import numpy as np 

# Set working directory


# %%%%% Loading the data
# Directory containing the DICOM files and converting them into a volume
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"

# Get a list of all .dcm files (you can filter as needed)
dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)]

# Assuming you have the necessary function to load and sort the DICOM files
sorted_dicom_files = gf.get_sorted_dicom_files(dicom_dir)
volume = np.stack([gf.load_dicom(file[0]) for file in sorted_dicom_files], axis=0)

# Load the first DICOM file for the image properties
dicom = pydicom.dcmread(sorted_dicom_files[0][0])


#%%%%%%%% Rescaling of the volume 

rescaled_volume = gf.rescale_volume(dicom, volume)

# Transpose from (Z, Y, X) → (X, Y, Z)
rescaled_volume = np.transpose(rescaled_volume, (2, 1, 0))
# Optional: Flip left-right to ensure anatomical orientation
# rescaled_volume = np.fliplr(rescaled_volume)

#%%%%%%%% Visualization of the slice in 3D space

slice_index = 250

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Extract the frontal slice (fix the X-coordinate and take all Y and Z values)
#frontal_slice = rescaled_volume[:, slice_index, :]  # Fix X, take all Y and Z

# Display the inclined slice
ax.imshow(np.fliplr(np.rot90(rescaled_volume[:, slice_index, :], k=-1)), cmap="gray")

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Inclined Slice with Mapped Coordinates (-45 degrees)")


plt.legend()
plt.show()


