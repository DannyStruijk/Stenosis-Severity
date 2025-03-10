# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:29:06 2025

@author: u840707
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pydicom

dicom_file_path = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\dicoms\dicom_viewer_0002\0002.DCM"
dicom = pydicom.dcmread(dicom_file_path)

# Extract the pixel data
image_data = dicom.pixel_array

# Select slice that is to be visualized 
middle_slice = image_data[49, :, :]  # Select the middle slice (slice 48)

# Plot the image using matplotlib
plt.imshow(middle_slice, cmap='gray')
plt.title("DICOM Image")
plt.axis('off')  # Hide axes for better visualization
plt.show()