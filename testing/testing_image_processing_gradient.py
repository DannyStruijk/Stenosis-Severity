# SCRIPT TO REFINE THE SEGMENTATION OF THE AORTIC LEAFLETS 

# Importing the necessary packages

import sys
sys.path.append(r"H:\\DATA\Afstuderen\\2.Code\\Stenosis-Severity\\gui")
import gui_functions as gf
import functions
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import nrrd
from skimage import exposure

# %% PREPROCESSING
# READING IN THE DICOM & THE EDGE DETECTED IMAGE

from scipy.ndimage import zoom, gaussian_filter

# original image
edge_detected_image = r"H:\DATA\Afstuderen\3.Data\Image Processing\aos14\gradient_magnitude.nrrd"

# Load the dicom
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"
dicom_reversed = gf.get_sorted_dicom_files(dicom_dir)
dicom = dicom_reversed[::-1]
raw_volume = gf.dicom_to_matrix(dicom)

# Convert DICOM to matrix
# Decide whether to use the raw DICOM or the edge detected image, exported from 3Dslicer
initial_volume, header = nrrd.read(edge_detected_image)
volume = np.transpose(initial_volume, (2,1,0))

# Extract DICOM attributes, also to reformat the image to HU units
dicom_template = pydicom.dcmread(dicom[0][0])
slope = float(getattr(dicom_template, "RescaleSlope", 1))
intercept = float(getattr(dicom_template, "RescaleIntercept", 0))
raw_volume_hu = raw_volume.astype(np.float32) * slope + intercept

# Clipping the DICOM to remove calcifications
low, high = 100, 340
clipped_dicom = np.clip(raw_volume_hu, low, high)

# Smoothing the image abit
smoothed_dicom = gaussian_filter(clipped_dicom, sigma = 1)

# Extract voxel spacing
slice_thickness = float(dicom_template.SliceThickness)
pixel_spacing_y = float(dicom_template.PixelSpacing[0])
pixel_spacing_x = float(dicom_template.PixelSpacing[1])
pixel_spacing = (slice_thickness, pixel_spacing_y, pixel_spacing_x)
dicom_origin= np.array(dicom_template.ImagePositionPatient, dtype=float)

# %% EDGE DETECTION

from skimage.filters import sobel
from scipy.ndimage import gaussian_filter

# Window HU to soft tissue range (leaflets)
leaflet_window = np.clip(raw_volume_hu, 0, 400)

# Initialize gradient volume
gradient_volume = np.zeros_like(leaflet_window, dtype=np.float32)

# Process each slice independently
for i in range(raw_volume_hu.shape[0]):
    slice_smoothed = gaussian_filter(leaflet_window[i, :, :], sigma=0.8)
    gradient_volume[i, :, :] = sobel(slice_smoothed)

# Optional: normalize for visualization
# gradient_volume = exposure.rescale_intensity(gradient_volume, out_range=(0, 255))




# %% VISUALIZE DICOM SLICES
import matplotlib.pyplot as plt
import time

# Select which volume to visualize
# You can choose: raw_volume_hu, clipped_dicom, smoothed_dicom, or volume (NRRD)
display_volume = gradient_volume# change this to visualize other versions

# Loop through all slices and display them
plt.figure(figsize=(6, 6))
for i in range(display_volume.shape[0]):
    plt.imshow(display_volume[i, :, :], cmap='gray')
    plt.title(f"Slice {i+1}/{display_volume.shape[0]}")
    plt.axis('off')
    plt.pause(0.05)  # controls the speed of scrolling (in seconds)
    plt.clf()        # clear figure for next slice

plt.close()
