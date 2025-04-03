import numpy as np
import os 
from scipy.ndimage import map_coordinates

# Set working directory
os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/gui")

import gui_functions as gf


annular_normal = np.array([ 0.755, 0, 0.655])

# %%%%% Loading the data
# Directory containing the DICOM files and converting them into a volume
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"
sorted_dicom_files = gf.get_sorted_dicom_files(dicom_dir)
volume = np.stack([gf.load_dicom(file[0]) for file in sorted_dicom_files], axis=0)

#%%%%%%% Image analysis
# Acquire image properties
depth, height, width = volume.shape
center_x, center_y, center_z  = width // 2, height // 2, depth//2

angle_x = np.radians(-10)
angle_y = np.radians(0)

# Define a grid for the slice
x_vals = np.linspace(0, width-1 , width)
y_vals = np.linspace(0, height - 1, height)
X, Y = np.meshgrid(x_vals, y_vals)

slice_index = 22

# Initialize empty list to hold valid coordinates
valid_coords = []

# Iterate through the grid
for i in range(height):
    for j in range(width):
        # Compute Z value for the inclined plane
        Z = slice_index + (X[i, j] - center_x) * np.tan(angle_x) + (Y[i, j] - center_y) * np.tan(angle_y)
        
        # Check if Z is within bounds
        if 0 <= Z < depth:
            valid_coords.append([Z, Y[i, j], X[i, j]])

# Convert valid coordinates to numpy array
valid_coords = np.array(valid_coords)

# Stack coordinates for interpolation
coords = valid_coords.T

# Interpolate the volume at new coordinates
inclined_slice = map_coordinates(volume, coords, order=1).reshape((len(np.unique(valid_coords[:, 1])), len(np.unique(valid_coords[:, 2]))))


# # Compute new depth values for the inclined plane
# Z = slice_index + (X - center_x) * np.tan(angle_x) + (Y - center_y) * np.tan(angle_y)

# # Stack coordinates for interpolation
# coords = np.vstack((Z.ravel(), Y.ravel(), X.ravel()))

# # Interpolate the volume at new coordinates
# inclined_slice = map_coordinates(volume, coords, order=1).reshape((width, height))

# %%% VISUALIZATION

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Display the inclined slice
ax.imshow(inclined_slice, cmap="gray")

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Inclined Slice with Mapped Coordinates")

# Show the plot
plt.legend()
plt.show()