import numpy as np
import os 
from scipy.ndimage import map_coordinates

# Set working directory
os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/gui")

import gui_functions as gf
import pyvista as pv
from math import pi

import sys
from PyQt5 import QtWidgets, QtCore, QtGui

# %%%%%%% Importing the data
# Importing the GUI functions for loading DICOM files and getting sorted DICOM files
# This assumes you have a module named gui_functions.py with the necessary functions

# %%%%%%% Loading the libraries
sys.path.append(r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6")
sys.path.append(r"H:/DATA/Afstuderen/2.Code/Stenosis-Severity/gui")
# %%%%% Loading the data
# Directory containing the DICOM files and converting them into a volume
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"
sorted_dicom_files = gf.get_sorted_dicom_files(dicom_dir)
volume = np.stack([gf.load_dicom(file[0]) for file in sorted_dicom_files], axis=0)

# Reformat to (X, Y, Z)
volume = volume.transpose(2, 1, 0)
#%%%%%%% Image analysis
# Acquire image properties
depth, height, width = volume.shape
center_x, center_y, center_z  = width // 2, height // 2, depth//2

angle_x = np.radians(0)  # Rotation around X-axis
angle_y = np.radians(0)    # Rotation around Y-axis (unused in this case)

# Define a grid for the slice
y_vals = np.linspace(0, width-1 , width)
x_vals = np.linspace(0, height - 1, height)
X, Y = np.meshgrid(x_vals, y_vals)

slice_index = 23

# Initialize empty list to hold valid coordinates
valid_coords = []

# Iterate through the grid
for i in range(height):
    for j in range(width):
        # Compute Z value for the inclined plane (original Z computation)
        Z = slice_index + (X[i, j] - center_x) * np.tan(angle_x) + (Y[i, j] - center_y) * np.tan(angle_y)

        if 0 <= Z < depth:
            valid_coords.append([Z, Y, X])

# Convert valid coordinates to numpy array
valid_coords = np.array(valid_coords)

# Stack coordinates for interpolation
coords = valid_coords.T

# Interpolate the volume at new coordinates
inclined_slice = map_coordinates(volume, coords, order=1).reshape(
    (len(np.unique(valid_coords[:, 1])), len(np.unique(valid_coords[:, 2])))
)

#%%%%%%%%%%%%%% Rotation with function

import matplotlib.pyplot as plt

# Example: Rotate around Z-axis by 45 degrees
axis = np.array([0, 1, 0])  # Z-axis
angle = np.radians(0)  # Convert 45 degrees to radians

# Define a grid for the slice
y_vals = np.linspace(0, width-1 , width)
x_vals = np.linspace(0, height - 1, height)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.full_like(X, slice_index) 

# Reshape X, Y, Z into a 2D array of points
original_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
center_original = np.mean(original_points, axis=0)

# Get the rotation matrix
R = gf.rotation_matrix(axis, angle)

# Rotate the points
rotated_points = np.dot(original_points - center_original, R.T) + center_original  # Apply rotation and then translate back

# Reshape the rotated points back into a grid for surface plotting
X_rot = rotated_points[:, 0].reshape(X.shape)
Y_rot = rotated_points[:, 1].reshape(Y.shape)
Z_rot = rotated_points[:, 2].reshape(Z.shape)

# Map the rotated coordinates to the volume using map_coordinates
rotated_values = map_coordinates(volume, [X_rot, Y_rot, Z_rot], order=1)

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Display the inclined slice
ax.imshow(rotated_values, cmap="gray")

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Inclined Slice with Mapped Coordinates")

# Show the plot
plt.legend()
plt.show()


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
ax.set_title("Inclined Slice with Mapped Coordinates (-45 degrees)")

# Show the plot
plt.legend()
plt.show()