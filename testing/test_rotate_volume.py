import numpy as np
import os 

os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/gui")

import gui_functions as gf 
from scipy.ndimage import map_coordinates
import scipy.ndimage
import pydicom
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.ndimage import affine_transform

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


#%%%%%%%% Visualization of the slice in 3D space

slice_index = 22

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Display the inclined slice
ax.imshow(rescaled_volume[87], cmap="gray")

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Inclined Slice with Mapped Coordinates (-45 degrees)")

# Show the plot
plt.legend()
plt.show()

# Wrap the volume as a PyVista object
volume_mesh = pv.wrap(rescaled_volume)

# Set up the plotter
plotter = pv.Plotter()
plotter.add_volume(volume_mesh, cmap="gray", opacity="sigmoid")
plotter.show()

#%%%%%%%%%%% ROTATE THE STRUCTURE

# Determine the angle and the plane in which you roate
angle = np.radians(45)
axis = [0,1,0]
R = gf.rotation_matrix(axis, angle)

# Rotate the volume around the specified axis
rotated_volume = gf.rotated_volume(rescaled_volume, R)

#%%%%%%%%% VISUALIZATION OF THE RESCALED VOLUME

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Display the inclined slice
ax.imshow(rotated_volume[180], cmap="gray")

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Inclined Slice with Mapped Coordinates (-45 degrees)")

# Show the plot
plt.legend()
plt.show()

