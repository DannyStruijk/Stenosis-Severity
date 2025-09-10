import sys
sys.path.append(r"H:\\DATA\Afstuderen\\2.Code\\Stenosis-Severity\\gui")

import gui_functions as gf
import functions
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import vtk
import nrrd


# %% PREPROCESSING
# READING IN THE DICOM & THE EDGE DETECTED IMAGE

edge_detected_image = r"H:\DATA\Afstuderen\3.Data\Image Processing\aos14\gradient_magnitude.nrrd"
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"

dicom_reversed = gf.get_sorted_dicom_files(dicom_dir)
dicom = dicom_reversed[::-1]

# Convert DICOM to matrix
# volume = gf.dicom_to_matrix(dicom)
initial_volume,header = nrrd.read(edge_detected_image)
volume = np.transpose(initial_volume, (2,1,0))
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
reoriented_volume, rotation_matrix, rotation_center = functions.reorient_volume(rescaled_volume, 
                                                                                annular_normal, 
                                                                                dicom_origin, 
                                                                                pixel_spacing)



# %% PLOTTING THE ANNOTATED LANDMARKS

# In this code the landmarks will be laid over the dicom to assure proper overlay
# This is the original which is used
# lps_coords = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\annotations\ras_coordinates.txt"

# now i am going to try to use the lcc landmarks
lps_coords = r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos14\landmarks\lcc_template_landmarks.txt"

# Calculate the voxel positions of the landmarks 
# Uses the DICOM origin and spacing in order to convert
landmark_voxels = functions.landmarks_to_voxel(lps_coords, dicom_origin, pixel_spacing)

# Rescale the landmarks 
# landmarks_voxel: N x 3 array in (z, y, x)
landmarks_zyx = landmark_voxels[:, [2,1,0]].astype(float)
landmarks_scaled = landmarks_zyx.astype(float)
# print("Old: ", landmarks_scaled)

# # zoom factors used for the volume
# Landmarks need to be zoomed accordingly as the original volume is also zooemd
landmarks_scaled[:, 0] *= slice_thickness  # z-axis
landmarks_scaled[:, 1] *= pixel_spacing_y  # y-axis
landmarks_scaled[:, 2] *= pixel_spacing_x  # x-axis
landmarks_scaled_int = np.round(landmarks_scaled).astype(int)
# print("New: ", landmarks_scaled)

# # Rotate the landmarks
output_shape = reoriented_volume.shape
landmarks_rotated = functions.reorient_landmarks(landmarks_scaled_int, rotation_matrix, dicom_origin, pixel_spacing, output_shape)



#%% VOXELIZING & ROTATING THE VTK

# Rading and loading the VTK into python
vtk_path = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\reconstructions\reconstructed_lcc.vtk"
points_vtk = functions.vtk_to_pointcloud(vtk_path, dicom_origin, pixel_spacing)

# Formatting the landmarks correctly 
points_vtk_zyx = points_vtk[:, [2,1,0]].astype(float)
points_vtk_scaled = points_vtk_zyx.astype(float)

# Landmarks need to be scaled to match alignment with the zoomed volume
points_vtk_scaled[:, 0] *= slice_thickness  # z-axis
points_vtk_scaled[:, 1] *= pixel_spacing_y  # y-axis
points_vtk_scaled[:, 2] *= pixel_spacing_x  # x-axis
points_vtk_scaled_int = np.round(points_vtk_scaled).astype(int)

# Rotating the VTK points using the same rotation center and matrix as priorly
vtk_points_rotated = functions.rotate_vtk_landmarks(points_vtk_scaled, rotation_matrix, rotation_center)
vtk_points_rotated = np.round(vtk_points_rotated).astype(int)




# %%% PLOTTING SINGLE SLICE

# Slice index (X)
slice_idx = 52

# Extract the transversal slice
transversal_slice = reoriented_volume[slice_idx, :, :]

# Plot the slice
plt.figure(figsize=(6,6))
plt.imshow(transversal_slice, cmap="gray")
plt.title(f"Transversal slice at x={slice_idx}")
plt.axis("off")

count = 0
# Overlay landmarks that lie on this slice
for z, y, x in vtk_points_rotated:
    if z == slice_idx:  # Only plot landmarks on this slice
        plt.scatter(x, y, c='r', s=5)  # z → horizontal, y → vertical
        count+=1
        
    
print(count)
plt.show()



#%% PLOTTING ENTIRE FIGURE

# Loop over slice indices
for slice_idx in range(45,60):  # or any range you want
    # Extract the transversal slice
    transversal_slice = reoriented_volume[slice_idx, :, :]
    
    # Clear the current figure
    plt.clf()
    
    # Show the slice
    plt.imshow(transversal_slice, cmap="gray")
    plt.title(f"Transversal slice at x={slice_idx}")
    # plt.axis("off")
    
    # Overlay landmarks that lie on this slice
    for z, y, x in vtk_points_rotated:
        if z == slice_idx:  # Only plot landmarks on this slice
            plt.scatter(x, y, c='r', s=1)  
            # print("Gevonden")
    
    # Draw and pause
    plt.draw()
    plt.pause(0.5)  # pause 2 seconds
    
plt.close()



#%% ACTIVE CONTOURS INITIATION FOR SLICE x=53

from skimage.filters import gaussian
from skimage.segmentation import active_contour
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# --- Preparing the reoriented volume and snake ---
slice_nr = 53
image = reoriented_volume[slice_nr, :, :]

# Extract points in this slice
snake_list = []
for z, y, x in vtk_points_rotated:
    if z == slice_nr:   # use slice_nr to match the slice
        snake_list.append((y, x))
        
# Remove duplicates of the snake
snake = np.array(snake_list)  # shape (N, 2)
unique, idx = np.unique(snake, axis=0, return_index=True)
snake_ordered = snake[np.sort(idx)]

# Find the starting point. For now that is the point in the array that is closest to commissure 1
commissure_1 = landmarks_rotated[1][1:3]
commissure_2 = landmarks_rotated[0][1:3]
start_idx, start_point = functions.find_closest_point(snake_ordered, commissure_1)

# Order the poins accordingly 
backbone = functions.mst_backbone_path(snake_ordered, 9, 19)


######## PLOTTING THE FIGURE

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Snake Points with Indices")
# ax.invert_yaxis()  # optional for image coordinates

# Plot all snake points
ax.scatter(backbone[:,0], backbone[:,1], c='blue', s=60, label='Snake Points')

# Label each point with its index
for i, (x, y) in enumerate(backbone):
    ax.text(x + 0.3, y + 0.3, str(i), color='red', fontsize=12)

# Plot commissure 1
# ax.scatter(commissure_1[0], commissure_1[1], c='green', s=100, marker='X', label='Commissure 1')
# ax.scatter(commissure_2[0], commissure_2[1], c='blue', s=100, marker='X', label='Commissure 2')


ax.legend()
plt.show()

# %%% PLOTTING SINGLE SLICE

# Slice index (X)
slice_idx = 52

# Extract the transversal slice
transversal_slice = reoriented_volume[slice_idx, :, :]

# Plot the slice
plt.figure(figsize=(6,6))
plt.imshow(transversal_slice, cmap="gray")
plt.title(f"Transversal slice at x={slice_idx}")
plt.axis("off")

count = 0
# Overlay landmarks that lie on this slice
for z, y, x in vtk_points_rotated:
    if z == slice_idx:  # Only plot landmarks on this slice
        plt.scatter(x, y, c='r', s=5)  # z → horizontal, y → vertical
        count+=1

plt.scatter(commissure_1[1], commissure_1[0], c='green', s=50, marker='X', label='Commissure 1')
ax.scatter(commissure_2[0], commissure_2[1], c='blue', s=100, marker='X', label='Commissure 2')


    
print(count)
plt.show()


# %% Create a snake


# --- Active contour parameters ---
alpha = 1  # elasticity (snake tension)
beta = 50  # rigidity (smoothness)
gamma = 0.01  # step size


# --- Run active contour ---
snake_refined = active_contour(
    image,
    backbone,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    w_line= 10,
    boundary_condition = "fixed"
)

--- Plotting the result ---
plt.figure(figsize=(6,6))
plt.imshow(image, cmap='gray')
plt.plot(snake_ordered[:,1], snake_ordered[:,0], 'r', lw=2, label='Initial snake')
# plt.plot(snake_refined[:,1], snake_refined[:,0], '-b', lw=2, label='Refined snake')
plt.legend()
plt.axis('off')
plt.show()

