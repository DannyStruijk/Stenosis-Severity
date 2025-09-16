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


# %% PREPROCESSING
# READING IN THE DICOM & THE EDGE DETECTED IMAGE

# original image
edge_detected_image = r"H:\DATA\Afstuderen\3.Data\Image Processing\aos14\gradient_magnitude.nrrd"
# edge_detected_image = r"H:\DATA\Afstuderen\3.Data\Image Processing\aos14\blurred_sig1.2.nrrd"

dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"
dicom_reversed = gf.get_sorted_dicom_files(dicom_dir)
dicom = dicom_reversed[::-1]

# Convert DICOM to matrix

# Decide whether to use the raw DICOM or the edge detected image, exported from 3Dslicer
use_edge = True
if use_edge == False:
    volume = gf.dicom_to_matrix(dicom)
else:
    initial_volume,header = nrrd.read(edge_detected_image)
    volume = np.transpose(initial_volume, (2,1,0))
    
# Histogram equalization - Increase contrast    
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
rcc_coords = r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos14\landmarks\rcc_template_landmarks.txt"
ncc_coords = r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos14\landmarks\ncc_template_landmarks.txt" 

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

# %% RETRIEVE ALL OF THE DIFFERENT COORDINATES IN ORDER TO 

# --- Load only the first landmark from each file ---
lcc_com = functions.landmarks_to_voxel(lps_coords, dicom_origin, pixel_spacing)[0:1, :]
rcc_com = functions.landmarks_to_voxel(rcc_coords, dicom_origin, pixel_spacing)[0:1, :]
ncc_com = functions.landmarks_to_voxel(ncc_coords, dicom_origin, pixel_spacing)[0:1, :]

# --- Combine into a single array ---
first_landmarks_voxels = np.vstack([lcc_com, rcc_com, ncc_com])  # shape: 3 x 3

# --- Rescale to (z, y, x) ---
landmarks_zyx = first_landmarks_voxels[:, [2,1,0]].astype(float)
landmarks_scaled = landmarks_zyx.copy()
landmarks_scaled[:, 0] *= slice_thickness  # z-axis
landmarks_scaled[:, 1] *= pixel_spacing_y  # y-axis
landmarks_scaled[:, 2] *= pixel_spacing_x  # x-axis
landmarks_scaled_int = np.round(landmarks_scaled).astype(int)

# --- Rotate landmarks to match reoriented volume ---
output_shape = reoriented_volume.shape
commissures_rotated = functions.reorient_landmarks(
    landmarks_scaled_int,
    rotation_matrix,
    dicom_origin,
    pixel_spacing,
    output_shape
)

# Fit a circle through the commissures to be used in active contours
circle_points = functions.circle_through_commissures(commissures_rotated)
circle_snake =  circle_points[:, 1:3] 




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
slice_idx = 46

# Extract the transversal slice
transversal_slice = reoriented_volume[slice_idx, :, :]

# Plot the slice
plt.figure(figsize=(6,6))
plt.imshow(transversal_slice, cmap="gray")
plt.title(f"Transversal slice at x={slice_idx}")
plt.axis("off")

count = 0
# Overlay landmarks that lie on this slice
for z, y, x in circle_points:
    # if z == slice_idx:  # Only plot landmarks on this slice
        plt.scatter(x, y, c='r', alpha=0.7,s=2)  # z → horizontal, y → vertical
        count+=1
        
    
print(count)
plt.show()



#%% PLOTTING ENTIRE FIGURE

# Loop over slice indices
for slice_idx in range(43,49):  # or any range you want
    # Extract the transversal slice
    transversal_slice = reoriented_volume[slice_idx, :, :]
    
    # Clear the current figure
    plt.clf()
    
    # Show the slice
    plt.imshow(transversal_slice, cmap="gray")
    plt.title(f"Transversal slice at x={slice_idx}")
    # plt.axis("off")
    
    # Overlay landmarks that lie on this slice
    for z, y, x in commissures_rotated:
        if z == slice_idx:  # Only plot landmarks on this slice
            plt.scatter(x, y, c='r', s=1)  
            # print("Gevonden")
    
    # Draw and pause
    plt.draw()
    plt.pause(1)  # pause 2 seconds
    
plt.close()



#%% ACTIVE CONTOURS INITIATION FOR SLICE x=53

from skimage.filters import gaussian
from skimage.segmentation import active_contour
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from skimage import exposure

# --- Preparing the reoriented volume and snake ---
slice_nr = 53
image_pre = reoriented_volume[slice_nr, :, :]
image = exposure.equalize_hist(image_pre)   # output is in [0,1]

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

# Downsample the amount of points of the backbone, reduces snake complexity
if len(backbone) > 2:
    start_point = backbone[0:1]           # first point
    end_point = backbone[-1:]             # last point
    middle_points = backbone[1:-1]        # points in between
    middle_points_downsampled = middle_points[::4]  # take every 2nd point
    backbone_reduced = np.vstack([start_point, middle_points_downsampled, end_point])
else:
    # If backbone has only 2 points, just keep as is
    backbone_reduced = backbone


######## PLOTTING THE FIGURE

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Snake Points with Indices")
# ax.invert_yaxis()  # optional for image coordinates

# Plot all snake points
ax.scatter(backbone_reduced[:,0], backbone_reduced[:,1], c='blue', s=60, label='Snake Points')

# Label each point with its index
for i, (x, y) in enumerate(backbone_reduced):
    ax.text(x + 0.3, y + 0.3, str(i), color='red', fontsize=12)

# Plot commissure 1
# ax.scatter(commissure_1[0], commissure_1[1], c='green', s=100, marker='X', label='Commissure 1')
# ax.scatter(commissure_2[0], commissure_2[1], c='blue', s=100, marker='X', label='Commissure 2')


ax.legend()
plt.show()

# %%% PLOTTING SINGLE SLICE

# Slice index (X)
slice_idx = 53

# Extract the transversal slice
transversal_slice = image

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
from scipy import ndimage
from skimage import exposure
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
import numpy as np

# --- Active contour parameters ---
alpha = 0.5  # elasticity (snake tension)
beta = 2    # rigidity (smoothness)
gamma = 0.1  # step size
total_iterations = 10  # total iterations you want to observe

# --- Image initialization ---
img = image_pre.astype(np.float32)
img = img / img.max()   # scale values to [0,1]
new_image = exposure.equalize_adapthist(img, clip_limit=0.03)

# --- Initialize snake ---
snake_current = circle_snake.copy()

# --- Iteratively update the snake ---
for i in range(total_iterations):
    snake_current = active_contour(
        new_image,
        circle_snake,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        w_edge=0,
        w_line=10,
        max_num_iter=1,           # run only 1 iteration per loop
        boundary_condition="periodic"
    )
    
    # --- Plot current snake ---
    plt.figure(figsize=(6,6))
    plt.imshow(new_image, cmap='gray')
    # plt.plot(circle_snake[:, 1], circle_snake[:, 0], 'ro', markersize=1, label='Initial points')
    plt.plot(snake_current[:, 1], snake_current[:, 0], '-b', lw=2, alpha=0.7, label=f'Snake after {i+1} iterations')
    plt.legend()
    plt.axis('off')
    plt.pause(0.2)
    plt.show()
    
# %%% CREATE AND APPLY ROI MASK

from skimage.draw import polygon2mask
import SimpleITK as sitk 

# Step 1: Build ROI mask from snake
roi_mask = polygon2mask(new_image.shape, snake_current) 
num_true = np.sum(roi_mask)
print("Number of True pixels in mask:", num_true)


# Step 2: Apply mask to the image
roi_image = new_image * roi_mask.astype(new_image.dtype)

# # Apply closing (on the image directly)
sitk_image = sitk.GetImageFromArray(roi_image)
closed = sitk.BinaryMorphologicalClosing(sitk_image > 180, [2,2,2])  # adjust threshold + radius

# Convert SimpleITK image back to numpy array
closed_array = sitk.GetArrayFromImage(closed)


# Step 3: Show result
plt.figure(figsize=(6,6))
plt.imshow(roi_image, cmap='gray')
# plt.plot(snake_current[:, 1], snake_current[:, 0], '-r', lw=1)  # overlay snake
plt.axis('off')
plt.show()

# %%% ACTIVE CONTOURS FOR THE CUSP RECONSTRUCTION

from skimage import exposure
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
import numpy as np

# --- Active contour parameters ---
alpha = 0.0002  # elasticity (snake tension)
beta = 0.01    # rigidity (smoothness)
gamma = 0.001  # step size
total_iterations = 5  # total iterations you want to observe

# --- Image initialization ---
img = image_pre.astype(np.float32)
img = img / img.max()   # scale values to [0,1]

# --- Clip values below threshold ---
threshold = 0.3   # set threshold as fraction of max intensity
img_clipped = roi_image.copy()
img_clipped[img_clipped < threshold] = 0

# --- Show histogram ---
# Compute statistics
mean_intensity = np.mean(img)
std_intensity = np.std(img)

print(f"Average intensity: {mean_intensity:.4f}")
print(f"Standard deviation: {std_intensity:.4f}")
# new_image = exposure.equalize_adapthist(img_clipped, clip_limit=0.01)

# --- Initialize snake ---
snake_current = backbone_reduced.copy()

# --- Iteratively update the snake ---
for i in range(total_iterations):
    snake_current = active_contour(
        img_clipped,
        snake_current,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        w_edge=0,
        w_line=2,
        max_num_iter=1,           # run only 1 iteration per loop
        boundary_condition="fixed"
    )
    
    # --- Plot current snake ---
    plt.figure(figsize=(6,6))
    plt.imshow(img_clipped, cmap='gray')
    plt.plot(backbone_reduced[:, 1], backbone_reduced[:, 0], 'ro', markersize=1, label='Initial points')
    plt.plot(snake_current[:, 1], snake_current[:, 0], '-b', lw=2, alpha=0.9, label=f'Snake after {i+1} iterations')
    plt.legend()
    plt.axis('off')
    plt.pause(0.5)
    plt.show()
    

# %% REGION GROWING SEGMENTATION

from skimage import segmentation

# Calculate the seed point
lcc_corners = landmarks_rotated[[0,1,3]]
lcc_seed = functions.midpoint_xy(lcc_corners)

region_mask = segmentation.flood(roi_image, lcc_seed, tolerance=0.20)

# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(roi_image, cmap='gray')
plt.imshow(region_mask, cmap='Reds', alpha=0.4)  # overlay mask in red
plt.scatter(lcc_seed[1], lcc_seed[0], color='blue', s=50, label='Seed')  # note: x=col, y=row
plt.title('Single Cusp Region Growing')
plt.legend()
plt.axis('off')
plt.show()

# %% IMAGE PROCESSING TESTING

from skimage import exposure

grad = transversal_slice 

# Normalize and equalize
img_eq = exposure.equalize_hist(grad)   # output is in [0,1]

# Plot comparison
fig, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].imshow(image_pre, cmap='gray'); axes[0].set_title("Original gradient")
axes[1].imshow(img_eq, cmap='gray'); axes[1].set_title("Equalized gradient")
plt.show()