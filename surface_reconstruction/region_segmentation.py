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

from scipy.ndimage import zoom, gaussian_filter

# original image
edge_detected_image = r"H:\DATA\Afstuderen\3.Data\Image Processing\aos14\gradient_magnitude.nrrd"

# Use this one if using the blurred version
# edge_detected_image = r"H:\DATA\Afstuderen\3.Data\Image Processing\aos14\blurred_sig1.2.nrrd" 

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


# %% REORIENTATION
# Calculating the vector perpendicualr to the annulus
patient_nr= "aos14"  # insert here which patient you would like to analyze
annular_normal = functions.get_annular_normal(patient_nr)

# Reorient the edge-detected image
rescaled_volume = functions.zoom(volume, pixel_spacing)
reoriented_volume, rotation_matrix, rotation_center = functions.reorient_volume(rescaled_volume, 
                                                                                annular_normal, 
                                                                                dicom_origin, 
                                                                                pixel_spacing)

# Now als do the reorientation on the regular DICOM
rescaled_dicom = functions.zoom(smoothed_dicom, pixel_spacing)
reoriented_dicom, rotation_matrix_dicom, rotation_center_dicom = functions.reorient_volume(rescaled_dicom,
                                                                                           annular_normal,
                                                                                           dicom_origin,
                                                                                           pixel_spacing)


# %% PLOTTING THE ANNOTATED LANDMARKS FOR ALL CUSPS

# Landmark files
landmark_files = {
    "LCC": r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos14\landmarks\lcc_template_landmarks.txt",
    "RCC": r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos14\landmarks\rcc_template_landmarks.txt",
    "NCC": r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos14\landmarks\ncc_template_landmarks.txt"
}

# Dictionary to store processed landmarks
landmarks_rotated_dict = {}

# Loop over all cusps
for cusp_name, file_path in landmark_files.items():
    print(cusp_name, file_path)
    # 1. Convert from LPS to voxel coordinates
    landmark_voxels = functions.landmarks_to_voxel(file_path, dicom_origin, pixel_spacing)
    print("Landmark_voxels: ", landmark_voxels)
    # 2. Reorder axes for numpy indexing (z, y, x)
    landmarks_zyx = landmark_voxels[:, [2, 1, 0]].astype(float)
    
    # 3. Scale according to voxel spacing
    landmarks_scaled = landmarks_zyx.copy()
    landmarks_scaled[:, 0] *= slice_thickness   # z-axis
    landmarks_scaled[:, 1] *= pixel_spacing_y  # y-axis
    landmarks_scaled[:, 2] *= pixel_spacing_x  # x-axis
    
    # 4. Round to integer voxel indices
    landmarks_scaled_int = np.round(landmarks_scaled).astype(int)
    
    # 5. Rotate landmarks to match reoriented volume
    landmarks_rotated = functions.reorient_landmarks(
        landmarks_scaled_int, rotation_matrix, dicom_origin, pixel_spacing, reoriented_volume.shape
    )
    print("landmarks rotated are: ", landmarks_rotated)
    
    # Store in dictionary
    landmarks_rotated_dict[cusp_name] = landmarks_rotated


# Access rotated landmarks:
lcc_rotated = landmarks_rotated_dict["LCC"]
rcc_rotated = landmarks_rotated_dict["RCC"]
ncc_rotated = landmarks_rotated_dict["NCC"]

# %% RETRIEVE ALL OF THE DIFFERENT COORDINATES IN ORDER TO 


# --- Load only the first landmark from each file ---
lcc_com = lcc_rotated[0, :]
rcc_com = rcc_rotated[0, :]
ncc_com = ncc_rotated[0, :]

# --- Combine into a single array ---
commissures = np.vstack([lcc_com, rcc_com, ncc_com])  # shape: 3 x 3

# Fit a circle through the commissures to be used in active contours
circle_points = functions.circle_through_commissures(commissures)
circle_snake =  circle_points[:, 1:3] 

# Find more information - Finding the maximum height of the commissure and the minimum height of the hinge point
# this way, the region of interest is also defined for the z-axis
all_rotated = np.vstack([lcc_rotated, rcc_rotated, ncc_rotated])

# Find min and max along the z-axis (axis 0 corresponds to rows, z is column 0 in zyx)
z_min = np.min(all_rotated[:, 0])
z_max = np.max(all_rotated[:, 0])

print(f"Minimum z-coordinate across all cusps: {z_min}")
print(f"Maximum z-coordinate across all cusps: {z_max}")


# %%% PLOTTING SINGLE SLICE

from skimage import exposure

# Slice index (X)
slice_idx = 53

# Extract the transversal slice
transversal_slice = reoriented_dicom[slice_idx, :, :]

# Plot the slice
plt.figure(figsize=(6,6))
plt.imshow(transversal_slice, cmap="gray")
plt.title(f"Transversal slice at x={slice_idx}")
plt.axis("off")

count = 0
# Overlay landmarks that lie on this slice
# for z, y, x in circle_points:
    # if z == slice_idx:  # Only plot landmarks on this slice
        # plt.scatter(x, y, c='r', alpha=0.7,s=2)  # z → horizontal, y → vertical
        # count+=1

    
print(count)
plt.show()



#%% PLOTTING ENTIRE FIGURE

# Loop over slice indices
for slice_idx in range(z_min,z_max):  # or any range you want
    # Extract the transversal slice
    transversal_slice = reoriented_dicom[slice_idx, :, :]
    
    # Clear the current figure
    plt.clf()
    
    # Show the slice
    plt.imshow(transversal_slice, cmap="gray")
    plt.title(f"Transversal slice at x={slice_idx}")
    # plt.axis("off")
    
    # Overlay landmarks that lie on this slice
    for z, y, x in commissures:
        if z == slice_idx:  # Only plot landmarks on this slice
            plt.scatter(x, y, c='r', s=1)  
            # print("Gevonden")
    
    # Draw and pause
    plt.draw()
    plt.pause(1.5)  # pause 2 seconds
    
plt.close()


# %% Create a snake
from scipy import ndimage
from skimage import exposure
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale

# --- Active contour parameters ---
alpha = 0.01  # elasticity (snake tension)
beta = 0.1    # rigidity (smoothness)
gamma = 0.01  # step size
total_iterations = 20  # total iterations you want to observe

# --- Image initialization ---
# --- Preparing the reoriented volume and snake ---
slice_nr =45
image_pre = reoriented_volume[slice_nr, :, :]

# Upsample and normalization of both image and snake
scale_factor = 4
gradient_upsampled = rescale(image_pre, scale_factor, order=3, preserve_range=True, anti_aliasing=True).astype(image_pre.dtype)
circle_snake_rescaled = circle_snake.copy() * scale_factor
snake_current = circle_snake_rescaled

# Normalization & increasing local contrast
img = gradient_upsampled.astype(np.float32)
img = img / img.max()   # scale values to [0,1]
new_image = exposure.equalize_adapthist(img, clip_limit=0.03)

# --- Iteratively update the snake ---
for i in range(total_iterations):
    snake_current = active_contour(
        new_image,
        snake_current,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        w_edge=1,
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
    plt.pause(0.01)
    plt.show()
    
    
    
# %% FIND THE WHOLE AORTIC WALL

# To find the whole aortic wall the active contours method is applied to each slice
# Then, each found wall is accumulated with the previous one to create an aortic valve


from skimage import exposure
from skimage.segmentation import active_contour
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk, erosion, dilation, skeletonize
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from skimage.draw import polygon2mask

# ------------------------------------- INITIALIZING VARIABLES FOR ACTIVE CONTOURS ------------------

# --- Active contour parameters ---
alpha = 0.01   # elasticity (snake tension)
beta = 0.1     # rigidity (smoothness)
gamma = 0.01   # step size
total_iterations = 20  # total iterations per slice
scale_factor = 4

# --- Define z-bounds from landmarks (ensure int range) ---
z_min_int = int(np.floor(z_min))
z_max_int = int(np.ceil(z_max))
print(f"Processing slices from {z_min_int} to {z_max_int}")

# --- Store all contours ---
aortic_wall_contours = {}
upsampled_snakes = {}

# Initialize empty snake, needed for iteration
prev_snake = 0

# Visualization options
show_intermediate = False #True if you want to see the intermediate iterations of the active contours
crop_bool = True #True if you want to see the cropped version of the skeletonization


# --------------------------------------PREPARING THE USED IMAGE ---------------------------

# --- Loop through relevant slices ---
for slice_nr in range(z_min_int, z_max_int + 1):
    
    image_pre = reoriented_volume[slice_nr, :, :]

    # Upsample image
    gradient_upsampled = rescale(image_pre, scale_factor, order=3, preserve_range=True, anti_aliasing=True).astype(image_pre.dtype)
    img = gradient_upsampled.astype(np.float32)
    img = img / img.max()  # normalize to [0,1]
    
    # Increase local contrast
    new_image = exposure.equalize_adapthist(img, clip_limit=0.03)

    # Initialize snake — reuse previous one or start from circle
    if slice_nr == z_min_int:
        snake_current = circle_snake.copy() * scale_factor
    else:
        # Use the previous contour as initialization (propagation)
        snake_current = prev_snake.copy()


    # --------------------------------------- FINDING THE AORTIC WALL --------------------------
    
    # Iteratively update the snake
    for i in range(total_iterations):
        snake_current = active_contour(
            new_image,
            snake_current,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            w_edge=1,
            w_line=10,
            max_num_iter=1,
            boundary_condition="periodic"
        )

        # --- Show evolution every 5 iterations, if wanted ---
        if show_intermediate == True: 
            if (i + 1) % 5 == 0 or i == total_iterations - 1:
                plt.figure(figsize=(6, 6))
                plt.imshow(new_image, cmap="gray")
                plt.plot(snake_current[:, 1], snake_current[:, 0], '-b', lw=2, alpha=0.8, label=f'Iteration {i+1}')
                plt.title(f"Snake evolution — Slice {slice_nr}, Iteration {i+1}")
                plt.legend()
                plt.axis("off")
                plt.pause(0.05)
                plt.show()
                
    # --- Final visualization per slice ---
    plt.figure(figsize=(6, 6))
    plt.imshow(new_image, cmap="gray")
    plt.plot(snake_current[:, 1], snake_current[:, 0], '-r', lw=2, alpha=0.9, label='Final contour')
    plt.title(f"Final aortic wall contour — Slice {slice_nr}")
    plt.legend()
    plt.axis("off")
    plt.pause(0.1)
    plt.show()

    
    # -----------------------------------------BUILD ROI ----------------------------------------
    
    # Step 1: Build ROI mask from snake
    roi_mask = polygon2mask(new_image.shape, snake_current) 
    num_true = np.sum(roi_mask)
    print("Number of True pixels in mask:", num_true)

    # Step 2: Apply mask to the image
    reoriented_slice_dicom = reoriented_dicom[slice_nr, :, :]
    upscaled_slice = rescale(reoriented_slice_dicom, 4, order=3, preserve_range=True, anti_aliasing=True).astype(reoriented_slice_dicom.dtype)
    roi_image = upscaled_slice* roi_mask.astype(reoriented_slice_dicom.dtype)
    
    # Step 3: Show result
    plt.figure(figsize=(6,6))
    plt.imshow(upscaled_slice, cmap='gray')
    plt.axis('off')
    plt.show()

    # Flatten the ROI image and remove zeros (background)
    roi_pixels = roi_image[roi_mask]  # Only consider pixels within the ROI

    # Compute percentiles only within ROI
    p_low, p_high = np.percentile(roi_pixels, (10, 100))

    # Clip and rescale only inside ROI
    roi_windowed = np.zeros_like(roi_image, dtype=np.float32)
    roi_windowed[roi_mask] = np.clip(roi_image[roi_mask], p_low, p_high)
    roi_windowed[roi_mask] = (roi_windowed[roi_mask] - p_low) / (p_high - p_low)



    # ---------------------------------------- SKELETONIZATION -----------------------------------
    # Rescaling to [0,1] the image
    roi_windowed = roi_windowed.astype(np.float32)
    roi_windowed/= roi_windowed.max()

    # otsu thresholding
    roi_pixels = roi_windowed[roi_mask]
    threshold_val = threshold_otsu(roi_pixels)  # scale to [0,1] if roi_float is normalized
    fixed_mask = (roi_windowed > threshold_val) & roi_mask

    # Invert mask so black lines become "foreground"
    inverted_mask = ~fixed_mask

    # Apply morphological closing
    closed_inverted = dilation(inverted_mask, disk(3))
    eroded_foreground = closed_inverted * roi_mask

    # Sekeltonization in order to retrieve thin boundaries for region growing segmentation
    skeleton = skeletonize(eroded_foreground)

    # Upsample skeleton mask
    thicker_skeleton = dilation(skeleton, disk(3))

    # Invert back to original foreground-background convention
    closed_mask = ~closed_inverted

    # optional: if False, the full picture will be visualized
    if crop_bool == False:
        plt.figure(figsize=(6,6))
        plt.imshow(roi_windowed, cmap='gray')           # DICOM in grayscale
        plt.imshow(thicker_skeleton, cmap='Reds', alpha=0.6)  # skeleton in red
        plt.axis('off')
        plt.title(f"Skeleton overlay on DICOM - Slice {slice_nr}")
        plt.show()
     
    # optional: Zoom in on the aortic valve based on the commissures
    else:
        # commissures: each (x, y)
        points = np.array([lcc_com, rcc_com, ncc_com])*4
        x_min, x_max = points[:, 2].min(), points[:, 2].max()  # x = index 2
        y_min, y_max = points[:, 1].min(), points[:, 1].max()  # y = index 1
        
        # add optional padding (e.g. 10 pixels)
        pad = 100
        x_min, x_max = int(x_min - pad), int(x_max + pad)
        y_min, y_max = int(y_min - pad), int(y_max + pad)
        
        # crop the ROI
        cropped_roi = new_image[y_min:y_max, x_min:x_max]
        cropped_skeleton = thicker_skeleton[y_min:y_max, x_min:x_max]
        
        # visualize cropped region
        plt.figure(figsize=(6,6))
        plt.imshow(cropped_roi, cmap='gray')
        plt.imshow(cropped_skeleton, cmap='Reds', alpha=0.4)
        plt.axis('off')
        plt.title(f"Cropped around commissures - Slice {slice_nr}")
        plt.show()

    # Store contour (downscale back to original pixel spacing)
    final_snake = snake_current / scale_factor
    aortic_wall_contours[slice_nr] = final_snake
    upsampled_snakes[slice_nr] = snake_current.copy()  
    prev_snake = snake_current  # store for next slice


# %% VOXELIZATION AORTIC WALL

# --- Combine all contours into a single array ---
aortic_wall_points = []

for slice_nr, contour in aortic_wall_contours.items():
    # contour is shape (num_points, 2) = (y, x)
    z_coords = np.full((contour.shape[0], 1), slice_nr)  # slice index as z
    contour_3d = np.hstack([z_coords, contour])          # (z, y, x)
    aortic_wall_points.append(contour_3d)

# Stack all points into a single array
aortic_wall_points = np.vstack(aortic_wall_points)
print(f"Total points in aortic wall array: {aortic_wall_points.shape[0]}")

# Initialize empty volume
seg_aortic_wall = np.zeros_like(reoriented_volume, dtype=np.uint8)

# Fill in the points from the point cloud
for z, y, x in aortic_wall_points:
    # Ensure indices are integers and within bounds
    z = int(z)
    y = int(y)
    x = int(x)
    if 0 <= z < seg_aortic_wall.shape[0] and 0 <= y < seg_aortic_wall.shape[1] and 0 <= x < seg_aortic_wall.shape[2]:
        seg_aortic_wall[z, y, x] = 1

print(f"Segmentation shape: {seg_aortic_wall.shape}, voxels: {np.sum(seg_aortic_wall)}")


# Choose a slice to visualize (e.g., middle slice)
slice_idx = 60

# Check how many pixels are labeled as aortic wall on this slice
num_aortic_voxels = np.sum(seg_aortic_wall[slice_idx])
print(f"Slice {slice_idx} has {num_aortic_voxels} voxels labeled as aortic wall")

# Extract the transversal slice
transversal_slice = new_image

# Plot the volume slice
plt.figure(figsize=(6, 6))
plt.imshow(transversal_slice, cmap="gray")
plt.title(f"Transversal slice {slice_idx} with aortic wall overlay")

# Overlay the segmentation in red with some transparency
# plt.imshow(seg_aortic_wall[slice_idx], cmap="Reds", alpha=0.4)  # alpha controls transparency

plt.axis("off")
plt.show()



# Start inverse rotation
inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
reoriented_aortic_wall = functions.reorient_landmarks(aortic_wall_points, inverse_rotation_matrix,
                                            dicom_origin, pixel_spacing, reoriented_volume.shape)


#%% PLOTTING ENTIRE FIGURE — TWO VOLUMES

# Loop over slice indices
for slice_idx in range(z_min_int, z_max_int):  # using your z bounds
    # Extract the transversal slices
    slice_volume = reoriented_volume[slice_idx, :, :]
    slice_dicom = reoriented_dicom[slice_idx, :, :]
    
    # Mask for points on this slice
    mask = aortic_wall_points[:, 0] == slice_idx
    pts_on_slice = aortic_wall_points[mask]
    
    # --- Figure 1: Reoriented volume ---
    plt.figure(figsize=(6,6))
    plt.imshow(slice_volume, cmap="gray")
    plt.title(f"Slice {slice_idx} — Reoriented volume")
    plt.scatter(pts_on_slice[:, 2], pts_on_slice[:, 1], c='r', s=0.7)
    plt.axis("off")
    plt.draw()
    plt.pause(1.5)
    plt.close()
    
    # --- Figure 2: Reoriented DICOM ---
    plt.figure(figsize=(6,6))
    plt.imshow(slice_dicom, cmap="gray")
    plt.title(f"Slice {slice_idx} — Reoriented DICOM")
    plt.scatter(pts_on_slice[:, 2], pts_on_slice[:, 1], c='r', s=0.7)
    plt.axis("off")
    plt.draw()
    plt.pause(1.5)
    plt.close()


