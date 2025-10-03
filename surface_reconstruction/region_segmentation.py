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


# %%% PLOTTING SINGLE SLICE

from skimage import exposure

# Slice index (X)
slice_idx = 52

# Extract the transversal slice
transversal_slice = reoriented_dicom[slice_idx, :, :]

# --- Increasing the contrast of the image --- skipped for now
# transverse = transversal_slice.astype(np.float32)
# transverse = transverse / transverse.max()   # scale values to [0,1]
# new_transverse = exposure.equalize_adapthist(transverse, clip_limit=0.01)

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
for slice_idx in range(43,57):  # or any range you want
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
total_iterations = 50  # total iterations you want to observe

# --- Image initialization ---
# --- Preparing the reoriented volume and snake ---
slice_nr =55
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
    
# %%% CREATE AND APPLY ROI MASK

from skimage.draw import polygon2mask

# Step 1: Build ROI mask from snake
roi_mask = polygon2mask(new_image.shape, snake_current) 
num_true = np.sum(roi_mask)

print("Number of True pixels in mask:", num_true)

# Step 2: Apply mask to the image
reoriented_slice_dicom = reoriented_dicom[55, :, :]
upscaled_slice = rescale(reoriented_slice_dicom, 4, order=3, preserve_range=True, anti_aliasing=True).astype(reoriented_slice_dicom.dtype)
roi_image = upscaled_slice* roi_mask.astype(reoriented_slice_dicom.dtype)

# Step 3: Show result
plt.figure(figsize=(6,6))
plt.imshow(roi_image, cmap='gray')
plt.axis('off')
plt.show()



# %% HISTOGRAM PLOTTING & WINDOWING

# Flatten the ROI image and remove zeros (background)
roi_pixels = roi_image[roi_mask]  # Only consider pixels within the ROI

# Compute percentiles only within ROI
p_low, p_high = np.percentile(roi_pixels, (30, 100))

# Clip and rescale only inside ROI
roi_windowed = np.zeros_like(roi_image, dtype=np.float32)
roi_windowed[roi_mask] = np.clip(roi_image[roi_mask], p_low, p_high)
roi_windowed[roi_mask] = (roi_windowed[roi_mask] - p_low) / (p_high - p_low)

# Plot the new region of interest when windowing was applied
plt.figure(figsize=(6,6))
plt.imshow(roi_windowed, cmap='gray')
plt.axis('off')
plt.show()

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(roi_pixels.ravel(), bins=50, color='blue', alpha=0.7)
plt.title("Histogram of ROI")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



# %% REGION GROWING SEGMENTATION

from skimage import segmentation

# Calculate the seed point
lcc_corners = landmarks_rotated[[0,1,3]]
lcc_seed = functions.midpoint_xy(lcc_corners)

region_mask = segmentation.flood(roi_image, lcc_seed, tolerance=0.10)

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

from skimage.morphology import closing, disk, erosion, dilation, skeletonize
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import zoom
from skimage import segmentation
from skimage.transform import rescale

factor = 4 

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

plt.figure(figsize=(6,6))
plt.imshow(roi_windowed, cmap='gray')           # DICOM in grayscale
plt.imshow(thicker_skeleton, cmap='Reds', alpha=0.6)  # skeleton in red
plt.axis('off')
plt.title("Skeleton overlay on DICOM")
plt.show()

# %% REGION GROWING SEGMENTATION

# Set the skeleton pixels to the minimal values to establish the boundaries
seg_image = roi_windowed.copy()
min_val = seg_image.min()
seg_image[skeleton] = min_val

# Show result
plt.figure(figsize=(6,6))
plt.imshow(seg_image, cmap='gray')
plt.axis('off')
plt.show()

# OK - Continue if the boundaries look good. now region growing can begin. 
from skimage import segmentation
from skimage.transform import rescale

# Calculate the seed point
lcc_corners = landmarks_rotated[[0,1,3]]
lcc_seed = functions.midpoint_xy(lcc_corners)
region_mask = segmentation.flood(seg_image, lcc_seed, tolerance=0.4)

# Upsample both the snake, mask and skeleton by factor of four 
factor = 4 
skeleton_up = rescale(skeleton.astype(float), factor, order=0, preserve_range=True).astype(bool)
seg_image_up = rescale(seg_image, factor, order=3, preserve_range=True, anti_aliasing=True).astype(seg_image.dtype)
region_mask_up = rescale(region_mask.astype(float), factor, order=0, preserve_range=True).astype(bool)


# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(seg_image_up, cmap='gray')
# plt.imshow(region_mask_up, cmap='Reds', alpha=0.4)  # overlay mask in red
# plt.scatter(lcc_seed[1], lcc_seed[0], color='blue', s=50, label='Seed')  # note: x=col, y=row
plt.title('Single Cusp Region Growing')
plt.legend()
plt.axis('off')
plt.show()
