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


# Use the _test files if you want to have the recently newly annotated landmarks of 3Dslicer
# Landmark files
landmark_files = {
    "LCC": r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos14\landmarks\lcc_template_landmarks_test.txt",
    "RCC": r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos14\landmarks\rcc_template_landmarks_test.txt",
    "NCC": r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos14\landmarks\ncc_template_landmarks_test.txt"
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

from skimage import exposure
from skimage.segmentation import active_contour
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk, erosion, dilation, skeletonize
from skimage.filters import threshold_otsu
from skimage.draw import polygon2mask

# To find the whole aortic wall the active contours method is applied to each slice
# Then, each found wall is accumulated with the previous one to create an aortic valve

# ------------------------------------- INITIALIZING VARIABLES FOR ACTIVE CONTOURS ------------------

alpha = 0.01   # elasticity (snake tension)
beta = 0.1     # rigidity (smoothness)
gamma = 0.01   # step size
total_iterations = 20
scale_factor = 4

z_min_int = int(np.floor(z_min))
z_max_int = int(np.ceil(z_max))
print(f"Processing slices from {z_min_int} to {z_max_int}")

aortic_wall_contours = {}
upsampled_snakes = {}
slice_data = {}   # ✅ NEW: store per-slice info (snake, ROI mask, skeleton)

prev_snake = 0
show_intermediate = False
crop_bool = True

# -------------------------------------- LOOP THROUGH SLICES ---------------------------

for slice_nr in range(z_min_int, z_max_int + 1):
    
    image_pre = reoriented_volume[slice_nr, :, :]

    # Upsample image
    gradient_upsampled = rescale(image_pre, scale_factor, order=3, preserve_range=True, anti_aliasing=True).astype(image_pre.dtype)
    img = gradient_upsampled.astype(np.float32)
    img = img / img.max()
    
    new_image = exposure.equalize_adapthist(img, clip_limit=0.03)

    # Initialize snake
    if slice_nr == z_min_int:
        snake_current = circle_snake.copy() * scale_factor
    else:
        snake_current = prev_snake.copy()

    # --------------------------------------- ACTIVE CONTOUR EVOLUTION --------------------------
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

        if show_intermediate and ((i + 1) % 5 == 0 or i == total_iterations - 1):
            plt.figure(figsize=(6, 6))
            plt.imshow(new_image, cmap="gray")
            plt.plot(snake_current[:, 1], snake_current[:, 0], '-b', lw=2, alpha=0.8)
            plt.title(f"Snake evolution — Slice {slice_nr}, Iteration {i+1}")
            plt.axis("off")
            plt.show()
                
    # --- Final visualization per slice ---
    plt.figure(figsize=(6, 6))
    plt.imshow(new_image, cmap="gray")
    plt.plot(snake_current[:, 1], snake_current[:, 0], '-r', lw=2, alpha=0.9)
    plt.title(f"Final aortic wall contour — Slice {slice_nr}")
    plt.axis("off")
    plt.show()

    # ----------------------------------------- BUILD ROI ----------------------------------------
    roi_mask = polygon2mask(new_image.shape, snake_current) 

    reoriented_slice_dicom = reoriented_dicom[slice_nr, :, :]
    upscaled_slice = rescale(reoriented_slice_dicom, 4, order=3, preserve_range=True, anti_aliasing=True).astype(reoriented_slice_dicom.dtype)
    roi_image = upscaled_slice * roi_mask.astype(reoriented_slice_dicom.dtype)
    
    roi_pixels = roi_image[roi_mask]
    p_low, p_high = np.percentile(roi_pixels, (10, 100))

    roi_windowed = np.zeros_like(roi_image, dtype=np.float32)
    roi_windowed[roi_mask] = np.clip(roi_image[roi_mask], p_low, p_high)
    roi_windowed[roi_mask] = (roi_windowed[roi_mask] - p_low) / (p_high - p_low)

    # ---------------------------------------- SKELETONIZATION -----------------------------------
    roi_windowed /= roi_windowed.max()
    roi_pixels = roi_windowed[roi_mask]
    threshold_val = threshold_otsu(roi_pixels)
    fixed_mask = (roi_windowed > threshold_val) & roi_mask
    inverted_mask = ~fixed_mask
    closed_inverted = dilation(inverted_mask, disk(3))
    eroded_foreground = closed_inverted * roi_mask
    skeleton = skeletonize(eroded_foreground)
    thicker_skeleton = dilation(skeleton, disk(3))

    # Visualization (optional crop)
    if crop_bool:
        points = np.array([lcc_com, rcc_com, ncc_com]) * 4
        x_min, x_max = points[:, 2].min(), points[:, 2].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        pad = 100
        x_min, x_max = int(x_min - pad), int(x_max + pad)
        y_min, y_max = int(y_min - pad), int(y_max + pad)
        cropped_roi = new_image[y_min:y_max, x_min:x_max]
        cropped_skeleton = thicker_skeleton[y_min:y_max, x_min:x_max]
        plt.figure(figsize=(6,6))
        plt.imshow(cropped_roi, cmap='gray')
        plt.imshow(cropped_skeleton, cmap='Reds', alpha=0.4)
        plt.axis('off')
        plt.title(f"Cropped around commissures - Slice {slice_nr}")
        plt.show()

    # Store data for *every* slice
    slice_data[slice_nr] = {
        "skeleton": skeleton.copy(),
        "roi_mask": roi_mask.copy(),
        "snake": snake_current.copy()
    }

    # Store contour info
    final_snake = snake_current / scale_factor
    aortic_wall_contours[slice_nr] = final_snake
    upsampled_snakes[slice_nr] = snake_current.copy()  
    prev_snake = snake_current


# %%------------------------------- EXTRACTION OF THE BOUNDARIES ---------------------------

from skimage.morphology import binary_erosion
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
import numpy as np
import functions

# Parameters
alpha, beta, gamma = 0.1, 0.1, 0.01
total_iterations = 20
scale_factor = 4  # upscaling

# Leaflet-specific storage
LCC_data, RCC_data, NCC_data = {}, {}, {}

# Retrieval and upscaling of points
hinge_lcc = lcc_rotated[2][1:3] * 4
hinge_rcc = rcc_rotated[2][1:3] * 4
hinge_ncc = ncc_rotated[2][1:3] * 4

rcc_lcc_com = lcc_com[1:3] * 4
lcc_ncc_com = ncc_com[1:3] * 4
ncc_rcc_com = rcc_com[1:3] * 4

center = lcc_rotated[3][1:3] * 4


# Loop over slices
for slice_nr, slice_info in slice_data.items():  # or use range(len(slice_data))
    print(f"Processing slice {slice_nr}...")

    # --- Upscale slice for visualization / mask ---
    reoriented_slice = reoriented_dicom[slice_nr, :, :]
    upscaled = rescale(reoriented_slice, scale_factor, order=3, preserve_range=True, anti_aliasing=True).astype(reoriented_slice.dtype)

    # --- Active contour refinement ---
    temp_skeleton = slice_info["skeleton"]
    downsampled_snake = slice_info["snake"].copy()  # downsampled version
    snake_current = functions.resample_closed_contour(downsampled_snake)

    # Preparing the image for the active contours    
    blurred_skeleton_norm = gaussian_filter(temp_skeleton.astype(float), sigma=7)
    blurred_skeleton_norm /= blurred_skeleton_norm.max() + 1e-8

    # Active contours
    for i in range(total_iterations):
        snake_current = active_contour(
            blurred_skeleton_norm,
            snake_current,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            w_edge=1,
            w_line=10,
            max_num_iter=1,
            boundary_condition="periodic"
        )

    # --- Isolate inner skeleton ---
    skeleton_mask = polygon2mask(temp_skeleton.shape, snake_current)
    roi_mask_eroded = binary_erosion(skeleton_mask)
    inner_skeleton = temp_skeleton * roi_mask_eroded.astype(temp_skeleton.dtype)

    # First slice: find and store the closest points
    prev_lcc_ncc_idx = functions.closest_contour_point(lcc_ncc_com, snake_current)
    prev_rcc_lcc_idx = functions.closest_contour_point(rcc_lcc_com, snake_current)
    prev_ncc_rcc_idx = functions.closest_contour_point(ncc_rcc_com, snake_current)
    
    # Use stored indices for this slice
    lcc_ncc_idx = prev_lcc_ncc_idx
    rcc_lcc_idx = prev_rcc_lcc_idx
    ncc_rcc_idx = prev_ncc_rcc_idx

    # --- Find hinge indices on contour ---
    aortic_wall_contour = snake_current
    i_lcc = functions.closest_contour_point(lcc_rotated[2][1:3]*scale_factor, aortic_wall_contour)
    i_rcc = functions.closest_contour_point(rcc_rotated[2][1:3]*scale_factor, aortic_wall_contour)
    i_ncc = functions.closest_contour_point(ncc_rotated[2][1:3]*scale_factor, aortic_wall_contour)

    # --- Create contour segments - i.e. the part of the aortic wall that belongs to the aortic leaflet ---
    seg_lcc_ncc = functions.contour_segment(aortic_wall_contour, i_lcc, i_ncc)
    seg_ncc_rcc = functions.contour_segment(aortic_wall_contour, i_ncc, i_rcc)
    seg_rcc_lcc = functions.contour_segment(aortic_wall_contour, i_rcc, i_lcc)


    # --- Create leaflet-specific masks and boundaries ---
    center = lcc_rotated[3][1:3]*scale_factor

    lcc_ncc_mask, lcc_ncc_boundary = functions.create_boundary_mask(center, aortic_wall_contour, i_lcc, i_ncc, seg_lcc_ncc, upscaled, inner_skeleton)
    rcc_lcc_mask, rcc_lcc_boundary = functions.create_boundary_mask(center, aortic_wall_contour, i_rcc, i_lcc, seg_rcc_lcc, upscaled, inner_skeleton)
    ncc_rcc_mask, ncc_rcc_boundary = functions.create_boundary_mask(center, aortic_wall_contour, i_ncc, i_rcc, seg_ncc_rcc, upscaled, inner_skeleton)
    
    # Boundaries need to be cleaned, removing outliers which are not part of the bigger structure
    cleaned_lcc_ncc_boundary = functions.clean_boundary_from_mask(lcc_ncc_boundary)
    cleaned_rcc_lcc_boundary = functions.clean_boundary_from_mask(rcc_lcc_boundary)
    cleaned_ncc_rcc_boundary = functions.clean_boundary_from_mask(ncc_rcc_boundary)
    
    
    #--- Define commissures depending on slice ---
    if slice_nr == z_min:
        # Base slice: use regular commissures
        lcc_ncc_com_idx = functions.closest_contour_point(lcc_ncc_com, snake_current)
        rcc_lcc_com_idx = functions.closest_contour_point(rcc_lcc_com, snake_current)
        ncc_rcc_com_idx = functions.closest_contour_point(ncc_rcc_com, snake_current)
    
    else:
        # Try to find the new commissures via intersection
        intersections = functions.find_all_boundary_intersections(
            upscaled,
            seg_lcc_ncc,
            seg_rcc_lcc,
            seg_ncc_rcc,
            cleaned_lcc_ncc_boundary,
            cleaned_rcc_lcc_boundary,
            cleaned_ncc_rcc_boundary,
            slice_idx=slice_nr,
            plot=True,
        )
    
        # Initialize fallback commissure indices (use regular ones first)
        lcc_ncc_com_idx = functions.closest_contour_point(lcc_ncc_com, snake_current)
        rcc_lcc_com_idx = functions.closest_contour_point(rcc_lcc_com, snake_current)
        ncc_rcc_com_idx = functions.closest_contour_point(ncc_rcc_com, snake_current)
    
        # Replace only if a valid intersection is found
        if intersections.get("lcc_ncc") is not None:
            lcc_ncc_com_idx = functions.closest_contour_point(intersections["lcc_ncc"], snake_current)
        if intersections.get("rcc_lcc") is not None:
            rcc_lcc_com_idx = functions.closest_contour_point(intersections["rcc_lcc"], snake_current)
        if intersections.get("ncc_rcc") is not None:
            ncc_rcc_com_idx = functions.closest_contour_point(intersections["ncc_rcc"], snake_current)

    
    # --- COM-to-COM segments for actual leaflet boundaries ---
    seg_c2c = {
        "LCC": functions.contour_segment(snake_current, lcc_ncc_com_idx, rcc_lcc_com_idx),
        "RCC": functions.contour_segment(snake_current, ncc_rcc_com_idx, rcc_lcc_com_idx),
        "NCC": functions.contour_segment(snake_current, lcc_ncc_com_idx, ncc_rcc_com_idx)
    }

    # Store in leaflet-specific dictionaries
    LCC_data[slice_nr] = {
        "mask": lcc_ncc_mask,
        "lcc_ncc_boundary": cleaned_lcc_ncc_boundary,
        "rcc_lcc_boundary": cleaned_rcc_lcc_boundary,
        "com_to_com": seg_c2c["LCC"]
    }
    
    RCC_data[slice_nr] = {
        "mask": rcc_lcc_mask,
        "rcc_lcc_boundary": cleaned_rcc_lcc_boundary,
        "ncc_rcc_boundary": cleaned_ncc_rcc_boundary,
        "com_to_com": seg_c2c["RCC"]
    }
    
    NCC_data[slice_nr] = {
        "mask": ncc_rcc_mask,
        "ncc_rcc_boundary": cleaned_ncc_rcc_boundary,
        "lcc_ncc_boundary": cleaned_lcc_ncc_boundary,
        "com_to_com": seg_c2c["NCC"]
    }

    # --- Optional visualization ---
    plt.figure(figsize=(6, 6))
    plt.imshow(upscaled, cmap='gray', origin='upper')
    
    # COM-to-COM segments (actual leaflet wall boundaries)
    plt.plot(seg_c2c["LCC"][:, 1], seg_c2c["LCC"][:, 0], color='cyan', lw=3, label='LCC')
    plt.plot(seg_c2c["RCC"][:, 1], seg_c2c["RCC"][:, 0], color='green', lw=3, label='RCC')
    plt.plot(seg_c2c["NCC"][:, 1], seg_c2c["NCC"][:, 0], color='magenta', lw=3, label='NCC')
    plt.contour(lcc_ncc_boundary, levels=[0.5], colors='cyan', linewidths=2)
    plt.contour(rcc_lcc_boundary, levels=[0.5], colors='cyan', linewidths=2)
    plt.contour(ncc_rcc_boundary, levels=[0.5], colors='magenta', linewidths=2)
    plt.title(f"Leaflet Borders — Slice {slice_nr}")
    plt.axis('off')
    plt.legend()
    plt.show()
    
    
# %%%% TESTING - VISUALIZATION

import matplotlib.pyplot as plt
import numpy as np

# Collect all commissure points in zyx (scaled if needed)
commissures_scaled = np.vstack([
    lcc_com,
    rcc_com,
    ncc_com
])  # shape: (n_points, 3) → [z, y, x]

# Convert to int voxel coordinates for indexing
commissures_voxel = np.round(commissures_scaled).astype(int)

# Determine slice range
z_min_slice = z_min
z_max_slice = z_max

# Loop through slices and plot commissures
for slice_idx in range(z_min_slice, z_max_slice + 1):
    slice_image = reoriented_dicom[slice_idx, :, :]
    
    plt.figure(figsize=(6,6))
    plt.imshow(slice_image, cmap='gray')
    plt.title(f"Slice {slice_idx}")
    
    # Scatter commissures that are on this slice
    for z, y, x in commissures_voxel:
            plt.scatter(x, y, c='r', s=30)  # red dots
    
    plt.axis('off')
    plt.show()


# %% TESTING MERCEDES STAR DICTATES THE AORTIC WALL FOR THE LEAFLET

import functions

# Retrieve and plot
slice_mercedes = 49

# Get the upsampled DICOM slice
reoriented_slice = reoriented_dicom[slice_mercedes, :, :]
upscaled = rescale(reoriented_slice, scale_factor, order=3, preserve_range=True, anti_aliasing=True).astype(reoriented_slice.dtype)

# Upsample the circle snake
downsampled_snake = slice_data[slice_mercedes]["snake"]
upsampled_snake = functions.resample_closed_contour(downsampled_snake)

# Grab the boundaries to determine the seperation marks of the aortic wall
mercedes_lcc_ncc = LCC_data[slice_mercedes]["lcc_ncc_boundary"]
mercedes_rcc_lcc = LCC_data[slice_mercedes]["rcc_lcc_boundary"]
mercedes_ncc_rcc = NCC_data[slice_mercedes]["ncc_rcc_boundary"]

cleaned_mercedes_rcc_lcc = functions.clean_boundary_from_mask(mercedes_rcc_lcc)

# Visualization
plt.figure(figsize=(6, 6))
plt.imshow(upscaled, cmap="gray")
# plt.plot(downsampled_snake[:, 1], downsampled_snake[:, 0], '-b', lw=2, alpha=0.8, label='Upsampled Snake')

# Overlay Mercedes star boundaries
plt.contour(mercedes_lcc_ncc, levels=[0.5], colors='cyan', linewidths=2)
# cleaned_mercedes_rcc_lcc is now Nx2 array
plt.plot(cleaned_mercedes_rcc_lcc[:, 1], cleaned_mercedes_rcc_lcc[:, 0], color='magenta', lw=2)

plt.contour(mercedes_ncc_rcc, levels=[0.5], colors='green', linewidths=2)

# Formatting
plt.title(f"Mercedes Star & Aortic Wall — Slice {slice_mercedes}")
plt.axis("off")
plt.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.show()


# --- Extract the LCC–NCC boundary points ---
y_true, x_true = cleaned_mercedes_rcc_lcc[:, 0], cleaned_mercedes_rcc_lcc[:, 1]

if len(x_true) > 1:
    # Fit a line y = m*x + b through the True pixels (least squares)
    m, b = np.polyfit(x_true, y_true, 1)

    # Compute line coordinates spanning the image
    x_line = np.linspace(0, upscaled.shape[1]-1, 500)
    y_line = m * x_line + b

    # --- Find aortic wall (snake) coordinates ---
    snake_y, snake_x = upsampled_snake[:, 0], upsampled_snake[:, 1]

    # --- Compute intersection point with the snake ---
    # Find closest point on the snake to the line (min distance)
    line_points = np.vstack((y_line, x_line)).T
    snake_points = np.vstack((snake_y, snake_x)).T
    distances = np.linalg.norm(
        snake_points[:, None, :] - line_points[None, :, :],
        axis=2
    )
    idx_snake, idx_line = np.unravel_index(np.argmin(distances), distances.shape)
    intersection = snake_points[idx_snake]

    # --- Visualization ---
    plt.figure(figsize=(6, 6))
    plt.imshow(upscaled, cmap="gray")
    plt.contour(cleaned_mercedes_rcc_lcc, levels=[0.5], colors='cyan', linewidths=5, label='LCC–NCC boundary')
    plt.plot(x_line, y_line, '--r', lw=2, label='Fitted line through boundary')
    plt.plot(snake_x, snake_y, '-b', lw=1.5, alpha=0.8, label='Aortic wall (snake)')
    plt.plot(intersection[1], intersection[0], 'or', markersize=6, label='Intersection')

    plt.title(f"LCC–NCC Boundary & Intersection — Slice {slice_mercedes}")
    plt.axis("off")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.show()

    print(f"Intersection point (y, x): {intersection}")
else:
    print("LCC–NCC boundary has no True pixels or insufficient points for line fitting.")


# %% TESTING IT LOOPED

import functions

all_intersections = {}

for slice_idx in range(z_min, z_max):
    all_intersections[49] = functions.find_all_boundary_intersections(
        slice_idx,
        reoriented_dicom,
        scale_factor,
        slice_data,
        LCC_data,
        RCC_data,
        NCC_data,
        plot=True  # disable plotting for speed
    )