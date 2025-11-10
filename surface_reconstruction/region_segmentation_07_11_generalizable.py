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

# Load the dicom
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"
dicom_reversed = gf.get_sorted_dicom_files(dicom_dir)
dicom = dicom_reversed[::-1]
raw_volume = gf.dicom_to_matrix(dicom)

# Extract DICOM attributes, also to reformat the image to HU units
dicom_template = pydicom.dcmread(dicom[0][0])
slope = float(getattr(dicom_template, "RescaleSlope", 1))
intercept = float(getattr(dicom_template, "RescaleIntercept", 0))
raw_volume_hu = raw_volume.astype(np.float32) * slope + intercept

# Create an edge detected image based on the import DICOM
gradient_volume = functions.compute_edge_volume(raw_volume_hu, hu_window=(0, 400), sigma=0.8, normalize=False)
volume = gradient_volume

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


# %% FIND THE WHOLE AORTIC WALL

from skimage.segmentation import active_contour
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, dilation, skeletonize
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
    
    # No equalization since contrast is already enhacned
    # new_image = exposure.equalize_adapthist(img, clip_limit=0.03)
    new_image= img

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
mercedes_slice = lcc_rotated[3][0]

# Initialize previous intersections with None (persistent storage across slices)
prev_intersections = {
    "lcc_ncc": None,
    "rcc_lcc": None,
    "ncc_rcc": None
}

# Base commissures (from first slice landmarks)
base_commissure = {
    "lcc_ncc": lcc_ncc_com,
    "rcc_lcc": rcc_lcc_com,
    "ncc_rcc": ncc_rcc_com
}

mercedes_star = False

# Loop over slices
for slice_nr, slice_info in slice_data.items():
    print(f"Processing slice {slice_nr}...")

    # --- Upscale slice for visualization / mask ---
    reoriented_slice = reoriented_dicom[slice_nr, :, :]
    upscaled = rescale(
        reoriented_slice,
        scale_factor,
        order=3,
        preserve_range=True,
        anti_aliasing=True
    ).astype(reoriented_slice.dtype)

    # --- Active contour refinement ---
    temp_skeleton = slice_info["skeleton"]
    downsampled_snake = slice_info["snake"].copy()
    snake_current = functions.resample_closed_contour(downsampled_snake)

    # Preparing image for active contours    
    blurred_skeleton_norm = gaussian_filter(temp_skeleton.astype(float), sigma=7)
    blurred_skeleton_norm /= blurred_skeleton_norm.max() + 1e-8

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

    # Upsample contour for accurate indices
    aortic_wall_contour = functions.resample_closed_contour(snake_current)

    # Find closest points to hinge points
    i_lcc = functions.closest_contour_point(lcc_rotated[2][1:3]*scale_factor, aortic_wall_contour)
    i_rcc = functions.closest_contour_point(rcc_rotated[2][1:3]*scale_factor, aortic_wall_contour)
    i_ncc = functions.closest_contour_point(ncc_rotated[2][1:3]*scale_factor, aortic_wall_contour)

    # Create contour segments
    seg_lcc_ncc = functions.contour_segment(aortic_wall_contour, i_lcc, i_ncc)
    seg_ncc_rcc = functions.contour_segment(aortic_wall_contour, i_ncc, i_rcc)
    seg_rcc_lcc = functions.contour_segment(aortic_wall_contour, i_rcc, i_lcc)

    # Create leaflet masks and boundaries
    lcc_ncc_mask, lcc_ncc_boundary = functions.create_boundary_mask(
        center, aortic_wall_contour, i_lcc, i_ncc, seg_lcc_ncc, upscaled, inner_skeleton)
    rcc_lcc_mask, rcc_lcc_boundary = functions.create_boundary_mask(
        center, aortic_wall_contour, i_rcc, i_lcc, seg_rcc_lcc, upscaled, inner_skeleton)
    ncc_rcc_mask, ncc_rcc_boundary = functions.create_boundary_mask(
        center, aortic_wall_contour, i_ncc, i_rcc, seg_ncc_rcc, upscaled, inner_skeleton)

    # Clean boundaries
    cleaned_lcc_ncc_boundary = functions.clean_boundary_from_mask(lcc_ncc_boundary, aortic_wall_contour)
    cleaned_rcc_lcc_boundary = functions.clean_boundary_from_mask(rcc_lcc_boundary, aortic_wall_contour)
    cleaned_ncc_rcc_boundary = functions.clean_boundary_from_mask(ncc_rcc_boundary, aortic_wall_contour)

    # --- Find intersections with wall ---
    intersections = {}
    if slice_nr != z_min:
        intersections = functions.find_all_boundary_intersections(
            upscaled,
            seg_lcc_ncc,
            seg_rcc_lcc,
            seg_ncc_rcc,
            cleaned_lcc_ncc_boundary,
            cleaned_rcc_lcc_boundary,
            cleaned_ncc_rcc_boundary,
            slice_idx=slice_nr,
            plot=True
        )

    # --- Assign commissure indices robustly ---
    commissure_names = ["lcc_ncc", "rcc_lcc", "ncc_rcc"]
    commissure_indices = {}

    for name in commissure_names:
        # Prefer newly found intersection
        if intersections.get(name) is not None:
            prev_intersections[name] = intersections[name]
            commissure_indices[name] = functions.closest_contour_point(intersections[name], aortic_wall_contour)
        else:
            # Fallback: previous intersection if exists, else base landmark
            fallback = prev_intersections[name] if prev_intersections[name] is not None else base_commissure[name]
            commissure_indices[name] = functions.closest_contour_point(fallback, aortic_wall_contour)

    # Create COM-to-COM segments
    seg_c2c = {
        "LCC": functions.contour_segment(aortic_wall_contour, commissure_indices["lcc_ncc"], commissure_indices["rcc_lcc"]),
        "RCC": functions.contour_segment(aortic_wall_contour, commissure_indices["ncc_rcc"], commissure_indices["rcc_lcc"]),
        "NCC": functions.contour_segment(aortic_wall_contour, commissure_indices["lcc_ncc"], commissure_indices["ncc_rcc"])
    }

    # Store results
    LCC_data[slice_nr] = {
        "mask": lcc_ncc_mask.copy(),
        "lcc_ncc_boundary": cleaned_lcc_ncc_boundary.copy(),
        "rcc_lcc_boundary": cleaned_rcc_lcc_boundary.copy(),
        "com_to_com": seg_c2c["LCC"].copy(),
        "aortic_wall_contour": aortic_wall_contour.copy()
    }
    RCC_data[slice_nr] = {
        "mask": rcc_lcc_mask.copy(),
        "rcc_lcc_boundary": cleaned_rcc_lcc_boundary.copy(),
        "ncc_rcc_boundary": cleaned_ncc_rcc_boundary.copy(),
        "com_to_com": seg_c2c["RCC"].copy()
    }
    NCC_data[slice_nr] = {
        "mask": ncc_rcc_mask.copy(),
        "ncc_rcc_boundary": cleaned_ncc_rcc_boundary.copy(),
        "lcc_ncc_boundary": cleaned_lcc_ncc_boundary.copy(),
        "com_to_com": seg_c2c["NCC"].copy()
    }

    # Optional visualization (COM-to-COM segments + boundaries)
    plt.figure(figsize=(6, 6))
    plt.imshow(upscaled, cmap='gray', origin='upper')
    plt.plot(seg_c2c["LCC"][:, 1], seg_c2c["LCC"][:, 0], color='cyan', lw=3, label='LCC')
    plt.plot(seg_c2c["RCC"][:, 1], seg_c2c["RCC"][:, 0], color='green', lw=3, label='RCC')
    plt.plot(seg_c2c["NCC"][:, 1], seg_c2c["NCC"][:, 0], color='magenta', lw=3, label='NCC')
    plt.plot(cleaned_lcc_ncc_boundary[:, 1], cleaned_lcc_ncc_boundary[:, 0], color='cyan', lw=2)
    plt.plot(cleaned_rcc_lcc_boundary[:, 1], cleaned_rcc_lcc_boundary[:, 0], color='green', lw=2)
    plt.plot(cleaned_ncc_rcc_boundary[:, 1], cleaned_ncc_rcc_boundary[:, 0], color='magenta', lw=2)
    plt.title(f"Leaflet Borders — Slice {slice_nr}")
    plt.axis('off')
    plt.legend()
    plt.show()
    
    