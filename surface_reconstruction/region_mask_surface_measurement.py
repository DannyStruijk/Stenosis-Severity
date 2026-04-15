# SCRIPT TO REFINE THE SEGMENTATION OF THE AORTIC LEAFLETS 

# Importing the necessary packages

import sys
sys.path.append(r"H:\\DATA\Afstuderen\\2.Code\\Stenosis-Severity-backup\\gui")
import gui_functions as gf
import functions
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_dilation, cube, binary_closing
from skimage.segmentation import active_contour
from skimage.morphology import disk, dilation, skeletonize
from skimage.draw import polygon2mask
from skimage.filters import gaussian
from scipy.ndimage import binary_erosion
from skimage.restoration import inpaint

# %% ----------------------------------------PATIENT PATHS & SELECTION OF PATIENT -----------------------------

PATIENT_PATHS = {
    "CZE001": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE001/DICOM/00003852/AA44D04F/AA7BB8C5/000050B5",
    "CZE002": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE002/DICOM/0000AFC5/AAAA2796/AAFF16B0/0000ADA6",
    "CZE003": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE003/DICOM/0000AF6A/AA4272CE/AA72A45E/000050F2",
    "CZE004": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE004/DICOM/00002F76/AA1F4542/AAB1E4E9/0000CAC5",
    "CZE005": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE005/DICOM/0000AF52/AA590C3F/AAC428CF/0000FEE0",
    "CZE006": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE006/DICOM/00000EED/AA87381C/AAAEC035/00002581",
    "CZE007": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE007/DICOM/000065F6/AAB95BAE/AA7A7E4C/00005896",
    "CZE008": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE008/DICOM/000053DF/AA102722/AA7E9491/000073C6",
    "CZE010": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE010/DICOM/00003911/AA8B6291/AA8D4457/0000EDE8",
    "CZE011": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE011/DICOM/00005CB5/AAAA99F9/AA6670D7/0000949B",
    "CZE012": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE012/DICOM/00003CC2/AA022842/AA289D50/00007754",
    "CZE013": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE013/DICOM/0000EA91/AACEC60C/AA7E3964/0000E5BC",
    "CZE014": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE014/DICOM/00005E34/AA79D3B8/AA0DFFCE/0000C950",
    "CZE015": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE015/2de versie/DICOM/00003971/AA8D3DE7/AA73252D/0000D7A8",
    "CZE016": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE016/DICOM/00002765/AAECDDFB/AAE12657/00001053",
    "CZE017": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE017/DICOM/00009314/AA9EE817/AAD307C6/000000A9",
    "CZE018": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE018/DICOM/0000767B/AAC5650C/AADB5D54/0000CAEC",
    "CZE019": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE019/DICOM/00008A57/AA9CD7F4/AAA7440B/00006E56",
    "CZE020": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE020/DICOM/000057A3/AA56DBD9/AA4B0694/00005E78",
    "CZE022": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE022/DICOM/00002E36/AA0DF707/AAE959BA/0000B1D3",
    "CZE023": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE023/DICOM/0000B09F/AA9FDA4D/AA8D0F36/0000F21F",
    "CZE024": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE024/DICOM/0000431B/AA1DAD4A/AAAA0072/0000E6A7",
    "CZE025": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE025/DICOM/0000A32D/AAF725DA/AA6A556D/000058A9",
    "CZE026": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE026/DICOM/00007858/AACE1771/AA3C074D/000033A6",
    "CZE027": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE027/DICOM/000046F3/AA0B28CE/AA933D3E/00007339",
    "CZE029": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE029/DICOM/00001B91/AADD797F/AA3A0E6E/00002DA1",
    "CZE030": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE030/DICOM/000032F8/AAD2875F/AA7CD947/00002E0D",
    "CZE031": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/output_Rachelle/SAVI_Aos_dicom/CZE031/S8010",
    "CZE032": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/output_Rachelle/SAVI_Aos_dicom/CZE032/S8010",
    "CZE033": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/output_Rachelle/SAVI_Aos_dicom/CZE033/S8010",
    "CZE034": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/output_Rachelle/SAVI_Aos_dicom/CZE034/S10010",
    "CZE035": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/output_Rachelle/SAVI_Aos_dicom/CZE035/S8010",
    "CZE036": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/output_Rachelle/SAVI_Aos_dicom/CZE036/S8010",
    "CZE037": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/output_Rachelle/SAVI_Aos_dicom/CZE037/S9010",
    "CZE038": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/output_Rachelle/SAVI_Aos_dicom/CZE038/S10010"
}


# Choose which patient to work with
patient_nr = "CZE001"   # e.g. "aos_14" or "savi_07"

# Automatically load directory
dicom_dir = PATIENT_PATHS[patient_nr]


# %% -------------------------------------------- PREPROCESSING ----------------------------------------
# READING IN THE DICOM & THE EDGE DETECTED IMAGE

# Load the dicom
dicom_reversed = gf.get_sorted_dicom_files(dicom_dir)
dicom = dicom_reversed[::-1]
raw_volume = gf.dicom_to_matrix(dicom)

# Print to confirm dicom loading
print(f"Number of DICOM slices loaded: {len(dicom)}")
print(f"Shape of raw volume: {raw_volume.shape}")

# Extract DICOM attributes, also to reformat the image to HU units
dicom_template = pydicom.dcmread(dicom[0][0])
slope = float(getattr(dicom_template, "RescaleSlope", 1))
intercept = float(getattr(dicom_template, "RescaleIntercept", 0))

# Print slope and intercept to confirm - use print and intercept to calculate the real HU 
print(f"RescaleSlope: {slope}, RescaleIntercept: {intercept}")
raw_volume_hu = raw_volume.astype(np.float32) * slope + intercept

# Print to confirm HU transformation
print(f"Shape of HU volume: {raw_volume_hu.shape}")

# Create an edge detected image based on the imported DICOM
calc_mask = raw_volume_hu > 450
gradient_volume = functions.compute_edge_volume(raw_volume_hu, hu_window=(0, 450), sigma=2, normalize=False, visualize=True)
gradient_volume[calc_mask] = 0
volume = gradient_volume

# Print to confirm gradient image generation
print(f"Shape of edge-detected volume: {gradient_volume.shape}")

# Clipping the DICOM to remove calcifications
low, high = 100, 450  # CHANGED THIS FROM 340 to 500
clipped_dicom = np.clip(raw_volume_hu, low, high)

# Print to confirm clipping
print(f"Min and Max values of clipped DICOM: {clipped_dicom.min()}, {clipped_dicom.max()}")

# Smoothing the image a bit
smoothed_dicom = gaussian_filter(clipped_dicom, sigma=2)  # Note, sigma was changed for patient 11
smoothed_non_clipped = gaussian_filter(raw_volume_hu, sigma=2) # also non-clipped dicom is preserved for calcification retrieval

# Print to confirm smoothing
print(f"Min and Max values of smoothed DICOM: {smoothed_dicom.min()}, {smoothed_dicom.max()}")

# Extract voxel spacing
slice_thickness = functions.compute_slice_spacing(dicom)
pixel_spacing_y = float(dicom_template.PixelSpacing[0])
pixel_spacing_x = float(dicom_template.PixelSpacing[1])
pixel_spacing = (slice_thickness, pixel_spacing_y, pixel_spacing_x)

# Print voxel spacing information
print(f"Voxel spacing: {pixel_spacing}")

dicom_origin = np.array(dicom_template.ImagePositionPatient, dtype=float)

# Print origin
print(f"DICOM origin: {dicom_origin}")

# Prevent memory leakage
del raw_volume_hu
del raw_volume

# %%------------------------------------------REORIENTATION ---------------------------------------

from skimage.transform import rescale

# in this code the the anisotropic vxel is first concerted to isotropic voxel with the pixel spacing
# Next, the image is already upsampled for higher resolution. Results in a better rotation.

# Calculating the vector perpendicualr to the annulus
annular_normal = functions.get_annular_normal(patient_nr)
old_spacing = np.array(pixel_spacing)
target_spacing = pixel_spacing_x

zoom_factors = old_spacing / target_spacing

# Reorient the edge-detected image
rescaled_volume = functions.zoom(volume, zoom_factors)
reoriented_volume, rotation_matrix, rotation_center = functions.reorient_volume(rescaled_volume, 
                                                                                annular_normal, 
                                                                                dicom_origin, 
                                                                                pixel_spacing)

# Lastly, do the reorientation of the non-clipped DICOM 
rescaled_non_clipped_dicom = functions.zoom(smoothed_non_clipped, zoom_factors)
reoriented_non_clipped, rotation_matrix_dicom, rotation_center_dicom_non_clip = functions.reorient_volume(rescaled_non_clipped_dicom,
                                                                                           annular_normal,
                                                                                           dicom_origin,
                                                                                           pixel_spacing)

del rescaled_non_clipped_dicom
del rescaled_volume


# %% ----------------------------------- PLOTTING THE ANNOTATED LANDMARKS FOR ALL CUSPS --------------------------

# Use the _test files if you want to have the recently newly annotated landmarks of 3Dslicer
# Dynamically load the landmark files based on the patient number
landmark_files = {
    "LCC": f"H:/DATA/Afstuderen/3.Data/SSM/patient_database/{patient_nr}/landmarks/lcc_template_landmarks.txt",
    "RCC": f"H:/DATA/Afstuderen/3.Data/SSM/patient_database/{patient_nr}/landmarks/rcc_template_landmarks.txt",
    "NCC": f"H:/DATA/Afstuderen/3.Data/SSM/patient_database/{patient_nr}/landmarks/ncc_template_landmarks.txt"
}   


# Dictionary to store processed landmarks
landmarks_rotated_dict = {}

# Loop over all cusps
for cusp_name, file_path in landmark_files.items():
    print(cusp_name, file_path)
    # 1. Convert from LPS to voxel coordinates
    landmark_voxels = functions.landmarks_to_voxel(file_path, dicom_origin, pixel_spacing)
    print("DICOM Origin", dicom_origin, "Pixel_spacing ", pixel_spacing)
    print("Landmark_voxels: ", landmark_voxels)
    # 2. Reorder axes for numpy indexing (z, y, x)
    landmarks_zyx = landmark_voxels[:, [2, 1, 0]].astype(float)
    
    # 3. Scale according to voxel spacing
    landmarks_scaled = landmarks_zyx.copy()
    landmarks_scaled[:, 0] *= zoom_factors[0]
    landmarks_scaled[:, 1] *= zoom_factors[1]
    landmarks_scaled[:, 2] *= zoom_factors[2]

    
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



# %% ------------------------------------ RETRIEVE ALL OF THE DIFFERENT COORDINATES --------

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


# %% ------------------------------------------- FIND THE WHOLE AORTIC WALL ------------------------

from skimage.segmentation import active_contour
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, dilation, skeletonize
from skimage.filters import threshold_otsu
from skimage.draw import polygon2mask
from skimage.filters import gaussian

# To find the whole aortic wall the active contours method is applied to each slice
# Then, each found wall is accumulated with the previous one to create an aortic valve

# ------------------------------------- INITIALIZING VARIABLES FOR ACTIVE CONTOURS ------------------

alpha = 0.01   # elasticity (snake tension)
beta = 0.1    # rigidity (smoothness)
gamma = 0.01   # step size
total_iterations = 20

z_min_int = int(np.floor(z_min))
z_max_int = int(np.ceil(z_max))
print(f"Processing slices from {z_min_int} to {z_max_int}")

aortic_wall_contours = {}
upsampled_snakes = {}
slice_data = {}   

prev_snake = 0
show_intermediate = False
crop_bool = False

# -------------------------------------- LOOP THROUGH SLICES ---------------------------

for slice_nr in range(z_min_int, z_max_int + 1):
    
    image_pre = reoriented_volume[slice_nr, :, :]
    dicom_slice = reoriented_non_clipped[slice_nr, :, :]

 
    img = image_pre
    img = img / img.max()
    new_image= gaussian(img, sigma = 3)

    # Initialize snake
    if slice_nr == z_min_int:
        snake_current = circle_snake.copy()
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
            w_line=20,
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


    roi_mask = polygon2mask(new_image.shape, snake_current) 
    

    # Store data for *every* slice
    # note that the skeletonization now takes place in a later step
    slice_data[slice_nr] = {
        "roi_mask": roi_mask.copy(),
        "snake": snake_current.copy(),
        "slice": dicom_slice.copy(),
    }

    # Store contour info
    final_snake = snake_current
    aortic_wall_contours[slice_nr] = final_snake
    upsampled_snakes[slice_nr] = snake_current.copy()  
    prev_snake = snake_current


# %% ------------------------------------------ IDENTIFICATION OF CALCIFICATION -------------------------

from skimage.morphology import binary_erosion, disk

# Create an empty mask with the same dimensions as the original image
masked_valve = np.zeros(
    (
        reoriented_non_clipped.shape[0],                     # Z
        reoriented_non_clipped.shape[1],   # Y
        reoriented_non_clipped.shape[2]    # X
    ),
    dtype=reoriented_non_clipped.dtype
)

num_slices = reoriented_non_clipped.shape[0]
print(num_slices)

# Initialize calcification volume
calc_volume = np.zeros_like(masked_valve, dtype=bool)

for slice_nr in range(num_slices):

    # If this slice has stored data, apply ROI
    if slice_nr in slice_data:
        
        # Use the slice data 
        roi_mask= slice_data[slice_nr]["roi_mask"]
        slice_img = slice_data[slice_nr]["slice"]
        
        # The slice on which the threshold is based, is eroded to remove eventuel calcifications on the aortic wall
        # we want to have a clean histogram which is not skewed due to calcifications
        if slice_nr == z_min:
            roi_mask = binary_erosion(roi_mask, disk(3))   
            # Assuming roi_mask is already defined

        # Plot the original roi_mask
        plt.figure(figsize=(12, 6))
        
        # # Original ROI Mask (Before erosion)
        # plt.subplot(1, 2, 1)
        # plt.imshow(roi_mask, cmap='gray')
        # plt.title("ROI Mask Before Erosion")
        # plt.axis('off')    
        # plt.tight_layout()
        # plt.show()
                
        # Extract intensities inside ROI
        roi_values = slice_img[roi_mask > 0]
        
        if roi_values.size == 0:
            median_intensity = np.nan
        else:
            median_intensity = np.median(roi_values)
        
        print(f"Slice {slice_nr}: median intensity inside ROI = {median_intensity:.2f}")
        
        if slice_nr==z_min: 
            # Threshold for calcification: e.g., mean + 3*std
            calc_threshold =  np.percentile(roi_values, 99)  # upper bound used as threshold   
            print(f"Patient {patient_nr}: Threshold for calcification  = {calc_threshold:.2f}")
        
        # Print a histogram - necessary for seperating the calcification from the aortic valve
        print_histo = False
        if print_histo == True:
            plt.clf()
            plt.hist(roi_values.flatten(), bins=40, color='skyblue', edgecolor='black')
            plt.title(f"Slice {slice_nr} – ROI intensity histogram")
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Add vertical dashed line for threshold
            plt.axvline(calc_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {calc_threshold:.1f}')
            plt.legend()
            plt.pause(0.2)  # scroll speed
            
        

        # Safety check (recommended)
        if roi_mask.shape != slice_img.shape:
            raise ValueError(
                f"ROI mask shape {roi_mask.shape} does not match stored slice shape "
                f"{slice_img.shape} at slice {slice_nr}"
            )

        # Apply mask
        masked_valve[slice_nr] = slice_img * roi_mask
        
        # --- Calcification segmentation ---
        calc_mask = np.zeros_like(slice_img, dtype=bool)
        calc_mask[roi_mask > 0] = slice_img[roi_mask > 0] > calc_threshold
        calc_volume[slice_nr] = calc_mask  # store in 3D volume

    # Otherwise leave slice empty
    else:
        masked_valve[slice_nr] = 0

# # (optional) Visualization fo the ROI
for z in range(z_min, z_max + 1):
    if slice_nr not in slice_data:
        print(f"Warning: Slice {slice_nr} not found in slice_data")
    if z not in slice_data:
        continue

    plt.clf()

    # Show the original slice (masked valve)
    slice_img = masked_valve[z]  # or slice_data[z]["slice"] if you prefer original
    plt.imshow(slice_img, cmap="gray")

    # Overlay calcification in red using contour
    plt.contour(calc_volume[z], colors='red', linewidths=1)

    plt.title(f"Slice {z} – Calcification overlay")
    plt.axis("off")
    plt.pause(0.15)

plt.show()



# %%-------------------------------  EXTRACTION OF THE LEAFET BOUNDARIES ---------------------------

from skimage.morphology import binary_erosion
from scipy.ndimage import gaussian_filter


# Parameters
alpha, beta, gamma = 0.1, 0.1, 0.01
total_iterations = 20
BOUNDARY_THRESHOLD = 25  # can change later

# Leaflet-specific storage
LCC_data, RCC_data, NCC_data = {}, {}, {}

# Retrieval and upscaling of points
hinge_lcc = lcc_rotated[2][1:3]
hinge_rcc = rcc_rotated[2][1:3] 
hinge_ncc = ncc_rotated[2][1:3]

rcc_lcc_com = lcc_com[1:3] 
lcc_ncc_com = ncc_com[1:3] 
ncc_rcc_com = rcc_com[1:3] 

center = lcc_rotated[3][1:3]
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

# Boundary shrink detection state
boundary_event_state = {
    "lcc_ncc": {"was_above": False, "event_slice": None},
    "rcc_lcc": {"was_above": False, "event_slice": None},
    "ncc_rcc": {"was_above": False, "event_slice": None},
}

commissure_points = [rcc_lcc_com, lcc_ncc_com, ncc_rcc_com]
mercedes_star = False

# Loop over slices
for slice_nr, slice_info in slice_data.items():
    print("----------------------------------------")
    print(f"Processing slice {slice_nr}...")

    # --- Upscale slice for visualization / mask ---
    reoriented_slice = reoriented_non_clipped[slice_nr, :, :]
    roi_mask = slice_info["roi_mask"]

    # --- Active contour refinement ---
    init_snake = slice_info["snake"].copy()
    snake_current = functions.resample_closed_contour(init_snake)
    
    
    # --------- CONTEXT DEPENDENT CLIPPING  -------
    # Updated: now we are using the previously calculated eroded_foreground, i.e. the calculated ROI
    # So the whole skeletonization which took place in the previous loop will now be done. This is done
    # so that the context-dependent clipping can be achieved.
    
    
    # Check if commissure points from the previous slice are available
    commissure_points = {
        'lcc_ncc': prev_intersections['lcc_ncc'] if prev_intersections['lcc_ncc'] is not None else base_commissure['lcc_ncc'],
        'rcc_lcc': prev_intersections['rcc_lcc'] if prev_intersections['rcc_lcc'] is not None else base_commissure['rcc_lcc'],
        'ncc_rcc': prev_intersections['ncc_rcc'] if prev_intersections['ncc_rcc'] is not None else base_commissure['ncc_rcc']
    }
    
    init_coms = [
        commissure_points['lcc_ncc'],
        commissure_points['rcc_lcc'],
        commissure_points['ncc_rcc']
    ]
    
    init_mercedes = functions.create_mercedes_mask(reoriented_slice, init_coms, center, line_thickness = 8)
    slice_clipped = reoriented_slice.copy()
        
    # 1) Low clipping only inside ROI
    slice_clipped[(slice_clipped < 100) & (roi_mask > 0)] = 100
    
    # 2) High clipping inside mercedes: >450 -> 0
    slice_clipped[(init_mercedes == 1) & (slice_clipped > calc_threshold)] = 100
    
    # 3) High clipping outside mercedes: >450 -> 450
    slice_clipped[(init_mercedes == 0) & (slice_clipped > calc_threshold)] = 450
    
    # Check for any zero pixels
    num_zeros = np.sum(slice_clipped == 0)


     # ----------------------------------------- SKELETONIZATION  ----------------------------------------
    # Apply Gaussian blur to the reoriented slice (sigma controls the blur intensity)
    sigma = 0  # Adjust this value based on how much blur you want
    blurred_slice = gaussian(slice_clipped, sigma=sigma)
    
    # Multiply the blurred image with the ROI mask
    roi_image_blurred = blurred_slice * roi_mask.astype(slice_clipped.dtype)
    
    # Multiply the original (non-blurred) image with the ROI mask
    roi_image_original = slice_clipped * roi_mask.astype(slice_clipped.dtype)
   
    # Initialize the threshold variable for all slices
    if slice_nr == z_min:
        # Calculate the threshold for the first slice based on the 10th percentile
        roi_pixels_blurred = roi_image_blurred[roi_mask]  # Use blurred ROI pixels
        percentile_10th = np.percentile(roi_pixels_blurred, 10)  # 15th percentile as threshold
        # print(f"Calculated 10th Percentile Threshold for Slice {slice_nr}: {percentile_10th:.2f}")
    else:
        # Use the previously calculated threshold for all other slices
        percentile_10th = globals().get("percentile_10th", 0)
    
    # Apply the 10th percentile threshold to the blurred image
    percentile_mask = (roi_image_blurred > percentile_10th) & roi_mask
    
    # Visualize the histogram of the intensity values of the blurred image
    plot_histo = False
    
    if plot_histo == True: 
        plt.figure(figsize=(8, 6))
        plt.hist(roi_pixels_blurred, bins=50, color='gray', alpha=0.7)
        plt.title("Histogram of ROI Intensity Values (Blurred)")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        
        # Mark the 10th percentile threshold on the histogram
        plt.axvline(percentile_10th, color='g', linestyle='--', label=f'10th Percentile Threshold: {percentile_10th:.2f}')
        plt.legend()
        plt.show()

    # Apply the mask based on the 10th percentile threshold
    inverted_mask = ~percentile_mask
    closed_inverted = dilation(inverted_mask, disk(1))
    eroded_foreground = closed_inverted * roi_mask
    
    # Perform skeletonization and thickening
    skeleton = skeletonize(eroded_foreground)
    thicker_skeleton = dilation(skeleton, disk(3))
    
    # Final visualization of the skeletonized result
    # plt.figure(figsize=(6, 6))
    # plt.imshow(blurred_slice, cmap='gray')
    # plt.imshow(thicker_skeleton, cmap='Reds', alpha=0.0)
    # # Mercedes star overlay
    # plt.title(f"Skeletonization with Percentile Threshold — Slice {slice_nr}")
    # plt.axis("off")
    # plt.show()
    
    # Optionally, store the threshold for future slices
    if slice_nr == z_min:
        globals()["percentile_10th"] = percentile_10th  # Store the threshold for future slices


    # --------------------------------------------------FINDIGN BOUNDARIES -----------------------
    
    
    # Preparing image for active contours    
    blurred_skeleton_norm = gaussian_filter(skeleton.astype(float), sigma=7)
    blurred_skeleton_norm /= blurred_skeleton_norm.max() + 1e-8

    # The snake is rerun, to close off the entire aortic wall. Previous snake could have had holes in it
    # However update: since 20-01-2026 not using this contour anymore for aortic wall, as it was too small
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
    skeleton_mask = polygon2mask(skeleton.shape, snake_current)
    roi_mask_eroded = binary_erosion(skeleton_mask)
    inner_skeleton = skeleton * roi_mask_eroded.astype(skeleton.dtype)

    # Upsample contour for accurate indices
    aortic_wall_contour = functions.resample_closed_contour(init_snake)

    # Find closest points to hinge points
    i_lcc = functions.closest_contour_point(lcc_rotated[2][1:3], aortic_wall_contour)
    i_rcc = functions.closest_contour_point(rcc_rotated[2][1:3], aortic_wall_contour)
    i_ncc = functions.closest_contour_point(ncc_rotated[2][1:3], aortic_wall_contour)

    # Create contour segments
    seg_lcc_ncc = functions.contour_segment(aortic_wall_contour, i_lcc, i_ncc)
    seg_ncc_rcc = functions.contour_segment(aortic_wall_contour, i_ncc, i_rcc)
    seg_rcc_lcc = functions.contour_segment(aortic_wall_contour, i_rcc, i_lcc)

    # Create leaflet masks and boundaries
    lcc_ncc_mask, lcc_ncc_boundary = functions.create_boundary_mask(
        center, aortic_wall_contour, i_lcc, i_ncc, seg_lcc_ncc, slice_clipped, inner_skeleton)
    rcc_lcc_mask, rcc_lcc_boundary = functions.create_boundary_mask(
        center, aortic_wall_contour, i_rcc, i_lcc, seg_rcc_lcc, slice_clipped, inner_skeleton)
    ncc_rcc_mask, ncc_rcc_boundary = functions.create_boundary_mask(
        center, aortic_wall_contour, i_ncc, i_rcc, seg_ncc_rcc, slice_clipped, inner_skeleton)

    # Check if commissure points from the previous slice are available
    commissure_points = {
        'lcc_ncc': prev_intersections['lcc_ncc'] if prev_intersections['lcc_ncc'] is not None else base_commissure['lcc_ncc'],
        'rcc_lcc': prev_intersections['rcc_lcc'] if prev_intersections['rcc_lcc'] is not None else base_commissure['rcc_lcc'],
        'ncc_rcc': prev_intersections['ncc_rcc'] if prev_intersections['ncc_rcc'] is not None else base_commissure['ncc_rcc']
    }
    print("commissure_points: ", commissure_points, "init_coms: ", init_coms, center)
    com_points = [commissure_points['lcc_ncc'], commissure_points['rcc_lcc'], commissure_points['ncc_rcc']]

    # Clean boundaries with the commissure points (from previous slice or base)
    cleaned_lcc_ncc_boundary, mercedes_mask = functions.clean_boundary_from_mask(
        lcc_ncc_boundary, aortic_wall_contour, 3, com_points, center)
    cleaned_rcc_lcc_boundary, mercedes_mask  = functions.clean_boundary_from_mask(
        rcc_lcc_boundary, aortic_wall_contour, 3, com_points, center)
    cleaned_ncc_rcc_boundary, mercedes_mask = functions.clean_boundary_from_mask(
        ncc_rcc_boundary, aortic_wall_contour, 3, com_points, center)

    # --- Find intersections with wall ---
    intersections = {}
    if slice_nr != z_min:
        intersections = functions.find_all_boundary_intersections(
            slice_clipped,
            seg_lcc_ncc,
            seg_rcc_lcc,
            seg_ncc_rcc,
            cleaned_lcc_ncc_boundary,
            cleaned_rcc_lcc_boundary,
            cleaned_ncc_rcc_boundary,
            slice_nr,
            mercedes_mask,
            plot=True
            )
    
    commissure_names = ["lcc_ncc", "rcc_lcc", "ncc_rcc"]
    commissure_indices = {}
    
    # --------------------------------------------
    # 1. Check if we have EVER found all 3 before
    # --------------------------------------------
    have_full_prev = all(prev_intersections[name] is not None for name in commissure_names)
    
    # --------------------------------------------
    # 2. Decide fallback strategy for THIS slice
    # --------------------------------------------
    found_this_slice = {name: intersections.get(name) is not None for name in commissure_names}
        
    # Case A — All 3 found this slice → use them & update prev_intersections
    if all(found_this_slice.values()):
        # print(f"Slice {slice_nr}: All 3 intersections found. Using them and updating previous intersections.")
        for name in commissure_names:
            # print(f"  Using intersection for {name} from this slice.")
            prev_intersections[name] = intersections[name]
            commissure_indices[name] = functions.closest_contour_point(
                intersections[name],
                aortic_wall_contour
            )
    
    # Case B — Some intersections missing, use available ones and previous for missing ones
    elif have_full_prev:
        # print(f"Slice {slice_nr}: Missing intersections, but previous intersections are available. Using previous intersections.")
        for name in commissure_names:
            if intersections.get(name) is not None:  # If intersection is found, use it
                print(f"  Using intersection for {name} from this slice.")
                commissure_indices[name] = functions.closest_contour_point(
                    intersections[name],
                    aortic_wall_contour
                )
            else:  # Otherwise, use previous intersection
                # print(f"  Using previous intersection for {name}.")
                commissure_indices[name] = functions.closest_contour_point(
                    prev_intersections[name],
                    aortic_wall_contour
                )
    
    # Case C — Early slices where we do NOT have all 3 prev intersections yet
    # → fall back per-intersection to base commissures
    else:
        # print(f"Slice {slice_nr}: Not all intersections found and no full previous set. Falling back to base commissures for missing intersections.")
        for name in commissure_names:
            # Print the fallback logic for each commissure
            fallback = intersections.get(name) or prev_intersections.get(name) or base_commissure[name]
            if intersections.get(name) is not None:
                print(f"  Using intersection for {name} found in this slice.")
            elif prev_intersections.get(name) is not None:
                print(f"  Using previous intersection for {name}.")
            else:
                print(f"  Using base commissure for {name} (no previous or current intersection).")
            
            commissure_indices[name] = functions.closest_contour_point(
                fallback,
                aortic_wall_contour
            )

    # Create COM-to-COM segments
    seg_c2c = {
        "LCC": functions.contour_segment(aortic_wall_contour, commissure_indices["lcc_ncc"], commissure_indices["rcc_lcc"]),
        "RCC": functions.contour_segment(aortic_wall_contour, commissure_indices["ncc_rcc"], commissure_indices["rcc_lcc"]),
        "NCC": functions.contour_segment(aortic_wall_contour, commissure_indices["lcc_ncc"], commissure_indices["ncc_rcc"])
    }
    # Helper function
    def safe_copy(point, fallback=None):
        if point is None:
            return fallback
        else:
            return np.array(point)  # copy as array
    
    # ----------------- EVENT DETECTION ------------------
    # This segment calculates when the boundary becomes smaller than the set threshold, but was once larger than this threshold
    length_lcc = functions.update_boundary_shrink_state(
    cleaned_lcc_ncc_boundary,
    boundary_event_state,
    "lcc_ncc",
    slice_nr,
    threshold=BOUNDARY_THRESHOLD
    )
    
    length_rcc = functions.update_boundary_shrink_state(
        cleaned_rcc_lcc_boundary,
        boundary_event_state,
        "rcc_lcc",
        slice_nr,
        threshold=BOUNDARY_THRESHOLD
    )
    
    length_ncc = functions.update_boundary_shrink_state(
        cleaned_ncc_rcc_boundary,
        boundary_event_state,
        "ncc_rcc",
        slice_nr,
        threshold=BOUNDARY_THRESHOLD
    )
    
    # Store results with flat keys safely
    LCC_data[slice_nr] = {
        "mask": lcc_ncc_mask.copy(),
        "lcc_ncc_boundary": cleaned_lcc_ncc_boundary.copy(),
        "rcc_lcc_boundary": cleaned_rcc_lcc_boundary.copy(),
        "com_to_com": seg_c2c["LCC"].copy(),
        "aortic_wall_contour": aortic_wall_contour.copy(),
        "lcc_ncc_com": safe_copy(intersections.get("lcc_ncc"),prev_intersections.get("lcc_ncc") or base_commissure["lcc_ncc"]),
        "rcc_lcc_com": safe_copy(intersections.get("rcc_lcc"), prev_intersections.get("rcc_lcc") or base_commissure["rcc_lcc"]),
        "height": slice_nr,
        "boundary_length": length_lcc,
        "shrink_event_slice": boundary_event_state["lcc_ncc"]["event_slice"],
    }
    
    RCC_data[slice_nr] = {
        "mask": rcc_lcc_mask.copy(),
        "rcc_lcc_boundary": cleaned_rcc_lcc_boundary.copy(),
        "ncc_rcc_boundary": cleaned_ncc_rcc_boundary.copy(),
        "com_to_com": seg_c2c["RCC"].copy(),
        "rcc_lcc_com": safe_copy(intersections.get("rcc_lcc"), prev_intersections.get("rcc_lcc") or base_commissure["rcc_lcc"]),
        "ncc_rcc_com": safe_copy(intersections.get("ncc_rcc"), prev_intersections.get("ncc_rcc") or base_commissure["ncc_rcc"]),
        "height": slice_nr,
        "boundary_length": length_rcc,
        "shrink_event_slice": boundary_event_state["rcc_lcc"]["event_slice"],
    }
    
    NCC_data[slice_nr] = {
        "mask": ncc_rcc_mask.copy(),
        "ncc_rcc_boundary": cleaned_ncc_rcc_boundary.copy(),
        "lcc_ncc_boundary": cleaned_lcc_ncc_boundary.copy(),
        "com_to_com": seg_c2c["NCC"].copy(),
        "ncc_rcc_com": safe_copy(intersections.get("ncc_rcc"), prev_intersections.get("ncc_rcc") or base_commissure["ncc_rcc"]),
        "lcc_ncc_com": safe_copy(intersections.get("lcc_ncc"), prev_intersections.get("lcc_ncc") or base_commissure["lcc_ncc"]),
        "height": slice_nr,
        "boundary_length": length_ncc,
        "shrink_event_slice": boundary_event_state["ncc_rcc"]["event_slice"],
    }
    
    # Optional visualization (COM-to-COM segments + boundaries)
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_clipped, cmap='gray', origin='upper')
    plt.imshow(mercedes_mask, cmap='jet', alpha=0.3)  # Adjust alpha for transparency
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



# %% SEGMENT THE CALCIFICATION INTO LEAFLET_SPECIFIC 

# In this segment the calcification will be split into three different regions based on the a priori knowledge and the found boudnaries



# %% ----------------EXTRACT LEAFLET-SPECIFIC CENTER HEIGHTS ---------------

# The shrink ratio is where we say that the boundary has shrunk this much that we will argument that the mercedes star is there.
# this is the height on which the leaflet cap latches on
shrink_ratio = 0.9

lcc_event, lcc_thresh, lcc_max_slice = functions.find_shrink_slice_consecutive(LCC_data, shrink_ratio = shrink_ratio)
rcc_event, rcc_thresh, rcc_max_slice = functions.find_shrink_slice_consecutive(RCC_data, shrink_ratio = shrink_ratio)
ncc_event, ncc_thresh, ncc_max_slice = functions.find_shrink_slice_consecutive(NCC_data, shrink_ratio = shrink_ratio)

print("LCC_NCC max at:", lcc_max_slice, "shrink at:", lcc_event)
print("RCC_LCC max at:", rcc_max_slice, "shrink at:", rcc_event)
print("NCC_RCC max at:", ncc_max_slice, "shrink at:", ncc_event)

lcc_ncc_height = lcc_event
rcc_lcc_height = rcc_event
ncc_rcc_height = ncc_event



# %% EXTRACT LEAFLET SPECIFIC WHEN THE MERCEDES STAR OCCURS. THIS IS NECESSARY TO MAKE 3 SEPERATE REGIONS WITHIN THE ROI

import importlib
import functions
importlib.reload(functions)

lcc_max_slice, lcc_max_len = functions.find_max_length_slice(LCC_data)
rcc_max_slice, rcc_max_len = functions.find_max_length_slice(RCC_data)
ncc_max_slice, ncc_max_len = functions.find_max_length_slice(NCC_data)

print("Max slices:")
print("LCC:", lcc_max_slice)
print("RCC:", rcc_max_slice)
print("NCC:", ncc_max_slice)

mercedes_height = int(round((
    lcc_max_slice +
    rcc_max_slice +
    ncc_max_slice
) / 3))

print("Mercedes slice (average):", mercedes_height)

#LCC
data = LCC_data[mercedes_height]
wall_lcc = data["com_to_com"]
boundary_lcc_ncc = data["lcc_ncc_boundary"]
com_lcc_ncc = data["lcc_ncc_com"]
com_rcc_lcc = data["rcc_lcc_com"]

#RCC
data = RCC_data[mercedes_height]
wall_rcc = data["com_to_com"]
boundary_rcc_lcc = data["rcc_lcc_boundary"]
com_rcc_lcc = data["rcc_lcc_com"]
com_ncc_rcc = data["ncc_rcc_com"]

#NCC
data = NCC_data[mercedes_height]
wall_ncc = data["com_to_com"]
boundary_ncc_rcc = data["ncc_rcc_boundary"]
com_ncc_rcc = data["ncc_rcc_com"]
com_lcc_ncc = data["lcc_ncc_com"]

# Grab the slice shape from your reoriented data
slice_shape = reoriented_non_clipped[mercedes_height].shape

# Build the NCC leaflet mask
ncc_region_mask = functions.build_leaflet_mask(
    slice_shape=slice_shape,
    wall_coords=wall_ncc,       # the wall connecting the two commissures
    comA=com_lcc_ncc,         # starting commissure on the wall
    comB=com_ncc_rcc,         # ending commissure on the wall
    center=center        # leaflet center point
)

# Build the LCC leaflet mask
lcc_region_mask = functions.build_leaflet_mask(
    slice_shape=slice_shape,
    wall_coords=wall_lcc,       # the wall connecting the two commissures
    comA=com_lcc_ncc,         # starting commissure on the wall
    comB=com_rcc_lcc,         # ending commissure on the wall
    center=center         # leaflet center point
)

# Build the RCC leaflet mask
rcc_region_mask = functions.build_leaflet_mask(
    slice_shape=slice_shape,
    wall_coords=wall_rcc,       # the wall connecting the two commissures
    comA=com_ncc_rcc,         # starting commissure on the wall
    comB=com_rcc_lcc,         # ending commissure on the wall
    center=center         # leaflet center point
)




# -------------------------------
# Compute areas
# -------------------------------

pixel_area_mm2 = target_spacing ** 2

# Compute areas
ncc_area_mm2 = np.count_nonzero(ncc_region_mask) * pixel_area_mm2
lcc_area_mm2 = np.count_nonzero(lcc_region_mask) * pixel_area_mm2
rcc_area_mm2 = np.count_nonzero(rcc_region_mask) * pixel_area_mm2

print(f"NCC area: {ncc_area_mm2:.2f} mm²")
print(f"LCC area: {lcc_area_mm2:.2f} mm²")
print(f"RCC area: {rcc_area_mm2:.2f} mm²")


# Overlay the leaflet masks with transparency and different colors
plt.imshow(ncc_region_mask, cmap='Reds', alpha=0.4, label='NCC')
plt.imshow(lcc_region_mask, cmap='Greens', alpha=0.4, label='LCC')
plt.imshow(rcc_region_mask, cmap='Blues', alpha=0.4, label='RCC')

plt.title(f"Aortic Leaflet Masks - Slice {mercedes_height}")
plt.axis('off')
plt.show()




# %% CALCULATE THE PERIMETER TO CREATE BULLS


total_area = ncc_area_mm2 + lcc_area_mm2 + rcc_area_mm2
total_area_pixels = total_area/pixel_area_mm2
print(f"Total valve area: {total_area:.2f} mm2")

# Compute radius to cover one-third of total valve area
radius_one_third_area = np.sqrt((1/4) * total_area_pixels / np.pi)
print(f"Radius to cover one-third of the area: {radius_one_third_area:.2f} pixels")

# Build mask
yy, xx = np.meshgrid(np.arange(slice_shape[0]), np.arange(slice_shape[1]), indexing='ij')
dist_from_center = np.sqrt((yy - center[0])**2 + (xx - center[1])**2)

center_one_third_area_mask = dist_from_center <= radius_one_third_area

# Visualize
plt.figure(figsize=(6,6))
plt.imshow(slice_img, cmap='gray')
plt.imshow(center_one_third_area_mask, cmap='Reds', alpha=0.4)
plt.scatter(center[1], center[0], c='blue', marker='x')
plt.title(f"One-third area circular mask")
plt.axis('off')
plt.show()

# Calculate mask area in pixels
mask_area = np.count_nonzero(center_one_third_area_mask)
print(f"Mask area: {mask_area} pixels^2")

# Fraction of total valve area
fraction_covered = mask_area / total_area_pixels
print(f"Fraction of valve area covered: {fraction_covered:.2f}")



#%% CREATE THE MASKS FOR CENTRAL AND PERIPHERAL


# -------------------------------
# Central and peripheral NCC (one-third)
# -------------------------------
central_ncc_mask = ncc_region_mask & center_one_third_area_mask       # intersection
peripheral_ncc_mask = ncc_region_mask & (~center_one_third_area_mask) # remaining part

# -------------------------------
# Central and peripheral LCC (one-third)
# -------------------------------
central_lcc_mask = lcc_region_mask & center_one_third_area_mask
peripheral_lcc_mask = lcc_region_mask & (~center_one_third_area_mask)

# -------------------------------
# Central and peripheral RCC (one-third)
# -------------------------------
central_rcc_mask = rcc_region_mask & center_one_third_area_mask
peripheral_rcc_mask = rcc_region_mask & (~center_one_third_area_mask)

# -------------------------------
# Visualize all combinations
# -------------------------------
plt.figure(figsize=(6,6))
plt.imshow(slice_img, cmap='gray')

plt.imshow(central_ncc_mask, cmap='Reds', alpha=0.5)
plt.imshow(peripheral_ncc_mask, cmap='Reds', alpha=0.2)

plt.imshow(central_lcc_mask, cmap='Greens', alpha=0.5)
plt.imshow(peripheral_lcc_mask, cmap='Greens', alpha=0.2)

plt.imshow(central_rcc_mask, cmap='Blues', alpha=0.5)
plt.imshow(peripheral_rcc_mask, cmap='Blues', alpha=0.2)

plt.scatter(center[1], center[0], c='yellow', marker='x')
plt.title(f"Leaflet Central vs Peripheral Regions - Slice {mercedes_height}")
plt.axis('off')
plt.show()

# -------------------------------
# Save areas to file
# -------------------------------

area_dict = {
    "patient": patient_nr,
    "NCC_area_mm2": ncc_area_mm2,
    "LCC_area_mm2": lcc_area_mm2,
    "RCC_area_mm2": rcc_area_mm2
}

output_path = f"H:/DATA/Afstuderen/3.Data/output_valve_segmentation/{patient_nr}/patient_space"
output_file = os.path.join(output_path, f"{patient_nr}_region_areas.txt")

with open(output_file, "w") as f:
    for key, value in area_dict.items():
        f.write(f"{key}: {value}\n")

print(f"Saved region areas to: {output_file}")

