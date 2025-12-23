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
from scipy.ndimage import binary_erosion


# %% ----------------------------------------PATIENT PATHS & SELECTION OF PATIENT -----------------------------

PATIENT_PATHS = {
    # AoSstress patients
    "aos_2":  r"T:/Research_01/CZE-2020.67 - SAVI-AoS/AoS stress/CT/Aosstress02/DICOM/00002C38/AA3D97B3/AA3B5B73/000062B4",
    "aos_5":  r"T:/Research_01/CZE-2020.67 - SAVI-AoS/Aos stress/CT/Aosstress05/DICOM/0000CD6B/AA3BA81C/AAADD92A/0000C27F",
    "aos_6":  r"T:/Research_01/CZE-2020.67 - SAVI-AoS/Aos stress/CT/Aosstress06/DICOM/00008A58/AA245852/AA659577/0000A0F7",
    "aos_8":  r"T:/Research_01/CZE-2020.67 - SAVI-AoS/Aos stress/CT/Aosstress08/DICOM/0000A996/AA934448/AA0D3303/0000F9C1",
    "aos_9":  r"T:/Research_01/CZE-2020.67 - SAVI-AoS/Aos stress/CT/Aosstress09/DICOM/0000B2D9/AA876FC8/AABB814D/0000534F",
    "aos_11": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/Aos stress/CT/Aosstress11/DICOM/00006310/AAD9B219/AA824679/00004F79",
    "aos_12": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/Aos stress/CT/Aosstress12/DICOM/0000B416/AABF5153/AA9CA582/0000799A",
    "aos_13": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/Aos stress/CT/Aosstress13/DICOM/0000208C/AABDE934/AA243C5D/00002411",
    "aos_14": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/Aos stress/CT/Aosstress14/DICOM/000037EC/AA4EC564/AA3B0DE6/00007EA9",
    "aos_15": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/Aos stress/CT/Aosstress15/DICOM/00007464/AA714246/AA1B4F2E/00008A1B",

    # SAVI AoS patients
    "savi_01": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE001/DICOM/00003852/AA44D04F/AA7BB8C5/000050B5",
    "savi_02": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE002/DICOM/0000AFC5/AAAA2796/AAFF16B0/0000ADA6",
    "savi_03": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE003/DICOM/0000AF6A/AA4272CE/AA72A45E/000050F2",
    "savi_04": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE004/DICOM/00002F76/AA1F4542/AAB1E4E9/0000CAC5",
    "savi_05": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE005/DICOM/0000AF52/AA590C3F/AAC428CF/0000FEE0",
    "savi_06": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE006/DICOM/00000EED/AA87381C/AAAEC035/00002581",
    "savi_07": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE007/DICOM/000065F6/AAB95BAE/AA7A7E4C/00005896",
    "savi_08": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE008/DICOM/000053DF/AA102722/AA7E9491/000073C6",
    "savi_10": r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE010/DICOM/00003911/AA8B6291/AA8D4457/0000EDE8",
}

# Choose which patient to work with
patient_nr = "savi_01"   # e.g. "aos_14" or "savi_07"

# Automatically load directory
dicom_dir = PATIENT_PATHS[patient_nr]



# %% -------------------------------------------- PREPROCESSING ----------------------------------------
# READING IN THE DICOM & THE EDGE DETECTED IMAGE

from scipy.ndimage import zoom, gaussian_filter
import pydicom
import numpy as np

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

# Print slope and intercept to confirm
print(f"RescaleSlope: {slope}, RescaleIntercept: {intercept}")

raw_volume_hu = raw_volume.astype(np.float32) * slope + intercept

# Print to confirm HU transformation
print(f"Shape of HU volume: {raw_volume_hu.shape}")

# Create an edge detected image based on the imported DICOM
gradient_volume = functions.compute_edge_volume(raw_volume_hu, hu_window=(0, 450), sigma=2, normalize=False, visualize=True)
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



# %%------------------------------------------REORIENTATION ---------------------------------------
from skimage.transform import rescale

# in this code the the anisotropic vxel is first concerted to isotropic voxel with the pixel spacing
# Next, the image is already upsampled for higher resolution. Results in a better rotation.

# Calculating the vector perpendicualr to the annulus
annular_normal = functions.get_annular_normal(patient_nr)
scale_factor = 4

# Reorient the edge-detected image
rescaled_volume = functions.zoom(volume, pixel_spacing)
upsampled_volume = rescale(rescaled_volume, scale_factor, order=3, preserve_range=True, anti_aliasing=True).astype(rescaled_volume.dtype)
reoriented_volume, rotation_matrix, rotation_center = functions.reorient_volume(upsampled_volume, 
                                                                                annular_normal, 
                                                                                dicom_origin, 
                                                                                pixel_spacing)

# Now als do the reorientation on the regular DICOM
rescaled_dicom = functions.zoom(smoothed_dicom, pixel_spacing)
upsampled_dicom = rescale(rescaled_dicom, scale_factor, order=3, preserve_range=True, anti_aliasing=True).astype(rescaled_dicom.dtype)
reoriented_dicom, rotation_matrix_dicom, rotation_center_dicom = functions.reorient_volume(upsampled_dicom,
                                                                                           annular_normal,
                                                                                           dicom_origin,
                                                                                           pixel_spacing)

# Lastly, do the reorientation of the non-clipped DICOM 
rescaled_non_clipped_dicom = functions.zoom(smoothed_non_clipped, pixel_spacing)
upsampled_non_clipped_dicom = rescale(rescaled_non_clipped_dicom, scale_factor, order=3, preserve_range=True, anti_aliasing=True).astype(rescaled_non_clipped_dicom.dtype)
reoriented_non_clipped, rotation_matrix_dicom_non_clip, rotation_center_dicom_non_clip = functions.reorient_volume(upsampled_non_clipped_dicom,
                                                                                           annular_normal,
                                                                                           dicom_origin,
                                                                                           pixel_spacing)




# %% ----------------------------------- PLOTTING THE ANNOTATED LANDMARKS FOR ALL CUSPS --------------------------

# Use the _test files if you want to have the recently newly annotated landmarks of 3Dslicer
# Dynamically load the landmark files based on the patient number
landmark_files = {
    "LCC": f"H:/DATA/Afstuderen/3.Data/SSM/patient_database/{patient_nr}/landmarks/lcc_template_landmarks.txt",
    "RCC": f"H:/DATA/Afstuderen/3.Data/SSM/patient_database/{patient_nr}/landmarks/rcc_template_landmarks.txt",
    "NCC": f"H:/DATA/Afstuderen/3.Data/SSM/patient_database/{patient_nr}/landmarks/ncc_template_landmarks.txt"
}   

# I removed test from this!

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
    landmarks_scaled[:, 0] *= slice_thickness * scale_factor  # z-axis
    landmarks_scaled[:, 1] *= pixel_spacing_y  * scale_factor # y-axis
    landmarks_scaled[:, 2] *= pixel_spacing_x * scale_factor # x-axis

    
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



# %%% ------------------------------------------ PLOTTING SINGLE SLICE ---------------------------

from skimage import exposure

# Slice index (X)
slice_idx = 50

# Extract the transversal slice
transversal_slice = reoriented_dicom[slice_idx, :, :]

# Plot the slice
plt.figure(figsize=(6,6))
plt.imshow(transversal_slice, cmap="gray")
plt.title(f"Transversal slice at x={slice_idx}")
plt.axis("off")

count = 0
# Overlay landmarks that lie on this slice
for z, y, x in circle_points:
    if z == slice_idx:  # Only plot landmarks on this slice
        plt.scatter(x, y, c='r', alpha=0.7,s=2)  # z → horizontal, y → vertical
        count+=1

    
print(count)
plt.show()



#%%  ------------------------------------------- PLOTTING ENTIRE FIGURE  ------------------------------

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
    plt.pause(0.1)  # pause 2 seconds
    
plt.close()

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
crop_bool = False

# -------------------------------------- LOOP THROUGH SLICES ---------------------------

for slice_nr in range(z_min_int, z_max_int + 1):
    
    image_pre = reoriented_volume[slice_nr, :, :]
    dicom_slice = reoriented_non_clipped[slice_nr, :, :]

    # Upsample both the edge detected image as well as the regular dicom - not needed anymore i think, since upsampling is done priorly
    # gradient_upsampled = rescale(image_pre, scale_factor, order=3, preserve_range=True, anti_aliasing=True).astype(image_pre.dtype)
    # dicom_upsampled = rescale(dicom_slice, scale_factor, order=3, preserve_range=True, anti_aliasing=True).astype(dicom_slice.dtype)
    
    img = image_pre
    img = img / img.max()
    
    # yesequalization
    # new_image = exposure.equalize_adapthist(img, clip_limit=0.03)
    new_image= img

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
    
    # Apply Gaussian blur to the reoriented slice (sigma controls the blur intensity)
    sigma = 3  # Adjust this value based on how much blur you want
    blurred_slice = gaussian(reoriented_slice_dicom, sigma=sigma)
    
    # Multiply the blurred image with the ROI mask
    roi_image_blurred = blurred_slice * roi_mask.astype(reoriented_slice_dicom.dtype)
    
    # Multiply the original (non-blurred) image with the ROI mask
    roi_image_original = reoriented_slice_dicom * roi_mask.astype(reoriented_slice_dicom.dtype)
    
    # Plot both images side by side for comparison
    plt.figure(figsize=(12, 6))
    
    # Original ROI image (without blur)
    plt.subplot(1, 2, 1)
    plt.imshow(roi_image_original, cmap='gray')
    plt.title("ROI Image (No Blur)")
    plt.axis("off")
    
    # Blurred ROI image
    plt.subplot(1, 2, 2)
    plt.imshow(roi_image_blurred, cmap='gray')
    plt.title("ROI Image (With Blur)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------- PROCESSING ---------------------------------------
    
    # Initialize the threshold variable for all slices
    if slice_nr == z_min:
        # Calculate the threshold for the first slice based on the 10th percentile
        roi_pixels_blurred = roi_image_blurred[roi_mask]  # Use blurred ROI pixels
        percentile_10th = np.percentile(roi_pixels_blurred, 10)  # 10th percentile as threshold
        print(f"Calculated 10th Percentile Threshold for Slice {slice_nr}: {percentile_10th:.2f}")
    else:
        # Use the previously calculated threshold for all other slices
        percentile_10th = globals().get("percentile_10th", 0)
    
    # Apply the 10th percentile threshold to the blurred image
    percentile_mask = (roi_image_blurred > percentile_10th) & roi_mask
    
    # Visualize the histogram of the intensity values of the blurred image
    plt.figure(figsize=(8, 6))
    plt.hist(roi_pixels_blurred, bins=50, color='gray', alpha=0.7)
    plt.title("Histogram of ROI Intensity Values (Blurred)")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    
    # Mark the 10th percentile threshold on the histogram
    plt.axvline(percentile_10th, color='g', linestyle='--', label=f'10th Percentile Threshold: {percentile_10th:.2f}')
    plt.legend()
    plt.show()
    
    # ---------------------------------------- SKELETONIZATION -----------------------------------
    
    # Apply the mask based on the 10th percentile threshold
    inverted_mask = ~percentile_mask
    closed_inverted = dilation(inverted_mask, disk(3))
    eroded_foreground = closed_inverted * roi_mask
    
    # Perform skeletonization and thickening
    skeleton = skeletonize(eroded_foreground)
    thicker_skeleton = dilation(skeleton, disk(3))
    
    # Final visualization of the skeletonized result
    plt.figure(figsize=(6, 6))
    plt.imshow(blurred_slice, cmap='gray')
    plt.imshow(thicker_skeleton, cmap='Reds', alpha=0.5)
    plt.title(f"Skeletonization with Percentile Threshold — Slice {slice_nr}")
    plt.axis("off")
    plt.show()
    
    # Optionally, store the threshold for future slices
    if slice_nr == z_min:
        globals()["percentile_10th"] = percentile_10th  # Store the threshold for future slices

    # Visualization (optional crop)
    if crop_bool:
        points = np.array([lcc_com, rcc_com, ncc_com]) 
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
        "snake": snake_current.copy(),
        "slice": dicom_slice.copy()
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
        reoriented_dicom.shape[0],                     # Z
        reoriented_dicom.shape[1],   # Y
        reoriented_dicom.shape[2]    # X
    ),
    dtype=reoriented_dicom.dtype
)

num_slices = reoriented_dicom.shape[0]

# Initialize calcification volume
calc_volume = np.zeros_like(masked_valve, dtype=bool)
calc_volume_downsampled = np.zeros_like(reoriented_dicom, dtype=bool)

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
        
        # Original ROI Mask (Before erosion)
        plt.subplot(1, 2, 1)
        plt.imshow(roi_mask, cmap='gray')
        plt.title("ROI Mask Before Erosion")
        plt.axis('off')
        
       
        plt.tight_layout()
        plt.show()
                
        # Extract intensities inside ROI
        roi_values = slice_img[roi_mask > 0]
        
        if roi_values.size == 0:
            median_intensity = np.nan
        else:
            median_intensity = np.median(roi_values)
        
        print(f"Slice {slice_nr}: median intensity inside ROI = {median_intensity:.2f}")
        
        # Compute lower and upper percentiles
        lower = np.percentile(roi_values, 25)   # remove bottom 5%
        upper = np.percentile(roi_values, 95)  # remove top 5%
        
        # Keep only values within 5th–95th percentile
        soft_tissue_values = roi_values[(roi_values >= lower) & (roi_values <= upper)]
        
        
        if slice_nr==z_min:
            # Compute mean and std of soft tissue - this was done previously this way, now percentile is used
            # mean_val = soft_tissue_values.mean()
            # std_val = soft_tissue_values.std()
        
            # Threshold for calcification: e.g., mean + 3*std
            calc_threshold =  np.percentile(roi_values, 99)  # upper bound used as threshold   
            print(f"Patient {patient_nr}: Threshold for calcification  = {calc_threshold:.2f}")
        
        # Print a histogram - necessary for seperating the calcification from the aortic valve
        print_histo = True
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
        # calc_volume_downsampled[slice_nr] = rescale(calc_mask, 1/scale_factor, order=0, preserve_range=True, anti_aliasing=False).astype(dicom_slice.dtype)

    # Otherwise leave slice empty
    else:
        masked_valve[slice_nr] = 0

# # (optional) Visualization fo the ROI

for z in range(z_min, z_max + 1):
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


#%% --------------------------------  PLOTTING ENTIRE FIGURE  --------------------------------------

# Loop over slice indices
for slice_idx in range(z_min,z_max):  # or any range you want
    # Extract the transversal slice
    transversal_slice = calc_volume[slice_idx, :, :]
    
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
    plt.pause(0.1)  # pause 2 seconds
    
plt.close()



# %%-------------------------------  EXTRACTION OF THE LEAFET BOUNDARIES ---------------------------

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

commissure_points = [rcc_lcc_com, lcc_ncc_com, ncc_rcc_com]

mercedes_star = False

# Loop over slices
for slice_nr, slice_info in slice_data.items():
    print(f"Processing slice {slice_nr}...")

    # --- Upscale slice for visualization / mask ---
    reoriented_slice = reoriented_dicom[slice_nr, :, :]
    upscaled = reoriented_slice

    # --- Active contour refinement ---
    temp_skeleton = slice_info["skeleton"]
    downsampled_snake = slice_info["snake"].copy()
    snake_current = functions.resample_closed_contour(downsampled_snake)

    # Preparing image for active contours    
    blurred_skeleton_norm = gaussian_filter(temp_skeleton.astype(float), sigma=7)
    blurred_skeleton_norm /= blurred_skeleton_norm.max() + 1e-8

    # The snake is rerun, to close off the entire aortic wall. Previous snake could have had holes in it
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
        
    # --- Visualization of the intermediate active contour --- optional, if the contour is failigng
    # plt.figure(figsize=(6, 6))
    # plt.imshow(blurred_skeleton_norm, cmap="gray")  # Show the blurred skeleton (background)
    # # plt.plot(snake_current[:, 1], snake_current[:, 0], '-r', lw=2, alpha=0.8)  # Show the evolving snake
    # plt.title(f"Active contour iteration {i+1} — Slice {slice_nr}")
    # plt.axis("off")
    # plt.show()

    # --- Isolate inner skeleton ---
    skeleton_mask = polygon2mask(temp_skeleton.shape, snake_current)
    roi_mask_eroded = binary_erosion(skeleton_mask)
    inner_skeleton = temp_skeleton * roi_mask_eroded.astype(temp_skeleton.dtype)

    # Upsample contour for accurate indices
    aortic_wall_contour = functions.resample_closed_contour(snake_current)

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
        center, aortic_wall_contour, i_lcc, i_ncc, seg_lcc_ncc, upscaled, inner_skeleton)
    rcc_lcc_mask, rcc_lcc_boundary = functions.create_boundary_mask(
        center, aortic_wall_contour, i_rcc, i_lcc, seg_rcc_lcc, upscaled, inner_skeleton)
    ncc_rcc_mask, ncc_rcc_boundary = functions.create_boundary_mask(
        center, aortic_wall_contour, i_ncc, i_rcc, seg_ncc_rcc, upscaled, inner_skeleton)

    # Check if commissure points from the previous slice are available
    commissure_points = {
        'lcc_ncc': prev_intersections['lcc_ncc'] if prev_intersections['lcc_ncc'] is not None else base_commissure['lcc_ncc'],
        'rcc_lcc': prev_intersections['rcc_lcc'] if prev_intersections['rcc_lcc'] is not None else base_commissure['rcc_lcc'],
        'ncc_rcc': prev_intersections['ncc_rcc'] if prev_intersections['ncc_rcc'] is not None else base_commissure['ncc_rcc']
    }

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
            upscaled,
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
        print(f"Slice {slice_nr}: All 3 intersections found. Using them and updating previous intersections.")
        for name in commissure_names:
            print(f"  Using intersection for {name} from this slice.")
            prev_intersections[name] = intersections[name]
            commissure_indices[name] = functions.closest_contour_point(
                intersections[name],
                aortic_wall_contour
            )
    
    # Case B — Some intersections missing, use available ones and previous for missing ones
    elif have_full_prev:
        print(f"Slice {slice_nr}: Missing intersections, but previous intersections are available. Using previous intersections.")
        for name in commissure_names:
            if intersections.get(name) is not None:  # If intersection is found, use it
                print(f"  Using intersection for {name} from this slice.")
                commissure_indices[name] = functions.closest_contour_point(
                    intersections[name],
                    aortic_wall_contour
                )
            else:  # Otherwise, use previous intersection
                print(f"  Using previous intersection for {name}.")
                commissure_indices[name] = functions.closest_contour_point(
                    prev_intersections[name],
                    aortic_wall_contour
                )
    
    # Case C — Early slices where we do NOT have all 3 prev intersections yet
    # → fall back per-intersection to base commissures
    else:
        print(f"Slice {slice_nr}: Not all intersections found and no full previous set. Falling back to base commissures for missing intersections.")
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

    print(f"{slice_nr} intersections: ", intersections)

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
        "aortic_wall_contour": aortic_wall_contour.copy(),
        "height": slice_nr
    }
    RCC_data[slice_nr] = {
        "mask": rcc_lcc_mask.copy(),
        "rcc_lcc_boundary": cleaned_rcc_lcc_boundary.copy(),
        "ncc_rcc_boundary": cleaned_ncc_rcc_boundary.copy(),
        "com_to_com": seg_c2c["RCC"].copy(),
        "height": slice_nr
        
    }
    NCC_data[slice_nr] = {
        "mask": ncc_rcc_mask.copy(),
        "ncc_rcc_boundary": cleaned_ncc_rcc_boundary.copy(),
        "lcc_ncc_boundary": cleaned_lcc_ncc_boundary.copy(),
        "com_to_com": seg_c2c["NCC"].copy(),
        "height": slice_nr
    }
    
    # Optional visualization (COM-to-COM segments + boundaries)
    plt.figure(figsize=(6, 6))
    plt.imshow(upscaled, cmap='gray', origin='upper')
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
    # Print the size of the boundaries
    print(f"Size of LCC-NCC Boundary (Slice {slice_nr}):", cleaned_lcc_ncc_boundary.shape)
    print(f"Size of RCC-LCC Boundary (Slice {slice_nr}):", cleaned_rcc_lcc_boundary.shape)
    print(f"Size of NCC-RCC Boundary (Slice {slice_nr}):", cleaned_ncc_rcc_boundary.shape)
    
    
# %% --- Save each contour segment as VTK ---
import os
import numpy as np
import functions  # Import functions from your functions.py file

# Collect all boundaries (list of NumPy arrays) and corresponding heights (z-values)
lcc_boundaries = []
rcc_boundaries = []
ncc_boundaries = []
heights = []  # Corresponding heights for each boundary

for slice_nr in slice_data:
    print(f"Processing slice {slice_nr}...")

    # Get the boundary data from your existing dictionary (RCC_data, LCC_data, NCC_data)
    lcc_boundary = LCC_data[slice_nr]["lcc_ncc_boundary"]
    rcc_boundary = RCC_data[slice_nr]["rcc_lcc_boundary"]
    ncc_boundary = NCC_data[slice_nr]["ncc_rcc_boundary"]
    
    # Get the height (z-coordinate) for the current slice, which might be a value in your dataset
    slice_height = LCC_data[slice_nr]["height"]  # Ensure this field exists in your data

    # Store the boundaries and corresponding height
    lcc_boundaries.append(np.array(lcc_boundary))
    rcc_boundaries.append(np.array(rcc_boundary))
    ncc_boundaries.append(np.array(ncc_boundary))
    heights.append(slice_height)  # Store the z-coordinate (height)

# Combine boundaries for LCC, RCC, NCC into vtkPolyData objects with z-coordinate consideration
lcc_combined_polydata = functions.combine_boundaries_to_polydata(lcc_boundaries, heights)
rcc_combined_polydata = functions.combine_boundaries_to_polydata(rcc_boundaries, heights)
ncc_combined_polydata = functions.combine_boundaries_to_polydata(ncc_boundaries, heights)

# Save each combined polydata to a VTK file
output_path = "H:/DATA/Afstuderen/3.Data/output_valve_segmentation/savi01"
os.makedirs(output_path, exist_ok=True)

# Save the combined boundaries
functions.save_vtk_polydata(lcc_combined_polydata, os.path.join(output_path, "LCC_combined_boundaries.vtk"))
functions.save_vtk_polydata(rcc_combined_polydata, os.path.join(output_path, "RCC_combined_boundaries.vtk"))
functions.save_vtk_polydata(ncc_combined_polydata, os.path.join(output_path, "NCC_combined_boundaries.vtk"))

# %% ------------------------ COM-TO-COM SEGMENTS ----------------------------

import os
import numpy as np
import functions  # Import functions from your functions.py file

# Collect all com_to_com segments (list of NumPy arrays) and corresponding heights (z-values)
lcc_com_to_com = []
rcc_com_to_com = []
ncc_com_to_com = []
heights_com_to_com = []  # Corresponding heights for each com_to_com segment

for slice_nr in slice_data:
    print(f"Processing slice {slice_nr} for com_to_com segments...")

    # Get the com_to_com data from your existing dictionary (RCC_data, LCC_data, NCC_data)
    lcc_com_to_com_segment = LCC_data[slice_nr]["com_to_com"]
    rcc_com_to_com_segment = RCC_data[slice_nr]["com_to_com"]
    ncc_com_to_com_segment = NCC_data[slice_nr]["com_to_com"]
    
    # Get the height (z-coordinate) for the current slice, which might be a value in your dataset
    slice_height = LCC_data[slice_nr]["height"]  # Ensure this field exists in your data

    # Store the com_to_com segments and corresponding height
    lcc_com_to_com.append(np.array(lcc_com_to_com_segment))
    rcc_com_to_com.append(np.array(rcc_com_to_com_segment))
    ncc_com_to_com.append(np.array(ncc_com_to_com_segment))
    
    heights_com_to_com.append(slice_height)  # Store the z-coordinate (height)

# Combine com_to_com segments for LCC, RCC, NCC into vtkPolyData objects with z-coordinate consideration
lcc_com_to_com_polydata = functions.combine_boundaries_to_polydata(lcc_com_to_com, heights_com_to_com)
rcc_com_to_com_polydata = functions.combine_boundaries_to_polydata(rcc_com_to_com, heights_com_to_com)
ncc_com_to_com_polydata = functions.combine_boundaries_to_polydata(ncc_com_to_com, heights_com_to_com)

# Save each combined polydata to a VTK file for com_to_com segments
output_path = "H:/DATA/Afstuderen/3.Data/output_valve_segmentation/savi01"
os.makedirs(output_path, exist_ok=True)

# Save the combined com_to_com segments
functions.save_vtk_polydata(lcc_com_to_com_polydata, os.path.join(output_path, "LCC_combined_com_to_com.vtk"))
functions.save_vtk_polydata(rcc_com_to_com_polydata, os.path.join(output_path, "RCC_combined_com_to_com.vtk"))
functions.save_vtk_polydata(ncc_com_to_com_polydata, os.path.join(output_path, "NCC_combined_com_to_com.vtk"))

