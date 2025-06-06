import open3d as o3d
import numpy as np
import pydicom
import gui_functions as gf
import nrrd
from scipy.ndimage import binary_closing

#=================== SETTINGS

pointcloud_path = r"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data\aos14\mean_shape_reconstruction_14.ply"
output_nrrd_path = r"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data\aos14\mean_shape_voxelized_14.nrrd"
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"

#=================== LOAD POINT CLOUD & ACCOMPANYING DICOM

pcd = o3d.io.read_point_cloud(pointcloud_path)
print(f"Loaded point cloud with {len(pcd.points)} points.")

# Load and sort DICOM slices in ascending order (inferior to superior)
sorted_dicom_files = gf.get_sorted_dicom_files(dicom_dir)
sorted_dicom_files = sorted_dicom_files[::-1]  # Make sure Z increases upward
dicom = pydicom.dcmread(sorted_dicom_files[0][0])

# Extract spacing and origin
spacing_x, spacing_y = map(float, dicom.PixelSpacing)  # [mm]
slice_thickness = float(dicom.SliceThickness)          # [mm]
spacing = np.array([spacing_x, spacing_y, slice_thickness], dtype=np.float32)

# Dimensions
rows = dicom.Rows
columns = dicom.Columns
num_slices = len(sorted_dicom_files)

# Origin of the first slice
origin = np.array(dicom.ImagePositionPatient, dtype=np.float32)

# Convert point cloud to voxel indices
points = np.asarray(pcd.points, dtype=np.float32)
relative_coords = (points - origin) / spacing
voxel_coords = np.round(relative_coords).astype(int)

print("Voxel coordinates range:")
print("X:", voxel_coords[:, 0].min(), voxel_coords[:, 0].max())
print("Y:", voxel_coords[:, 1].min(), voxel_coords[:, 1].max())
print("Z:", voxel_coords[:, 2].min(), voxel_coords[:, 2].max())

# Create empty labelmap in (Z, Y, X) format
labelmap = np.zeros((num_slices, rows, columns), dtype=np.uint8)

# Fill in the voxelized points
for x, y, z in voxel_coords:
    if 0 <= x < columns and 0 <= y < rows and 0 <= z < num_slices:
        labelmap[z, y, x] = 1

num_voxels_filled = np.count_nonzero(labelmap)
print(f"Number of voxels marked as 1 in the labelmap: {num_voxels_filled}")

# ----------------- Apply morphological closing -----------------

structure = np.ones((3, 3, 3), dtype=bool)  # 3x3x3 cube structuring element
closed_labelmap = binary_closing(labelmap, structure=structure).astype(np.uint8)

num_voxels_after = np.count_nonzero(closed_labelmap)
print(f"Number of voxels after closing: {num_voxels_after}")

#=================== SAVE AS NRRD FILE

# NRRD header (matches orientation and spacing)
header = {
    'type': 'short',
    'dimension': 3,
    'space': 'left-posterior-superior',
    'sizes': list(closed_labelmap.shape),  # [Z, Y, X]
    'space directions': [
        [0.0, 0.0, spacing[2]],  # Z
        [0.0, spacing[1], 0.0],  # Y
        [spacing[0], 0.0, 0.0]   # X
    ],
    'kinds': ['domain', 'domain', 'domain'],
    'encoding': 'raw',
    'endian': 'little',
    'space origin': origin.tolist()
}

nrrd.write(output_nrrd_path, closed_labelmap, header)
print(f"Saved NRRD file to: {output_nrrd_path}")
