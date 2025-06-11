import open3d as o3d
import numpy as np
import pydicom
import gui_functions as gf
import nrrd
from scipy.ndimage import binary_closing

# =================== SETTINGS ===================

pointcloud_path = r"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data\aos14\mean_shape_reconstruction_14.ply"
output_nrrd_path = r"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data\aos14\mean_shape_voxelized_14.nrrd"
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"

# =================== CORE FUNCTIONS ===================

def load_point_cloud_and_dicom(pointcloud_path, dicom_dir):
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    print(f"Loaded point cloud with {len(pcd.points)} points.")

    sorted_dicom_files = gf.get_sorted_dicom_files(dicom_dir)[::-1]  # Flip so Z increases upward

    dicom = pydicom.dcmread(sorted_dicom_files[0][0])
    spacing_x, spacing_y = map(float, dicom.PixelSpacing)
    slice_thickness = float(dicom.SliceThickness)
    spacing = np.array([spacing_x, spacing_y, slice_thickness], dtype=np.float32)

    rows = dicom.Rows
    columns = dicom.Columns
    num_slices = len(sorted_dicom_files)
    origin = np.array(dicom.ImagePositionPatient, dtype=np.float32)

    return pcd, spacing, origin, rows, columns, num_slices, sorted_dicom_files

def voxelize_point_cloud(pcd, origin, spacing, rows, columns, num_slices):
    points = np.asarray(pcd.points, dtype=np.float32)
    relative_coords = (points - origin) / spacing
    voxel_coords = np.round(relative_coords).astype(int)

    print("Voxel coordinates range:")
    print("X:", voxel_coords[:, 0].min(), voxel_coords[:, 0].max())
    print("Y:", voxel_coords[:, 1].min(), voxel_coords[:, 1].max())
    print("Z:", voxel_coords[:, 2].min(), voxel_coords[:, 2].max())

    labelmap = np.zeros((num_slices, rows, columns), dtype=np.uint8)

    for x, y, z in voxel_coords:
        if 0 <= x < columns and 0 <= y < rows and 0 <= z < num_slices:
            labelmap[z, y, x] = 1

    print(f"Number of voxels marked as 1 in the labelmap: {np.count_nonzero(labelmap)}")
    return labelmap

def apply_morphological_closing(labelmap, structure_size=3):
    structure = np.ones((structure_size, structure_size, structure_size), dtype=bool)
    closed_labelmap = binary_closing(labelmap, structure=structure).astype(np.uint8)

    print(f"Number of voxels after closing: {np.count_nonzero(closed_labelmap)}")
    return closed_labelmap

def save_labelmap_as_nrrd(labelmap, spacing, origin, save_path):
    header = {
        'type': 'short',
        'dimension': 3,
        'space': 'left-posterior-superior',
        'sizes': list(labelmap.shape),
        'space directions': [
            [0.0, 0.0, spacing[2]],
            [0.0, spacing[1], 0.0],
            [spacing[0], 0.0, 0.0]
        ],
        'kinds': ['domain', 'domain', 'domain'],
        'encoding': 'raw',
        'endian': 'little',
        'space origin': origin.tolist()
    }

    nrrd.write(save_path, labelmap, header)
    print(f"Saved NRRD file to: {save_path}")

# =================== WRAPPER FUNCTION ===================

def convert_average_shape_to_nrrd(pcd_path, dicom_dir, output_nrrd_path, structure_size=3):
    """
    Full wrapper function to convert average shape point cloud into an NRRD labelmap aligned with a DICOM volume.

    Args:
        pcd_path (str): Path to point cloud (.ply).
        dicom_dir (str): Folder containing DICOM slices.
        output_nrrd_path (str): Output path for the NRRD file.
        structure_size (int, optional): Size of morphological closing kernel. Defaults to 3.
    """
    pcd, spacing, origin, rows, columns, num_slices, _ = load_point_cloud_and_dicom(pcd_path, dicom_dir)
    labelmap = voxelize_point_cloud(pcd, origin, spacing, rows, columns, num_slices)
    closed_labelmap = apply_morphological_closing(labelmap, structure_size=structure_size)
    save_labelmap_as_nrrd(closed_labelmap, spacing, origin, output_nrrd_path)

# =================== MAIN ===================

# if __name__ == "__main__":
#     convert_average_shape_to_nrrd(pointcloud_path, dicom_dir, output_nrrd_path)
