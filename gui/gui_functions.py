# gui_functions.py
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.transform import Rotation as R
import scipy.ndimage 

def load_dicom(dicom_file_path):
    """Loads a DICOM file and returns the pixel data."""
    dicom = pydicom.dcmread(dicom_file_path)
    return dicom.pixel_array

def get_slice(image_data, slice_index):
    """Returns the slice image data for a specific index."""
    return image_data[slice_index, :, :]

def enhance_contrast(slice_data):
    """Enhances the contrast of the slice using linear contrast stretching."""
    min_val = slice_data.min()
    max_val = slice_data.max()
    enhanced_image = 255 * (slice_data - min_val) / (max_val - min_val)
    return enhanced_image

def initialize_slice_index():
    """Initializes and returns the starting slice index."""
    return 90  # Start from a specific slice (e.g., 90th slice)

def update_image(slice_index, image_data, canvas, landmarks, vtk_surface_points = None):
    """Updates the image displayed in the GUI based on the current slice index, with landmarks and overlayed surface."""
    
    # Extract slice data
    slice_data = get_slice(image_data, slice_index)
    
    # Create a new figure for displaying the image
    fig, ax = plt.subplots()
    
    # Show the image slice in grayscale
    ax.imshow(slice_data, cmap='gray')
    ax.axis('off')  # Hide the axis for a clean image

    # Plot landmarks on the image (points or circles)
    for (x, y, z) in landmarks:
        if z == slice_index:  # Plot landmarks only on the current slice
            ax.plot(x, y, 'ro', markersize=10)  # 'ro' means red dots, markersize controls the size

    # Overlay the surface points on the image
    # vtk_surface_points should be a NumPy array with x, y, z coordinates (integer values)
    # surface_points_on_slice = vtk_surface_points[np.abs(vtk_surface_points[:, 2] - slice_index) < 2]  # Allow a tolerance for z-axis
    
    # if len(surface_points_on_slice) > 0:
    #     ax.scatter(surface_points_on_slice[:, 0], surface_points_on_slice[:, 1], c='b', s=10, label="Leaflet Surface", alpha=0.6)

    # Update the canvas with the new figure
    canvas.figure = fig
    canvas.draw()

    # Close the figure to prevent memory leaks
    plt.close(fig)  # Close the figure

def next_slice(slice_index, image_data, canvas, landmarks):
    """Displays the next slice."""
    if slice_index < image_data.shape[0] - 1:
        slice_index += 1
        update_image(slice_index, image_data, canvas, landmarks)  # Pass landmarks here
    return slice_index

def prev_slice(slice_index, image_data, canvas, landmarks):
    """Displays the previous slice."""
    if slice_index > 0:
        slice_index -= 1
        update_image(slice_index, image_data, canvas, landmarks)  # Pass landmarks here
    return slice_index

import os

def get_sorted_dicom_files(dicom_dir):
    """
    This function takes the directory containing DICOM files, reads each DICOM file,
    sorts them based on the Z position (SliceLocation or ImagePositionPatient), 
    and returns a sorted list of file paths with their corresponding Z positions.
    
    Parameters:
    dicom_dir (str): The directory containing the DICOM files.
    
    Returns:
    list of tuples: A sorted list of tuples where each tuple contains the DICOM file path
                    and the corresponding Z position.
    """
    # Get a list of all DICOM files in the directory
    dicom_files_test = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)]

    dicom_files_sorted = []

    for dicom_file in dicom_files_test:
        dicom_data = pydicom.dcmread(dicom_file)
        
        # Try to get the Z position (SliceLocation or ImagePositionPatient)
        if 'SliceLocation' in dicom_data:
            z_position = dicom_data.SliceLocation
        elif 'ImagePositionPatient' in dicom_data:
            # If ImagePositionPatient exists, take the third value as the Z position
            z_position = dicom_data.ImagePositionPatient[2]
        else:
            z_position = None

        # Append the file and its Z position to a list
        dicom_files_sorted.append((dicom_file, z_position))

    # Sort the list by Z position (ascending)
    dicom_files_sorted.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))

    return dicom_files_sorted


def get_sorted_image_data(sorted_dicom_files):
    """
    Given a sorted list of DICOM files, reads each file and returns a 3D NumPy array of image data.
    
    Parameters:
    sorted_dicom_files (list of tuples): List of sorted DICOM file paths and their Z positions.
    
    Returns:
    numpy.ndarray: 3D array containing all the DICOM slices.
    """
    image_data_list = []  # List to store individual 2D slices
    
    for dicom_file, z_position in sorted_dicom_files:
        dicom_data = pydicom.dcmread(dicom_file)  # Read the DICOM file
        image_data_list.append(dicom_data.pixel_array)  # Extract the pixel array (2D slice)
    
    # Stack the 2D slices into a 3D NumPy array (axis 0 is the slice dimension)
    image_data = np.stack(image_data_list, axis=0)
    #image_data = np.clip(image_data, 0, 65535)  # Clamp values to the max uint16 range
    #image_data = (image_data / 256).astype(np.uint8)  # Scale to uint8 range (0-255
    
    return image_data

from scipy.ndimage import map_coordinates

def dicom_to_matrix(image_data):
    """
    Converts a series of DICOM images to a 3D matrix
    
    Parameters: 
        - image_data: The SORTED DICOM series
        
    Output:
        - The 3D matrix containing all of the infromation
    
    """
    # Convert the DICOM slices to a 3D NumPy array
    volume = np.stack([load_dicom(file[0]) for file in image_data], axis=0)
    
    return volume

def rotate_3d_matrix(image_data, normal_vector):
    """
    Rotates the 3D DICOM volume so that the annular plane normal aligns with the Z-axis.

    Parameters:
        image_data (numpy.ndarray): 3D array of shape (Z, Y, X).
        normal_vector (numpy.ndarray): Normal vector of the annular plane.

    Returns:
        numpy.ndarray: Rotated 3D image volume.
    """
    # Normalize the annular plane normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Define the target viewing direction (Z-axis)
    z_axis = np.array([0, 0, 1])

    # Compute rotation axis and angle
    rotation_axis = np.cross(normal_vector, z_axis)
    rotation_norm = np.linalg.norm(rotation_axis)

    if rotation_norm < 1e-6:  # If already aligned, return original data
        return image_data

    rotation_axis /= rotation_norm  # Normalize axis
    angle = np.arccos(np.clip(np.dot(normal_vector, z_axis), -1.0, 1.0))  # Ensure numerical stability

    # Compute rotation matrix
    rotation_matrix = R.from_rotvec(angle * rotation_axis).as_matrix()

    # Get volume dimensions
    z_dim, y_dim, x_dim = image_data.shape
    center = np.array([z_dim / 2, y_dim / 2, x_dim / 2])  # Rotation center

    # Generate coordinate grid
    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij'
    )

    # Flatten and shift coordinates to center
    coords = np.vstack([z_coords.ravel(), y_coords.ravel(), x_coords.ravel()])
    coords_centered = coords - center[:, np.newaxis]  # Move to origin

    # Apply rotation
    rotated_coords = rotation_matrix @ coords_centered
    rotated_coords += center[:, np.newaxis]  # Move back

    # Reshape rotated coordinates
    z_rotated, y_rotated, x_rotated = rotated_coords.reshape(3, z_dim, y_dim, x_dim)

    # Interpolate the rotated image
    rotated_image_data = map_coordinates(image_data, [z_rotated, y_rotated, x_rotated], order=1, mode='nearest')

    return rotated_image_data


def rotate_image_fixed(image_data, angle_x=0, angle_y=0, angle_z=0):
    """
    Rotates a 3D image along the specified axes.

    Parameters:
        image_data (numpy.ndarray): The 3D volume (Z, Y, X).
        angle_x (float): Rotation around the X-axis (in degrees).
        angle_y (float): Rotation around the Y-axis (in degrees).
        angle_z (float): Rotation around the Z-axis (in degrees).

    Returns:
        numpy.ndarray: The rotated 3D volume.
    """
    if not isinstance(image_data, np.ndarray) or image_data.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array.")

    # Apply rotations sequentially
    rotated_data = image_data.copy()
    
    # KEEP IN MIND: The coordinate system is (z, y, x)
    if angle_x != 0:
        rotated_data = rotate(rotated_data, angle_x, axes=(0, 1), reshape=True, order=1, mode='nearest')  # YZ plane
    
    if angle_y != 0:
        rotated_data = rotate(rotated_data, angle_y, axes=(0, 2), reshape=True, order=1, mode='nearest')  # XZ plane
    
    if angle_z != 0:
        rotated_data = rotate(rotated_data, angle_z, axes=(1, 2), reshape=True, order=1, mode='nearest')  # XY plane

    return rotated_data

def calculate_rotation(annular_normal):
    """
    Calculate the rotation axis and angle required to align the Z-axis with the given normal vector.
    
    Parameters:
        annular_normal (numpy array): The normal vector of the annular plane.
        
    Returns:
        rotation_axis (numpy array): The axis around which the rotation should be performed.
        rotation_angle (float): The angle (in radians) required to align the Z-axis with the normal vector.
    """
    # Default normal vector (Z-axis)
    z_axis = np.array([1, 0, 0])
    
    # Compute the rotation axis (cross product of Z-axis and annular normal)
    rotation_axis = np.cross(z_axis, annular_normal)
    
    # Compute the rotation angle (using dot product and arccos)
    rotation_angle = np.arccos(np.dot(z_axis, annular_normal) / (np.linalg.norm(z_axis) * np.linalg.norm(annular_normal)))
    
    # Normalize the rotation axis (to ensure it's a unit vector)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis) if np.linalg.norm(rotation_axis) != 0 else rotation_axis
    
    return rotation_axis, rotation_angle

def rotation_matrix(axis, angle):
    """
    Create a rotation matrix for a given axis and angle.
    
    Parameters:
        axis (numpy array): The rotation axis (unit vector).
        angle (float): The rotation angle in radians.
    
    Returns:
        rotation_matrix (numpy array): The corresponding rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    ux, uy, uz = axis
    
    # Rotation matrix using the Rodrigues' rotation formula
    rotation_matrix = np.array([
        [cos_angle + ux**2 * (1 - cos_angle), ux * uy * (1 - cos_angle) - uz * sin_angle, ux * uz * (1 - cos_angle) + uy * sin_angle],
        [uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy**2 * (1 - cos_angle), uy * uz * (1 - cos_angle) - ux * sin_angle],
        [uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle, cos_angle + uz**2 * (1 - cos_angle)]
    ])
    
    return rotation_matrix
    
from scipy.ndimage import affine_transform

def apply_rotation(volume, rotation_axis, rotation_angle):
    """
    Rotate the 3D volume (DICOM image data) based on the given axis and angle.
    
    Parameters:
        volume (numpy array): The 3D volume (shape: Z, Y, X).
        rotation_axis (numpy array): The axis around which to rotate.
        rotation_angle (float): The angle by which to rotate (in radians).
    
    Returns:
        rotated_volume (numpy array): The rotated 3D volume.
    """
    # Compute the rotation matrix
    matrix = rotation_matrix(rotation_axis, rotation_angle)

    # Get the shape of the volume
    z_dim, y_dim, x_dim = volume.shape
    center = np.array([z_dim / 2, y_dim / 2, x_dim / 2])  # Rotation center

    # Create the affine transformation matrix for scipy.ndimage
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = matrix  # Apply rotation to 3x3 part
    transform_matrix[:3, 3] = center - matrix @ center  # Adjust translation

    # Apply affine transform to the volume
    rotated_volume = affine_transform(volume, transform_matrix[:3, :3], offset=transform_matrix[:3, 3], order=1, mode='nearest')

    return rotated_volume

def reslice_numpy_volume(volume, normal_vector, slice_thickness=1.0, num_slices=50):
    """
    Reslices a 3D NumPy volume along a specified normal vector.

    Parameters:
    - volume (numpy array): 3D volume (shape: [Z, Y, X]).
    - normal_vector (array-like): Normal vector defining the slicing direction.
    - slice_thickness (float): Distance between adjacent resampled slices.
    - num_slices (int): Number of slices to extract.

    Returns:
    - resliced_volume (numpy array): 3D array of resampled slices.
    """

    # Normalize the normal vector
    normal_vector = np.array(normal_vector, dtype=np.float64)
    normal_vector /= np.linalg.norm(normal_vector)

    # Get volume dimensions
    depth, height, width = volume.shape

    # Define center of the volume
    center = np.array([depth // 2, height // 2, width // 2])

    # Generate slice positions along the normal
    slice_positions = [center + i * slice_thickness * normal_vector for i in range(-num_slices // 2, num_slices // 2)]

    # Define the grid in the slice plane
    grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    
    # Reslicing using interpolation
    resliced_slices = []
    for slice_position in slice_positions:
        slice_z = np.full_like(grid_y, slice_position[0])  # Same Z-coordinate for the entire plane
        slice_y = grid_y + slice_position[1] - center[1]
        slice_x = grid_x + slice_position[2] - center[2]

        # Interpolate using scipy.ndimage.map_coordinates
        interpolated_slice = scipy.ndimage.map_coordinates(volume, [slice_z, slice_y, slice_x], order=1, mode="nearest")
        resliced_slices.append(interpolated_slice)

    return np.array(resliced_slices)