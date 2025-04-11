# gui_functions.py
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.transform import Rotation as R
import scipy.ndimage 
from scipy.ndimage import zoom

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

   
from scipy.ndimage import affine_transform


def rotation_matrix(axis, angle):
    """
    Create a rotation matrix for rotating around an arbitrary axis.
    """
    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    ux, uy, uz = axis

    # Rodrigues' rotation formula
    R = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])
    
    return R

def rescale_volume(dicom, volume):
    """ 
    Rescale the volume so that it matches the real spacing of the DICOM, instead of assuming the voxels to be isotropic
    """
    # Get pixel spacing correctly as a tuple of floats (Y, X)
    pixel_spacing = tuple(map(float, dicom.PixelSpacing))  # (Y, X)

    # Get slice thickness
    slice_thickness = float(dicom.SliceThickness)

    # Combine into new_spacing (Z, Y, X)
    original_spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])
    desired_spacing = (pixel_spacing[0], pixel_spacing[1], pixel_spacing[1])
    
    rescale_factors = np.array(original_spacing) / np.array(desired_spacing)
    rescaled_volume = zoom(volume, rescale_factors, order=1)
    return rescaled_volume
    
