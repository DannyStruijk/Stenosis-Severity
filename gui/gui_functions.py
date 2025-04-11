# gui_functions.py
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.ndimage import affine_transform
import os

def load_dicom(dicom_file_path):
    """Loads a DICOM file and returns the pixel data."""
    dicom = pydicom.dcmread(dicom_file_path)
    return dicom.pixel_array

def get_transversal_slice(image_data, slice_index):
    """Returns the slice image data for a specific index."""
    return image_data[:, :, slice_index]

def get_coronal_slice(image_data, slice_index):
    """Returns the slice image data for a specific index."""
    return image_data[:, slice_index, :]

def enhance_contrast(slice_data):
    """Enhances the contrast of the slice using linear contrast stretching."""
    min_val = slice_data.min()
    max_val = slice_data.max()
    enhanced_image = 255 * (slice_data - min_val) / (max_val - min_val)
    return enhanced_image

def update_transversal(slice_index_transversal, image_data, canvas, landmarks, vtk_surface_points = None):
    """Updates the image displayed in the GUI based on the current slice index, with landmarks and overlayed surface."""
    
    # Extract slice data
    slice_data = get_transversal_slice(image_data, slice_index_transversal)
    
    # Create a new figure for displaying the image
    fig, ax = plt.subplots()
    
    # Show the image slice in grayscale
    ax.imshow(np.rot90(slice_data, k=1), cmap='gray', origin='upper')
    #ax.imshow(slice_data, cmap='gray', origin='upper')
    ax.axis('off')  # Hide the axis for a clean image

    # Plot landmarks on the image (points or circles)
    for (x, y, z) in landmarks:
        if z == slice_index_transversal:  # Plot landmarks only on the current slice
            ax.plot(x, y, 'ro', markersize=5)  # 'ro' means red dots, markersize controls the size

    # Update the canvas with the new figure
    canvas.figure = fig
    canvas.draw()

    # Close the figure to prevent memory leaks
    plt.close(fig)  # Close the figure
    

def update_coronal(slice_index_coronal, image_data, canvas, angle=None):
    """Updates the image displayed in the GUI based on the current slice index, with landmarks and overlayed surface and optional angle line clipped to image edges."""
    
    # Extract slice data
    slice_data = get_coronal_slice(image_data, slice_index_coronal)

    # Create a new figure for displaying the image
    fig, ax = plt.subplots()

    # Rotate and flip for correct orientation
    rotated_slice = np.fliplr(np.rot90(slice_data, k=-1))
    ax.imshow(rotated_slice, cmap="gray")
    ax.axis('off')  # Hide axes

    if angle is not None:
        height, width = rotated_slice.shape
        center_x = width / 2
        center_y = height / 2
        angle_rad = np.radians(-angle)

        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        # Avoid division by zero
        if dx == 0:
            x_vals = [center_x, center_x]
            y_vals = [0, height]
        else:
            # Calculate intersections with image bounds
            x0, x1 = 0, width
            y_at_x0 = center_y + (x0 - center_x) * dy / dx
            y_at_x1 = center_y + (x1 - center_x) * dy / dx

            y0, y1 = 0, height
            x_at_y0 = center_x + (y0 - center_y) * dx / dy if dy != 0 else center_x
            x_at_y1 = center_x + (y1 - center_y) * dx / dy if dy != 0 else center_x

            points = []

            if 0 <= y_at_x0 <= height:
                points.append((x0, y_at_x0))
            if 0 <= y_at_x1 <= height:
                points.append((x1, y_at_x1))
            if 0 <= x_at_y0 <= width:
                points.append((x_at_y0, y0))
            if 0 <= x_at_y1 <= width:
                points.append((x_at_y1, y1))

            # If we got 2 valid points, draw the line
            if len(points) >= 2:
                (x1, y1), (x2, y2) = points[:2]
                ax.plot([x1, x2], [y1, y2], 'r--', linewidth=1)

    canvas.figure = fig
    canvas.draw()
    plt.close(fig)





def next_slice(slice_index, image_data, canvas, landmarks):
    """Displays the next slice."""
    if slice_index < image_data.shape[0] - 1:
        slice_index += 1
        update_transversal(slice_index, image_data, canvas, landmarks)  # Pass landmarks here
    return slice_index

def prev_slice(slice_index, image_data, canvas, landmarks):
    """Displays the previous slice."""
    if slice_index > 0:
        slice_index -= 1
        update_transversal(slice_index, image_data, canvas, landmarks)  # Pass landmarks here
    return slice_index


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
    Create a rotation matrix for rotating around an arbitrary axis.
    
    Input: 
        - axis: this determines around which axis the rotation matrix is. Format (X Y Z) is expected. [0 1 0] indicates Y-axis rotation
        - angle: angle in radians 
    
    Output:
        - R: the rotation matrix (Rodriguez Method) which is to be applied for affine transformation.
    
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
    
    Input:
        - dicom: one dicom file, in order to extract properties of the dicom for the pixel spacing
        - volume: volume, in (X Y Z) format, which is to be rescaled.
        
    Output:
        rescaled_volume: the rescaled format according to the pixel spacing of the DICOM. Pay attention whether
        the physical distance between slices and the slice thickness is similar
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

def rotated_volume(volume, rotation_matrix):
    """ 
    Rotate the matrix depending on the given rotation matrix
    
    Input:
        - volume: this is the (rescaled!) volume in the format (X Y Z) which is to be rotated.
        - rotation_matrix: this is the rodriguez matrix as calculated by rotation_matrix()
        
    Output:
        rotated_volume: rotated volume
    
    The offset is needed to determine around which center the transformation takes place
    
    """
    
    # Determine the center before rotating the structure
    center = 0.5 * np.array(volume.shape)
    offset = center - rotation_matrix @ center
    
    # Rotate the volume around the specified axis
    rotated_volume = affine_transform(volume, rotation_matrix, offset=offset, order=1)
    
    return rotated_volume
    