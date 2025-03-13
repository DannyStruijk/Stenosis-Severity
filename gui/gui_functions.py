# gui_functions.py
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

def update_image(slice_index, image_data, canvas, landmarks):
    """Updates the image displayed in the GUI based on the current slice index, with landmarks."""
    slice_data = get_slice(image_data, slice_index)
    # Optional contrast enhancement
    # enhanced_image = enhance_contrast(slice_data)

    fig, ax = plt.subplots()
    ax.imshow(slice_data, cmap='gray')
    ax.axis('off')

    # Plot landmarks on the image (points or circles)
    for (x, y, z) in landmarks:
        if z == slice_index: 
            ax.plot(x, y, 'ro', markersize=10)  # 'ro' means red dots

    # Update the canvas with the new figure
    canvas.figure = fig
    canvas.draw()

    # Close the figure after it is drawn to prevent memory leak
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
