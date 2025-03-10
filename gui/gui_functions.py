# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:01:56 2025

@author: u840707
"""

# functions/dicom_functions.py
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
    return 90

def update_image(slice_index, image_data, canvas):
    """Updates the image displayed in the GUI based on the current slice index."""
    slice_data = get_slice(image_data, slice_index)
    #enhanced_image = enhance_contrast(slice_data)
    
    fig, ax = plt.subplots()
    ax.imshow(slice_data, cmap='gray')
    ax.axis('off')
    
    # Update the canvas with the new figure
    canvas.figure = fig
    canvas.draw()

    # Close the figure after it is drawn to prevent memory leak
    plt.close(fig)  # Close the figure

def next_slice(slice_index, image_data, canvas):
    """Displays the next slice."""
    if slice_index < image_data.shape[0] - 1:
        slice_index += 1
        update_image(slice_index, image_data, canvas)
    return slice_index

def prev_slice(slice_index, image_data, canvas):
    """Displays the previous slice."""
    if slice_index > 0:
        slice_index -= 1
        update_image(slice_index, image_data, canvas)
    return slice_index
