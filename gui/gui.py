# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:29:06 2025

@author: u840707
"""


# gui.py
import tkinter as tk
from tkinter import Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os

os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/gui")

import gui_functions as gf

# Path to the DICOM file
dicom_file_path = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\dicoms\dicom_viewer_0002\0002.DCM"

# Load DICOM data
image_data = gf.load_dicom(dicom_file_path)

# Initialize slice index
slice_index = gf.initialize_slice_index()

# Create the main window
window = tk.Tk()
window.title("DICOM Slice Viewer")

# Create a canvas to display the image
canvas = FigureCanvasTkAgg(Figure(), master=window)
canvas.get_tk_widget().pack()

# Create a label to display the current slice index
slice_label = tk.Label(window, text=f"Current Slice: {slice_index}")
slice_label.pack()

# Function to update the slice and canvas
def update_slice():
    """Update the image displayed in the GUI with the current slice."""
    gf.update_image(slice_index, image_data, canvas)
    slice_label.config(text=f"Current Slice: {slice_index}")  # Update the label with the current slice index

# Add buttons for navigation
def prev_button_func():
    global slice_index
    slice_index = gf.prev_slice(slice_index, image_data, canvas)
    update_slice()

def next_button_func():
    global slice_index
    slice_index = gf.next_slice(slice_index, image_data, canvas)
    update_slice()

# Create the previous and next slice buttons
prev_button = Button(window, text="Previous Slice", command=prev_button_func)
prev_button.pack(side="left")

next_button = Button(window, text="Next Slice", command=next_button_func)
next_button.pack(side="right")

# Display the first slice
gf.update_image(slice_index, image_data, canvas)

# Start the Tkinter main loop
window.mainloop()
