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

# Store landmarks as (x, y) coordinates
landmarks = []

# Create the main window
window = tk.Tk()
window.title("DICOM Slice Viewer")

# Create a canvas to display the image
canvas = FigureCanvasTkAgg(Figure(), master=window)
canvas.get_tk_widget().pack()

# Create a label to display the current slice index
slice_label = tk.Label(window, text=f"Current Slice: {slice_index}")
slice_label.pack()

# Boolean flag to determine whether annotations are allowed
annotating = False

# Function to update the slice and canvas
def update_slice():
    """Update the image displayed in the GUI with the current slice."""
    gf.update_image(slice_index, image_data, canvas, landmarks)
    slice_label.config(text=f"Current Slice: {slice_index}")  # Update the label with the current slice index
    canvas.draw_idle()

def prev_button_func():
    global slice_index
    slice_index = gf.prev_slice(slice_index, image_data, canvas, landmarks)
    update_slice()

def next_button_func():
    global slice_index
    slice_index = gf.next_slice(slice_index, image_data, canvas, landmarks)
    update_slice()


# Create the previous and next slice buttons
prev_button = Button(window, text="Previous Slice", command=prev_button_func)
prev_button.pack(side="left")

next_button = Button(window, text="Next Slice", command=next_button_func)
next_button.pack(side="right")

# Function to toggle the annotation mode
def toggle_annotation():
    """Toggles the annotation mode."""
    global annotating
    annotating = not annotating  # Toggle the flag
    if annotating:
        annotate_button.config(text="Disable Annotation")
    else:
        annotate_button.config(text="Enable Annotation")

# Create the annotate button
annotate_button = Button(window, text="Enable Annotation", command=toggle_annotation)
annotate_button.pack(side="bottom")

# Function to handle mouse click event for annotations
def on_click(event):
    """Handle mouse click on the image and add annotation."""
    print("Click!")
    if annotating:
        # Get the mouse click position in pixel coordinates
        x, y = int(event.x), int(event.y)
        z = slice_index

        # Add the clicked point to the landmarks list
        landmarks.append((x, y, z))
        print(landmarks)
        
        # Update the displayed image with the new annotation
        update_slice()

print("Connecting mouse click event...")
canvas.figure.canvas.mpl_connect("button_press_event", on_click)
print("Mouse click event connected!")

# Display the first slice
gf.update_image(slice_index, image_data, canvas, landmarks)

# Start the Tkinter main loop
window.mainloop()

