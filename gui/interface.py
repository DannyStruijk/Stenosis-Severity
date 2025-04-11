import tkinter as tk
from tkinter import Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import matplotlib.pyplot as plt

os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/gui/")

import gui_functions as gf  # Import your gui_functions module
import pydicom
import subprocess 
import numpy as np

#%%%% Loading the data for the GUI

# Define the DICOM directory
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"

# Get sorted DICOM files based on Z position (using the function from gui_functions)
sorted_dicom_files = gf.get_sorted_dicom_files(dicom_dir)
volume = gf.dicom_to_matrix(sorted_dicom_files)

# Load the first DICOM file for the image properties
dicom = pydicom.dcmread(sorted_dicom_files[0][0])
rescaled_volume = gf.rescale_volume(dicom, volume)
rescaled_volume = np.transpose(rescaled_volume, (2, 1, 0))

# Optional: Flip left-right to ensure anatomical orientation
rescaled_volume = np.fliplr(rescaled_volume)

# Determine the angle and the plane in which you roate
angle = np.radians(45)
axis = [0,1,0]
R = gf.rotation_matrix(axis, angle)

# Rotate the volume around the specified axis
rotated_volume = gf.rotated_volume(rescaled_volume, R)

# Calcualte the rotation needed to look perpendicular to the annular plane
image_data = rotated_volume


#%%%%%% Actual GUI

class DicomSliceViewer:
    def __init__(self, dicom_files):
        # Initialize variables
        self.dicom_files = dicom_files  # Sorted list of DICOM files
        self.slice_index_transversal = 107  # Start with the first slice
        self.slice_index_coronal = 250# Start with the first slice
        self.image_data = image_data  # Corrected function
        self.landmarks = [] # List to save the landmarks
        self.annotating = False # State whether user is annotating
        self.degree = 45 # Determine the degree on which the volume is rotated.
        self.axis = [0,1,0] # Determine the axis for the rotation for the volume
        self.rotation = 0

        # Set up the window and canvas
        self.window = tk.Tk()
        self.window.title("DICOM Slice Viewer")
        self.window.geometry("1400x900")

        # Setup Figure and Axes for Matplotlib
        self.fig = Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().place(x=0, y=0)

        # Create another canvas for the coronal view
        self.fig2 = Figure()
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master = self.window)
        self.canvas2.get_tk_widget().place(x=500, y=0)


        # Setup Slice Label     
        self.slice_label = tk.Label(self.window, text=f"Current Slice: {self.slice_index_transversal}")
        self.slice_label.pack()

        # Setup Buttons
        self.create_buttons()

        # Setup Instruction Label
        self.instruction = tk.Label(self.window, text="Please annotate the three commissures.")
        self.instruction.pack(side='bottom')

        # Display the first slice
        self.update_slice()

        # Bind Mouse Event to Matplotlib
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def create_buttons(self):
        """Create the navigation and annotation buttons."""
        self.prev_button = Button(self.window, text="Previous Slice", command=self.prev_button_func)
        self.prev_button.place(relx=0.0, rely=1.0, anchor="sw", x=10, y=-10)

        self.next_button = Button(self.window, text="Next Slice", command=self.next_button_func)
        self.next_button.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)
        
        self.annotate_button = Button(self.window, text="Enable Annotation", command=self.toggle_annotation)
        self.annotate_button.place(relx= 0.5, rely=1.0, anchor="sw", x=10, y=-20)
        
        self.no_button = Button(self.window, text="No, start over", command=self.start_over)
        self.no_button.place(relx= 0.5, rely=1.0, anchor="sw", x=10, y=-50)
        
        self.decrease_button = Button(self.window, text=str(self.degree - 1), command=self.decrease_degree)
        self.decrease_button.place(relx= 0.7, rely=1.0, anchor="sw", x=10, y=-20)
        
        self.increase_button = Button(self.window, text=str(self.degree + 1), command=self.increase_degree)
        self.increase_button.place(relx= 0.7, rely=1.0, anchor="sw", x=10, y=-40)
        
        self.rotate_button = Button(self.window, text="Rotate volume", command = self.rotate_and_display)
        self.rotate_button.place(relx= 0.8, rely=1.0, anchor="sw", x=10, y=-20)

        
        print(self.image_data.shape)

    def update_slice(self):
        """Update the image displayed in the GUI with the current slice."""
        ax = self.fig.gca()
        
        # Save the current limits to restore them later
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        
        ax.clear()  # Clear the previous image
        
        # Update the image with the current slice
        gf.update_transversal(self.slice_index_transversal, self.image_data, self.canvas, self.landmarks)
        gf.update_coronal(self.slice_index_coronal, rescaled_volume, self.canvas2, self.degree)
        
        # Update the slice label
        self.slice_label.config(text=f"Current Slice: {self.slice_index_transversal}")
        
        # Redraw the canvas to reflect the updated image
        self.canvas.draw()
        
        # Rebind the click event (in case it gets unbound)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def prev_button_func(self):
        """Go to the previous slice."""
        if self.slice_index_transversal > 0:
            self.slice_index_transversal -= 1
            self.update_slice()

    def next_button_func(self):
        """Go to the next slice."""
        if self.slice_index_transversal < 274 - 1:
            self.slice_index_transversal += 1
            self.update_slice()
            
    def decrease_degree(self):
        self.degree -= 1
        self.rotation -= 1
        self.decrease_button.config(text=str(self.degree - 1))
        self.increase_button.config(text=str(self.degree + 1))
        
    def increase_degree(self):
        self.degree += 1
        self.rotation += 1
        self.decrease_button.config(text=str(self.degree - 1))
        self.increase_button.config(text=str(self.degree + 1))
        
        
    def rotate_and_display(self):
        # Determine the angle and the plane in which you rotate
        angle = np.radians(self.rotation)
        R = gf.rotation_matrix(self.axis, angle)

        # Rotate the volume around the specified axis
        self.image_data = gf.rotated_volume(self.image_data, R)
        self.rotation=0
        self.update_slice()

    def toggle_annotation(self):
        """Toggles the annotation mode."""
        self.annotating = not self.annotating  # Toggle the flag
        if self.annotating:
            self.annotate_button.config(text="Disable Annotation")
        else:
            self.annotate_button.config(text="Enable Annotation")

    def on_click(self, event):
        """Handle mouse click on the image and add annotation."""
        if event.xdata is None or event.ydata is None:
            print("Clicked outside the figure. Ignoring.")
            return

        if self.annotating:
            x, y = int(event.xdata), int(event.ydata)
            z = self.slice_index_transversal
            self.landmarks.append((x, y, z))  # Add the clicked point to the landmarks
            self.update_slice()  # Update the display with the new annotation

        if len(self.landmarks) == 3:
            self.instruction.config(text="Now annotate the leaflet tip")
        elif len(self.landmarks) == 4:
            self.annotation_complete()
            print("The coordinates of the commissures are: ", self.landmarks[0:3])
            print("And the coordinates of the leaflet tip is: ", self.landmarks[3])

    def annotation_complete(self):
        """Disable annotation and create an exit button once all landmarks are set."""
        self.annotating = False
        self.annotate_button.config(state=tk.DISABLED)
        self.instruction.config(text="Annotation complete. You may exit.")
    
        with open("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/b-spline_fitting/landmarks.txt", "w") as f:
            for landmark in self.landmarks:
                f.write(f"{landmark[0]} {landmark[1]} {landmark[2]}\n")
                
        reconstruct_button = Button(self.window, text = "Reconstruct", command = self.run_script)
        reconstruct_button.pack(side="right")
    
        exit_button = Button(self.window, text="Exit", command=self.window.destroy)
        exit_button.pack(side="bottom")

    def start_over(self):
        """Reset annotations and allow the user to restart the process."""
        self.landmarks = []
        self.annotate_button.config(state=tk.NORMAL, text="Enable Annotation")
        self.instruction.config(text="Please annotate the three commissures.")
        self.update_slice()
        
    def run_script(self):
        """Execute an external Python script."""
        script_path = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\b-spline_fitting\leaflet_surface_NURBS.py"  
        subprocess.run(["python", script_path], check=True) 

    def run(self):
        """Start the Tkinter main loop."""
        self.window.mainloop()

# Main execution
if __name__ == "__main__":
    viewer = DicomSliceViewer(sorted_dicom_files)  # Use the sorted DICOM files
    viewer.run()