import tkinter as tk
from tkinter import Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os

os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/gui/")

import gui_functions as gf  # Import your gui_functions module
import pydicom
import matplotlib.pyplot as plt  # Corrected import for matplotlib
import subprocess 
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

# Define the DICOM directory
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"

# Compute normal vector of the annular plane (You already have this)
annular_normal = np.array([ 0.66, 0, 0.746])  # Example

# Get sorted DICOM files based on Z position (using the function from gui_functions)
sorted_dicom_files = gf.get_sorted_dicom_files(dicom_dir)
volume = gf.dicom_to_matrix(sorted_dicom_files)

# Calcualte the rotation needed to look perpendicular to the annular plane
image_data = gf.reslice_numpy_volume(volume, annular_normal)

class DicomSliceViewer:
    def __init__(self, dicom_files):
        # Initialize variables
        self.dicom_files = dicom_files  # Sorted list of DICOM files
        self.slice_index = 25  # Start with the first slice
        self.image_data = image_data  # Corrected function
        self.landmarks = []
        self.annotating = False

        # Set up the window and canvas
        self.window = tk.Tk()
        self.window.title("DICOM Slice Viewer")
        self.window.geometry("600x600")

        # Setup Figure and Axes for Matplotlib
        self.fig = Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH)

        # Setup Slice Label     
        self.slice_label = tk.Label(self.window, text=f"Current Slice: {self.slice_index}")
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
        prev_button = Button(self.window, text="Previous Slice", command=self.prev_button_func)
        prev_button.place(relx=0.0, rely=1.0, anchor="sw", x=10, y=-10)

        next_button = Button(self.window, text="Next Slice", command=self.next_button_func)
        next_button.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

        self.annotate_button = Button(self.window, text="Enable Annotation", command=self.toggle_annotation)
        self.annotate_button.pack(side="bottom")

        self.no_button = Button(self.window, text="No, start over", command=self.start_over)
        self.no_button.pack(side="top")  
        
        self.overlay_button = Button(self.window, text="Overlay", command=self.overlay)
        self.overlay_button.pack(side="right")
        
        print(self.image_data.shape)

    def update_slice(self):
        """Update the image displayed in the GUI with the current slice."""
        ax = self.fig.gca()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()  # Preserve axis limits
        
        ax.clear()
        gf.update_image(self.slice_index, self.image_data, self.canvas, self.landmarks)
        
        ax.set_xlim(xlim)  # Restore axis limits
        ax.set_ylim(ylim)
        
        self.slice_label.config(text=f"Current Slice: {self.slice_index}")
        self.canvas.mpl_connect("button_press_event", self.on_click)


    def prev_button_func(self):
        """Go to the previous slice."""
        if self.slice_index > 0:
            self.slice_index -= 1
            self.update_slice()

    def next_button_func(self):
        """Go to the next slice."""
        if self.slice_index < len(self.dicom_files) - 1:
            self.slice_index += 1
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
            z = self.slice_index
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

    def overlay(self):
        # """Overlays the reconstructed leaflet surface on the CT image in the GUI."""
        # print("Overlaying...")
        # # Load the VTK surface
        # vtk_reader = vtk.vtkPolyDataReader()
        # vtk_reader.SetFileName(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\reconstructions\leaflet_surface_1.vtk")
        # vtk_reader.Update()
    
        # polydata = vtk_reader.GetOutput()
        # points = polydata.GetPoints()
    
        # if points is None:
        #     print("Error: No points found in the VTK file.")
        #     return
    
        # # Convert VTK points to NumPy array
        # np_points = vtk_to_numpy(points.GetData())
    
        # # Extract only points corresponding to the current slice (approximate in Z)
        # slice_z = self.slice_index
        # slice_points = np_points[np.abs(np_points[:, 2] - slice_z) < 2]  # Tolerance for Z match
    
        # if len(slice_points) == 0:
        #     print("No surface points found for this slice.")
        #     return
    
        # # Overlay the points on the image
        # ax = self.fig.gca()
        # ax.scatter(slice_points[:, 0], slice_points[:, 1], c='r', s=10, label="Leaflet Surface")
        # ax.legend()
        
    
        # self.canvas.draw()
        return 0 

    def run(self):
        """Start the Tkinter main loop."""
        self.window.mainloop()

# Main execution
if __name__ == "__main__":
    viewer = DicomSliceViewer(sorted_dicom_files)  # Use the sorted DICOM files
    viewer.run()
