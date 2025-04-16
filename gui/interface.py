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
        self.evalpts = None

        # Set up the window and canvas
        self.window = tk.Tk()
        self.window.title("DICOM Slice Viewer")
        self.window.geometry("1300x650")

        # Setup Figure and Axes for Matplotlib
        self.fig = Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().place(x=0, y=0)

        # Create another canvas for the coronal view
        self.fig2 = Figure()
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master = self.window)
        self.canvas2.get_tk_widget().place(x=500, y=0)

        # Setup Buttons
        self.create_buttons()

        # Setup Instruction Label
        self.instruction = tk.Label(self.window, text="Please annotate the three commissures.", font=("Helvetica", 12))
        self.instruction.place(relx=0.15, rely=1.0, anchor = 's', x=0, y = -100)

        # Display the first slice
        self.update_slice()

        # Bind Mouse Event to Matplotlib
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def create_buttons(self):
        """Create the navigation and annotation buttons."""
               
        # Create frame for the Next/Previous Slice buttons
        self.slice_frame = tk.Frame(self.window, bg="lightyellow", padx=10, pady=10, height = 150)
        self.slice_frame.place(relx=0.65, rely=1.0, anchor='sw', x=0, y=-10, height = 140, width = 180)
        
        self.slice_label = tk.Label(self.slice_frame, text=f"Current T Slice: {self.slice_index_transversal}", font=("Helvetica", 10, "bold"))
        self.slice_label.pack(side="top", pady=(0,15))
        
        self.prev_button = Button(self.slice_frame, text="Previous Slice", command=self.prev_button_func)
        self.prev_button.pack()
        
        self.next_button = Button(self.slice_frame, text="Next Slice", command=self.next_button_func)
        self.next_button.pack()
        
        # Create frame for the Coronal Slice navigation buttons
        self.coronal_slice_frame = tk.Frame(self.window, bg="lightyellow", padx=10, pady=10, height=150)
        self.coronal_slice_frame.place(relx=0.80, rely=1.0, anchor='sw', x=0, y=-10, height=140, width=180)
        
        self.coronal_slice_label = tk.Label(self.coronal_slice_frame, text=f"Current C Slice: {self.slice_index_coronal}", font=("Helvetica", 10, "bold"))
        self.coronal_slice_label.pack(side="top", pady=(0, 15))
        
        self.prev_coronal_button = Button(self.coronal_slice_frame, text="Previous Slice", command=self.prev_coronal_func)
        self.prev_coronal_button.pack()
        
        self.next_coronal_button = Button(self.coronal_slice_frame, text="Next Slice", command=self.next_coronal_func)
        self.next_coronal_button.pack()

        
        # Create frame for the annotation buttons
        self.annotation_frame = tk.Frame(self.window, bg = "lightblue", padx=10, pady=10)
        self.annotation_frame.place(relx=0.35, rely=1.0, anchor='sw', x=0, y=-10, height = 140, width = 180)
        
        self.annotate_label = tk.Label(self.annotation_frame, text="Not annotating", font=("Helvetica", 10, "bold"))
        self.annotate_label.pack(side="top", pady=(0,15))
        
        self.annotate_button = Button(self.annotation_frame, text="Enable Annotation", command=self.toggle_annotation)
        self.annotate_button.pack()
        
        self.no_button = Button(self.annotation_frame, text="No, start over", command=self.start_over)
        self.no_button.pack()
        

        
        # Create frame for rotating the figure
        self.rotate_frame = tk.Frame(self.window, bg="lightgreen", padx=10, pady=10)
        self.rotate_frame.place(relx=0.50, rely=1.0, anchor='sw', x=0, y=-10, height=140, width=180)
        
        # Place the label at the top with extra bottom padding
        self.rotate_label = tk.Label(self.rotate_frame, text=f"Current Angle: {self.degree}", font=("Helvetica", 10, "bold"))
        self.rotate_label.pack(side="top", pady=(0, 15))  # more space below the label
        
        # Create a sub-frame to hold and vertically center the buttons
        self.rotate_buttons_frame = tk.Frame(self.rotate_frame, bg="lightgreen")
        self.rotate_buttons_frame.pack(expand=True)
        
        # Add buttons to the sub-frame
        self.decrease_button = Button(self.rotate_buttons_frame, text="Decrease angle", command=self.decrease_degree)
        self.decrease_button.pack()
        
        self.increase_button = Button(self.rotate_buttons_frame, text="Increase angle", command=self.increase_degree)
        self.increase_button.pack()
        
        self.rotate_button = Button(self.rotate_buttons_frame, text="Rotate volume", command=self.rotate_and_display)
        self.rotate_button.pack()

        # NOTE: ONILY FOR TESTING PURPOSES!!! OVERLAY SHOULD NORMALLY ONLY BE AFTER RECONSTRUCT
        self.overlay_button = Button(self.annotation_frame, text = "Overlay", command=self.overlay)
        self.overlay_button.pack()



    def update_slice(self):
        """Update the image displayed in the GUI with the current slice."""
        ax = self.fig.gca()
        
        # Save the current limits to restore them later
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        
        ax.clear()  # Clear the previous image
        
        # Update the image with the current slice
        gf.update_transversal(self.slice_index_transversal, self.image_data, self.canvas, self.landmarks, self.evalpts)
        gf.update_coronal(self.slice_index_coronal, rotated_volume, self.canvas2, self.degree, self.evalpts)
        
        # Update the slice label
        self.slice_label.config(text=f"Current T Slice: {self.slice_index_transversal}")
        self.coronal_slice_label.config(text=f"Current C Slice: {self.slice_index_coronal}")

        
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
            
    def prev_coronal_func(self):
        if self.slice_index_coronal > 0:
            self.slice_index_coronal -= 1
            self.update_slice()

    def next_coronal_func(self):
        if self.slice_index_coronal < rescaled_volume.shape[1] - 1:
            self.slice_index_coronal += 1
            self.update_slice()
            
    def decrease_degree(self):
        self.degree -= 1
        self.rotation -= 1
        self.rotate_label.config(text=f"Current Angle: {self.degree}")
        
    def increase_degree(self):
        self.degree += 1
        self.rotation += 1
        self.rotate_label.config(text=f"Current Angle: {self.degree}")
        
        
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
            self.annotate_label.config(text="Currently annotating...")

            
        else:
            self.annotate_button.config(text="Enable Annotation")
            self.annotate_label.config(text="Not annotating")


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
    
        # Update instructions based on how many points have been annotated
        if len(self.landmarks) == 3:
            self.instruction.config(text="Now annotate the center of the aortic valve.")
        elif len(self.landmarks) == 4:
            self.instruction.config(text="Now annotate the three hinge points.")
        elif len(self.landmarks) == 7:
            self.annotation_complete()
            print("The coordinates of the commissures are:", self.landmarks[0:3])
            print("The coordinate of the center is:", self.landmarks[3])
            print("The coordinates of the hinge points are:", self.landmarks[4:7])

    def annotation_complete(self):
        """Disable annotation and create an exit button once all landmarks are set."""
        self.annotating = False
        self.annotate_button.config(state=tk.DISABLED)
        self.instruction.config(text="Annotation complete. Click Reconstruct.")
    
        with open("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/annotations/landmarks.txt", "w") as f:
            for landmark in self.landmarks:
                f.write(f"{landmark[0]} {landmark[1]} {landmark[2]}\n")
    
        reconstruct_button = Button(self.window, text="Reconstruct", command=self.run_script)
        reconstruct_button.place(relx = 0.1, rely=1.0, y=-50, anchor = 's')
    
        exit_button = Button(self.window, text="Exit", command=self.window.destroy)
        exit_button.place(relx = 0.1, rely=1.0, y=-25, anchor = 's')
        
        self.annotate_label.config(text="Done.")
        
        
        
    def overlay(self):
        base_path = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\reconstructions"
        self.evalpts = gf.load_leaflet_points(base_path)
        
    def start_over(self):
        """Reset annotations and allow the user to restart the process."""
        self.landmarks = []
        self.annotate_button.config(state=tk.NORMAL, text="Enable Annotation")
        self.instruction.config(text="Please annotate the three commissures.")
        self.update_slice()
        
    def run_script(self):
        """Execute an external Python script."""
        script_path = r"H:/DATA/Afstuderen/2.Code/Stenosis-Severity/surface_reconstruction/leaflet_interpolation.py"
        subprocess.run(["python", script_path], check=True) 
        self.instruction.config(text="The reconstructions have been made.")
        
        self.overlay_button = Button(self.annotation_frame, text = "Overlay", command=self.overlay)
        self.overlay_button.pack()

    def run(self):
        """Start the Tkinter main loop."""
        self.window.mainloop()

# Main execution
if __name__ == "__main__":
    viewer = DicomSliceViewer(sorted_dicom_files)  # Use the sorted DICOM files
    viewer.run()