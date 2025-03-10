import tkinter as tk
from tkinter import Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import gui_functions as gf

class DicomSliceViewer:
    def __init__(self, dicom_file_path):
        # Initialize variables
        self.dicom_file_path = dicom_file_path
        self.image_data = gf.load_dicom(dicom_file_path)
        self.slice_index = gf.initialize_slice_index()
        self.landmarks = []
        self.annotating = False

        # Set up the window and canvas
        self.window = tk.Tk()
        self.window.title("DICOM Slice Viewer")
        
        self.window.geometry("600x600")

        # Setup Figure and Axes for Matplotlib
        self.fig = Figure()
        #self.ax = self.fig.add_subplot()  # Add axes to the figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH)

        # Setup Slice Label
        self.slice_label = tk.Label(self.window, text=f"Current Slice: {self.slice_index}")
        self.slice_label.pack()

        # Setup Buttons
        self.create_buttons()
        
        # Setup Labels
        self.instruction = tk.Label(self.window, text = "Please annotate the three commissures.")
        self.instruction.pack(side='bottom')
        self.commissure_1 = tk.Label(self.window, text = "Coordinates 1st commissure:")
        self.commissure_1.pack(side = 'right')
        # self.commissure_2 = tk.Label(self.window, text = "Coordinates 2nd commissure:")
        # self.commissure_2.pack(side = 'right')
        # self.commissure_3 = tk.Label(self.window, text = "Coordinates 3rd commissure:")
        # self.commissure_3.pack(side = 'right')
        

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

    def update_slice(self):
        """Update the image displayed in the GUI with the current slice."""
        gf.update_image(self.slice_index, self.image_data, self.canvas, self.landmarks)
        self.slice_label.config(text=f"Current Slice: {self.slice_index}")  # Update slice label
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def prev_button_func(self):
        """Go to the previous slice."""
        self.slice_index = gf.prev_slice(self.slice_index, self.image_data, self.canvas, self.landmarks)
        self.update_slice()

    def next_button_func(self):
        """Go to the next slice."""
        self.slice_index = gf.next_slice(self.slice_index, self.image_data, self.canvas, self.landmarks)
        self.update_slice()

    def toggle_annotation(self):
        """Toggles the annotation mode."""
        self.annotating = not self.annotating  # Toggle the flag
        if self.annotating:
            #print("Annotation Mode Enabled")
            self.annotate_button.config(text="Disable Annotation")
        else:
            #print("Annotation Mode Disabled")
            self.annotate_button.config(text="Enable Annotation")

    def on_click(self, event):
        """Handle mouse click on the image and add annotation."""
        if event.xdata is None or event.y is None:
            print("Clicked outside the figure. Ignoring.")
            return

        if self.annotating:
            print(f"Click detected at: ({event.xdata}, {event.ydata})")
            x, y = int(event.xdata), int(event.ydata)
            z = self.slice_index
            self.landmarks.append((x, y, z))  # Add the clicked point to the landmarks
            self.update_slice()  # Update the display with the new annotation
            
        if len(self.landmarks)==2:
            self.commissures_done()
            
    def commissures_done(self):
        self.instruction.config(text="Are you done with the commissures?")
        self.annotate_button.config(text="Yes, continue with leaflet tip", command = self.annotate_leaflet)
        self.no_button = Button(self.window, text = "No, start over", command =  self.start_over)
        self.no_button.pack(side="top")    
    
    def start_over(self):
        self.landmarks = []
        self.no_button.destroy()
        self.update_slice()

    def annotate_leaflet(self):
        print("now do the leaflet")

    def run(self):
        """Start the Tkinter main loop."""
        self.window.mainloop()


# Main execution
if __name__ == "__main__":
    dicom_file_path = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\dicoms\dicom_viewer_0002\0002.DCM"
    viewer = DicomSliceViewer(dicom_file_path)
    viewer.run()
