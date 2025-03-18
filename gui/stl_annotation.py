import numpy as np
import pyvista as pv

# Load STL files
mesh1 = pv.read(r"H:\DATA\Afstuderen\3.Data\Harde Schijf Koen\SAVI_AoS\Afstudeerproject Koen Janssens\aos14\Mimics\aos_14.stl")
#mesh2 = pv.read(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\reconstructions\leaflet_surface_1_hinge.vtk")  # Update with the correct path

# Create plotter
plotter = pv.Plotter()
plotter.add_mesh(mesh1, color="lightgray", opacity=0.7)
#plotter.add_mesh(mesh2, color="blue", opacity=0.5)  # Adjust color and opacity as needed

# List to store selected points
commissures = []
labels = ["Commissure 1", "Commissure 2", "Commissure 3", "Leaflet Tip", "Hinge 1", "Hinge 2", "Hinge 3"]

# Callback function for selecting points
def callback(point):
    if len(commissures) < 7:
        landmark = point
        print(f"{labels[len(commissures)]}: {landmark}")
        commissures.append(point)

        # Add new point labels
        plotter.add_point_labels([point], [labels[len(commissures)-1]], 
                                 point_size=10, render_points_as_spheres=True, always_visible=True,
                                 text_color="red")

        # Ensure annotations remain visible
        plotter.render()
    else:
        print("All landmarks selected. Restart the program to reselect.")

# Enable picking with left-click
plotter.enable_surface_point_picking(callback, show_message=True)

# Show interactive window
plotter.show()

# Save the selected points (optional)
if len(commissures) == 7:
    np.savetxt("commissures.txt", np.array(commissures))
    print("Commissures saved to 'commissures.txt'.")
