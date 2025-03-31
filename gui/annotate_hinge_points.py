import numpy as np
import pyvista as pv

# Load STL files
mesh1 = pv.read(r"H:\DATA\Afstuderen\3.Data\Harde Schijf Koen\AoS Stress\Afstudeerproject Koen Janssens\aos14\Mimics\aos_14.stl")

# Create plotter
plotter = pv.Plotter()
plotter.add_mesh(mesh1, color="lightgray", opacity=0.7)

# List to store selected points
hinge_points = []
labels = ["Hinge_1", "Hinge_2", "Hinge_3"]

# Callback function for selecting points
def callback(point):
    if len(hinge_points) < 7:
        landmark = point
        print(f"{labels[len(hinge_points)]}: {landmark}")
        hinge_points.append(point)

        # Add new point labels
        plotter.add_point_labels([point], [labels[len(hinge_points)-1]], 
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
if len(hinge_points) == 3:
    np.savetxt("hinge_points.txt", np.array(hinge_points))
    print("Hinge_points saved to 'hinge_points.txt'.")
