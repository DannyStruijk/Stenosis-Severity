import numpy as np
import pyvista as pv

# Load STL file
mesh = pv.read(r"H:\DATA\Afstuderen\3.Data\Harde Schijf Koen\SAVI_AoS\Afstudeerproject Koen Janssens\aos14\Mimics\aos_14.stl")

# Create plotter
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightgray", opacity=0.7)

# List to store selected commissure points
commissures = []
labels = ["Commissure 1", "Commissure 2", "Commissure 3"]

# Callback function for selecting points
def callback(point):
    if len(commissures) < 3:
        print(f"{labels[len(commissures)]}: {point}")
        commissures.append(point)

        # Add new point labels without clearing the previous ones
        plotter.add_point_labels([point], [labels[len(commissures)-1]], 
                                 point_size=10, render_points_as_spheres=True, always_visible=True,
                                 text_color="red")

        # Ensure annotations remain visible
        plotter.render()

    else:
        print("All three commissures selected. Restart the program to reselect.")

# Enable picking with left-click
plotter.enable_surface_point_picking(callback, show_message=True)

# Show interactive window
plotter.show()

# Save the selected points (optional)
if len(commissures) == 3:
    np.savetxt("commissures.txt", np.array(commissures))
    print("Commissures saved to 'commissures.txt'.")
