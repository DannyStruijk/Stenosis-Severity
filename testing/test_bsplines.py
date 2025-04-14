import numpy as np
import pyvista as pv

control_points = [
    [
        np.array([-1.978389948370209, 75.25113054518995, 119.8155938766923]),
        np.array([4.593810043369483, 79.87395213866128, 117.1182990827481]),
        np.array([11.16601004, 84.49677373, 120.27691924])
    ],
    [
        np.array([7.992229516757652, 75.08135792354658, 116.63201085467692]),
        np.array([12.17569841037245, 80.36541656018179, 117.38310994427064]),
        np.array([11.166110035109174, 84.4967837321326, 120.27692924294125])
    ],
    [
        np.array([13.18538679, 76.23405939, 117.38310994]),
        np.array([12.17569841037245, 80.36541656018179, 117.38310994427064]),
        np.array([11.165910035109174, 84.49676373213259, 120.27690924294124])
    ],
    [
        np.array([15.451703939133594, 79.7098135712322, 123.48147016800833]),
        np.array([12.17569841037245, 80.36541656018179, 117.38310994427064]),
        np.array([11.166210035109174, 84.4967937321326, 120.27693924294125])
    ],
    [
        np.array([13.120887990942187, 84.28431750408579, 133.02860440765406]),
        np.array([12.14344901302568, 84.3905456181092, 123.56367007346114]),
        np.array([11.165810035109175, 84.49675373213259, 120.27689924294124])
    ]
]


# Define labels for each control point
labels = [
    ["commissure_1", "arch_control_1", "leaflet_tip"],
    ["hinge_arch_1", "center", "leaflet_tip"],
    ["hinge",  "center", "leaflet_tip"],
    ["hinge_arch_2", "center", "leaflet_tip"],
    ["commissure_2", "arch_control_2", "leaflet_tip"]
]

# Flatten the points and labels for easier processing
points = np.array([pt for row in control_points for pt in row])
flat_labels = [label for row in labels for label in row]

# Create a PyVista point cloud from the control points
point_cloud = pv.PolyData(points)

# Create a plotter
plotter = pv.Plotter()

# Add the control points as spheres
plotter.add_mesh(point_cloud, color="red", point_size=15, render_points_as_spheres=True)

# Add labels to each control point
plotter.add_point_labels(points, flat_labels, point_color="black", point_size=10, font_size=10)

# Add the leaflet surfaces (replace with your actual file paths)
mesh = pv.read(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\reconstructions\leaflet_surface_1_interpolated.vtk")
mesh2 = pv.read(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\reconstructions\added_leaflet_surface_1.vtk")

# Add the meshes for the leaflet
plotter.add_mesh(mesh, color="lightgray", opacity=0.7)
plotter.add_mesh(mesh2, color="lightgray", opacity=0.7)

# Show the plot
plotter.show()
