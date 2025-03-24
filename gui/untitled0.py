import pyvista as pv
import numpy as np

# Define the new control points
control_points = [
    [(-1.978389948370209, 75.25113054518995, 119.8155938766923), 
     [4.593810043369483, 79.87395213866128, 120.04625655981678], 
     np.array([ 11.16601004,  84.49677373, 120.27691924])],
    
    [[7.992229516757652, 75.08135792354658, 116.63201085467692], 
     [9.73874868052191, 79.85866764485552, 117.43725926651581], 
     [11.166110035109174, 84.4967837321326, 120.27692924294125]],
    
    [np.array([ 13.18538679,  76.23405939, 117.38310994]), 
     [12.17569841037245, 80.36541656018179, 117.38310994427064], 
     [11.165910035109174, 84.49676373213259, 120.27690924294124]],
    
    [[15.451703939133594, 79.7098135712322, 123.48147016800833], 
     [13.464171596991275, 82.17101433850873, 120.8894809713516], 
     [11.166210035109174, 84.4967937321326, 120.27693924294125]],
    
    [(13.120887990942187, 84.28431750408579, 133.02860440765406), 
     [12.14344901302568, 84.3905456181092, 126.65276182529766], 
     [11.165810035109175, 84.49675373213259, 120.27689924294124]]
]

# Flatten the control points into a single list
flattened_points = [item for sublist in control_points for item in sublist]

# Convert to numpy array for plotting
points_np = np.array(flattened_points)

# Create a pyvista plotter
plotter = pv.Plotter()

# Create the point cloud for the control points
point_cloud = pv.PolyData(points_np)

# Add the points to the plot
plotter.add_mesh(point_cloud, color='blue', point_size=10)

# Define the labels for each point
labels = [
    ['commissure_1', 'arch_left_1', 'arch_control_1', 'leaflet_tip'],
    ['hinge_arch_1', 'center_1', 'center', 'leaflet_tip_1'],
    ['hinge', 'center_1', 'center', 'leaflet_tip_2'],
    ['hinge_arch_2', 'center_1', 'center', 'leaflet_tip_3'],
    ['commissure_2', 'arch_right_1', 'arch_control_2', 'leaflet_tip_4']
]

# Loop through the control points and labels
for i, label_set in enumerate(labels):
    for j, label in enumerate(label_set):
        label_position = points_np[i * 4 + j]
        plotter.add_point_labels([label_position], [label], point_size=20, font_size=12)


mesh = pv.read(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\reconstructions\leaflet_surface_1_interpolated.vtk")
plotter.add_mesh(mesh, color="lightgray", opacity=0.7)


# Display the plot
plotter.show()
