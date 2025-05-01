import pyvista as pv
import numpy as np

# Load the mesh
mesh = pv.read(r"H:\DATA\Afstuderen\3.Data\Edited STLs\aos14\aos_14.stl")

# Extract vertices (x, y, z coordinates) from the mesh
points = mesh.points

# Create a dictionary to store the highest z-value for each unique (x, y) coordinate
unique_points = {}

for point in points:
    x, y, z = point
    # Use (x, y) as the key, and keep the point with the highest z-value
    if (x, y) not in unique_points:
        unique_points[(x, y)] = point
    else:
        if z > unique_points[(x, y)][2]:  # Compare z-coordinate
            unique_points[(x, y)] = point

# Convert the dictionary of unique points into an array of points
unique_vertices = np.array(list(unique_points.values()))

# Create a new mesh with the unique points
# This will essentially create the inner surface by only keeping the points with the highest z for each (x, y)
inner_surface_mesh = pv.PolyData(unique_vertices)

# Visualize the mesh to confirm it looks like the desired surface
inner_surface_mesh.plot()

# Optionally, save the resulting mesh to an STL file
inner_surface_mesh.save("inner_surface.stl")
