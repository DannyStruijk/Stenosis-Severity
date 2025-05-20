# import pyvista as pv
# import numpy as np

# # Load the mesh
# mesh = pv.read(r"H:\DATA\Afstuderen\3.Data\SSM\aos14\aos_14.stl")

# # Extract vertices (x, y, z coordinates) from the mesh
# points = mesh.points

# # Estimate resolution
# bounds = mesh.bounds  # xmin, xmax, ymin, ymax, zmin, zmax
# extent = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]

# # Estimate mean distance between points
# from sklearn.neighbors import NearestNeighbors

# nbrs = NearestNeighbors(n_neighbors=2).fit(points)
# distances, _ = nbrs.kneighbors(points)
# avg_spacing = np.mean(distances[:, 1])

# print(f"Estimated average point spacing: {avg_spacing:.3f}")
# print(f"Bounding box extent: {extent}")

import pyvista as pv
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load STL mesh
mesh = pv.read(r"H:\DATA\Afstuderen\3.Data\SSM\aos14\aos_14.stl")  # Update with your path

# Step 1: Estimate average point spacing
points = mesh.points
nbrs = NearestNeighbors(n_neighbors=2).fit(points)
distances, _ = nbrs.kneighbors(points)
avg_spacing = np.mean(distances[:, 1])
print(f"Average spacing: {avg_spacing:.3f} mm")

# Step 2: Set voxel size slightly below average spacing
voxel_size = round(avg_spacing * 0.95, 3)  # E.g. 0.5 mm
bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
extent = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]

dims = [
    int(np.ceil(extent[0] / voxel_size)),
    int(np.ceil(extent[1] / voxel_size)),
    int(np.ceil(extent[2] / voxel_size)),
]
print(f"Voxel dimensions: {dims}")

# Step 3: Convert mesh to signed distance function
implicit = mesh.compute_implicit_distance(bound=2.0)

# Step 4: Sample over regular grid (voxelize)
grid = implicit.sample_over_regular_grid(
    dimensions=dims,
    bounds=mesh.bounds,
    compute_normals=False
)

# Step 5: Threshold to fill inside (inside = negative)
filled = grid.threshold(value=0.0, invert=True)

# Optional: Visualize result
filled.plot(opacity=0.6)

# Save as VTK or VTI for 3D Slicer if needed
# filled.save("filled_volume.vtk")
