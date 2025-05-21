import vtk
import numpy as np
import open3d as o3d

saved_data = r"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data"
vtk_file = r"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data\ncc_reconstruction.vtk"

# Read surface
reader = vtk.vtkPolyDataReader()
reader.SetFileName(vtk_file)
reader.Update()
surface = reader.GetOutput()

# Sample points densely
sampler = vtk.vtkPolyDataPointSampler()
sampler.SetInputData(surface)
sampler.SetDistance(0.05)
sampler.Update()

points_vtk = sampler.GetOutput().GetPoints()
n_points = points_vtk.GetNumberOfPoints()

points_np = np.array([points_vtk.GetPoint(i) for i in range(n_points)])

# Define the aortic valve plane normal vector (unit vector)
# Example: pointing roughly along z-axis
plane_normal = np.array([0, 0, 1])  
plane_normal = plane_normal / np.linalg.norm(plane_normal)

# Define thickness (distance to expand perpendicular to plane)
thickness = 0.2

# Offset points along plane normal both ways
points_plus = points_np + (thickness / 2) * plane_normal
points_minus = points_np - (thickness / 2) * plane_normal

# Combine original + offsets
thickened_points = np.vstack((points_np, points_plus, points_minus))

# Create Open3D PointCloud
pcd_thick = o3d.geometry.PointCloud()
pcd_thick.points = o3d.utility.Vector3dVector(thickened_points)

# Visualize thickened point cloud
o3d.visualization.draw_geometries([pcd_thick], window_name="Thickness Perpendicular to Plane")

# Save the thickened point cloud as a PLY file
# Save path inside your saved_data folder
save_path = saved_data + r"\thickened_points.ply"
o3d.io.write_point_cloud("thickened_points.ply", pcd_thick)