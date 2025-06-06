import vtk
import numpy as np
import open3d as o3d

saved_data = r"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data"
vtk_file = r"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data\aos14\ncc_reconstruction_14.vtk"

# Read surface
reader = vtk.vtkPolyDataReader()
reader.SetFileName(vtk_file)
reader.Update()
surface = reader.GetOutput()

# Sample points densely
sampler = vtk.vtkPolyDataPointSampler()
sampler.SetInputData(surface)
sampler.SetDistance(0.5)
sampler.Update()

points_vtk = sampler.GetOutput().GetPoints()
n_points = points_vtk.GetNumberOfPoints()

points_np = np.array([points_vtk.GetPoint(i) for i in range(n_points)])

# Define the aortic valve plane normal vector (unit vector)
# Example: pointing roughly along z-axis
plane_normal = np.array([0, 0, 1])  
plane_normal = plane_normal / np.linalg.norm(plane_normal)

# Define thickness (distance to expand perpendicular to plane)
thickness = 2.5

# Randomly displace each point along the normal within the thickness range
num_samples = 5  # number of random samples per point
displacements = np.random.uniform(-thickness/2, thickness/2, size=(points_np.shape[0] * num_samples, 1))
repeated_points = np.repeat(points_np, num_samples, axis=0)
thickened_points = repeated_points + displacements * plane_normal

# Create Open3D PointCloud
pcd_thick = o3d.geometry.PointCloud()
pcd_thick.points = o3d.utility.Vector3dVector(thickened_points)

# Visualize thickened point cloud
pcd_thick.paint_uniform_color([1, 0.5, 0])  # orange for mean shape
o3d.visualization.draw_geometries([pcd_thick], window_name="Thickness Perpendicular to Plane")

# Save the thickened point cloud as a PLY file
# Save path inside your saved_data folder
# save_path = saved_data + r"\thickened_points_14.ply"
# o3d.io.write_point_cloud(save_path, pcd_thick)