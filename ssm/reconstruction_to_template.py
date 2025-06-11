import vtk
import numpy as np
import open3d as o3d
import os

saved_data = r"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data"

def load_vtk_surface(vtk_file_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    return reader.GetOutput()

def sample_surface_points(surface, distance=0.5):
    sampler = vtk.vtkPolyDataPointSampler()
    sampler.SetInputData(surface)
    sampler.SetDistance(distance)
    sampler.Update()

    points_vtk = sampler.GetOutput().GetPoints()
    n_points = points_vtk.GetNumberOfPoints()
    points_np = np.array([points_vtk.GetPoint(i) for i in range(n_points)])
    return points_np

def create_thickened_point_cloud(points, plane_normal, thickness=2.5, num_samples=5):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    displacements = np.random.uniform(-thickness/2, thickness/2, size=(points.shape[0] * num_samples, 1))
    repeated_points = np.repeat(points, num_samples, axis=0)
    thickened_points = repeated_points + displacements * plane_normal

    pcd_thick = o3d.geometry.PointCloud()
    pcd_thick.points = o3d.utility.Vector3dVector(thickened_points)
    return pcd_thick

def save_thickened_point_cloud(pcd, vtk_file_path):
    """
    Save the thickened point cloud in the same style as your saved data:
    saves to saved_data/<subject_folder>/thickened_points_<subject_number>.ply
    """
    subject_folder = os.path.basename(os.path.dirname(vtk_file_path))  # e.g. 'aos14'
    save_folder = os.path.join(saved_data, subject_folder)
    os.makedirs(save_folder, exist_ok=True)

    subject_number = subject_folder[3:]  # get '14' from 'aos14'
    save_filename = f"thickened_points_{subject_number}.ply"
    save_path = os.path.join(save_folder, save_filename)

    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Thickened point cloud saved to: {save_path}")

def sample_and_thicken_points(vtk_file_path, plane_normal, distance=0.5, thickness=2.5, num_samples=5, save_path=None):
    """
    Full pipeline: load VTK, sample points, create thickened point cloud, and optionally save.

    Args:
        vtk_file_path (str): Path to VTK surface file.
        plane_normal (np.ndarray): Normal vector for thickening.
        distance (float): Sampling distance.
        thickness (float): Thickening distance.
        num_samples (int): Samples per point for thickening.
        save_path (str or None): If given, save thickened point cloud to this path.

    Returns:
        o3d.geometry.PointCloud: Thickened point cloud.
    """
    surface = load_vtk_surface(vtk_file_path)
    points_np = sample_surface_points(surface, distance=distance)
    pcd_thick = create_thickened_point_cloud(points_np, plane_normal, thickness=thickness, num_samples=num_samples)

    if save_path:
        o3d.io.write_point_cloud(save_path, pcd_thick)
        print(f"Saved thickened point cloud to: {save_path}")
