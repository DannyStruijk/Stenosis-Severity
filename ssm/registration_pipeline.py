import numpy as np
import open3d as o3d
from pycpd import DeformableRegistration
import ssm_functions as fun

def load_and_prepare_data(patient_ids, base_path, reconstruction_path, reconstruction_landmarks_path):
    """
    Loads STL meshes and landmarks for patients, loads the reconstruction mesh and landmarks,
    and preprocesses the reconstruction to match the number of points in patient meshes.

    Args:
        patient_ids (list): List of patient IDs to process.
        base_path (str): Base folder path where patient data is stored.
        reconstruction_path (str): Path to the thickened reconstruction point cloud file (.ply).
        reconstruction_landmarks_path (str): Path to the reconstruction landmarks file (.txt).

    Returns:
        pointclouds (list): List of Open3D point clouds for each patient.
        landmarks (list): List of landmarks arrays for each patient.
        reconstruction (o3d.geometry.PointCloud): Preprocessed reconstruction point cloud.
        reconstruction_landmarks (np.ndarray): Landmarks on the reconstruction.
        centroid_template (np.ndarray): Centroid of the template for alignment.
        scaling_factor (float): Scaling factor applied to the reconstruction.
    """
    # Generate file paths for STL meshes and landmarks of patients
    stl_paths, landmark_paths = fun.generate_paths(patient_ids, base_path)
    # Load patient meshes and landmarks
    pointclouds, landmarks = fun.load_meshes_and_landmarks(stl_paths, landmark_paths)

    # Target number of points to match for reconstruction preprocessing (same as first patient)
    target_num_points = np.asarray(pointclouds[0].points).shape[0]

    # Load and preprocess reconstruction mesh and landmarks
    reconstruction, reconstruction_landmarks, centroid_template, scaling_factor = fun.load_and_preprocess_reconstruction(
        reconstruction_path,
        reconstruction_landmarks_path,
        target_num_points
    )

    return pointclouds, landmarks, reconstruction, reconstruction_landmarks, centroid_template, scaling_factor


def perform_registration_pipeline(pointclouds, landmarks, template_pcd, template_landmarks, patient_ids, alpha=100):
    """
    Performs rigid and non-rigid registration of the template reconstruction
    to each patient mesh using their landmarks and point clouds.

    Args:
        pointclouds (list): List of patient point clouds.
        landmarks (list): List of patient landmarks.
        template_pcd (o3d.geometry.PointCloud): Template point cloud (the reconstruction).
        template_landmarks (np.ndarray): Landmarks on the template.
        patient_ids (list): List of patient IDs.
        alpha (float): Smoothness parameter for CPD non-rigid registration.

    Returns:
        registered_pointclouds (list): List of deformed template points aligned to each patient.
    """
    registered_pointclouds = []
    registered_TY = []

    # Loop over patients (skip template at index 0)
    for i in range(1, len(patient_ids)):
        patient_pcd = pointclouds[i]
        patient_landmarks = landmarks[i]

        # Rigid registration of patient landmarks to template landmarks
        T_rigid, aligned_patient_pcd, aligned_patient_landmarks = fun.perform_rigid_registration(
            patient_landmarks, template_landmarks, patient_pcd
        )

        # Compute RMS error between template and aligned patient landmarks
        rms_error = fun.compute_rms_error(template_landmarks, aligned_patient_landmarks)
        print(f"RMS landmark error after rigid registration (patient {patient_ids[i]} to template): {rms_error:.6f}")

        # Non-rigid registration using CPD with configurable alpha
        Y = np.asarray(template_pcd.points)  # Template (static)
        X = np.asarray(aligned_patient_pcd.points)  # Patient points (moving)

        nonrigid_reg = DeformableRegistration(X=X, Y=Y, alpha=alpha)  # alpha controls smoothness
        TY, _ = nonrigid_reg.register()

        registered_pointclouds.append(TY)

        # Optional: Convert to Open3D point cloud for visualization or further processing
        TY_pcd = o3d.geometry.PointCloud()
        TY_pcd.points = o3d.utility.Vector3dVector(TY)
        registered_TY.append(TY_pcd)
        
        # Visualization of the registration results
        aligned_patient_pcd.paint_uniform_color([0, 0, 1])  # Patient leaflet in blue
        TY_pcd.paint_uniform_color([1, 0, 0])       # Deformed template in red
        o3d.visualization.draw_geometries([aligned_patient_pcd, TY_pcd])

        

    return registered_pointclouds


def build_mean_shape(registered_pointclouds, scaling_factor, centroid_template):
    """
    Computes the mean shape from all registered (deformed) template point clouds,
    scales and translates it back to the original coordinate space.

    Args:
        registered_pointclouds (list): List of registered point arrays from each patient.
        scaling_factor (float): Scaling factor previously applied to the template.
        centroid_template (np.ndarray): Centroid used for translation.

    Returns:
        mean_shape_pcd (o3d.geometry.PointCloud): Point cloud of the mean shape.
    """
    # Average the registered points to get mean shape
    mean_shape_points = fun.average_pointcloud(registered_pointclouds)
    # Scale and translate mean shape to original scale and position
    mean_shape_points_scaled = mean_shape_points * scaling_factor
    mean_shape_points_translated = mean_shape_points_scaled + centroid_template

    mean_shape_pcd = o3d.geometry.PointCloud()
    mean_shape_pcd.points = o3d.utility.Vector3dVector(mean_shape_points_translated)
    mean_shape_pcd.paint_uniform_color([1, 0.5, 0])  # Orange color

    return mean_shape_pcd


def save_pointcloud(pcd, path):
    """
    Saves an Open3D point cloud to disk.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud to save.
        path (str): File path where to save the point cloud.
    """
    o3d.io.write_point_cloud(path, pcd)
    print(f"Mean shape saved to: {path}")


def run_registration_pipeline(patient_ids, base_path, reconstruction_path, reconstruction_landmarks_path, output_path, alpha=100):
    """
    Wrapper to run the full registration pipeline:
    - Load patient data and reconstruction
    - Perform rigid and non-rigid registration
    - Build and save the mean shape
    - Optional visualization
    
    Args:
        patient_ids (list): List of patient IDs to process.
        base_path (str): Base folder path for patient data.
        reconstruction_path (str): Path to reconstruction point cloud (.ply).
        reconstruction_landmarks_path (str): Path to reconstruction landmarks (.txt).
        output_path (str): File path to save the mean shape point cloud.
        alpha (float): Smoothness parameter for CPD non-rigid registration.
    """
    # Load data and reconstruction
    pointclouds, landmarks, template_pcd, template_landmarks, centroid, scale = load_and_prepare_data(
        patient_ids, base_path, reconstruction_path, reconstruction_landmarks_path
    )

    # Perform registration with specified alpha
    deformations = perform_registration_pipeline(
        pointclouds, landmarks, template_pcd, template_landmarks, patient_ids, alpha=alpha
    )

    # Build mean shape
    mean_shape = build_mean_shape(deformations, scale, centroid)

    # Save mean shape
    save_pointcloud(mean_shape, output_path)

    # Optional: visualize mean shape and template side-by-side
    # o3d.visualization.draw_geometries([mean_shape, template_pcd])
