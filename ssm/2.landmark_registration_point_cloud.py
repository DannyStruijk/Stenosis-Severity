import numpy as np
import open3d as o3d
import ssm_functions as fun
from pycpd import DeformableRegistration
from sklearn.decomposition import PCA
import copy

#%%%%%%%%%%%%%%%%%% SETTINGS

patient_ids = [13, 14, 15]
base_path = r"H:\DATA\Afstuderen\3.Data\SSM"
reconstruction_path = r"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data\thickened_points_14.ply"

#%%%%%%%%%%%%%%%%%% LOAD DATA

stl_paths, landmark_paths = fun.generate_paths(patient_ids, base_path)
pointclouds, landmarks = fun.load_meshes_and_landmarks(stl_paths, landmark_paths)

# Loading the reconstruction and assuring it has the same amoutn of vertices
target_num_points = np.asarray(pointclouds[0].points).shape[0]
reconstruction = fun.load_and_preprocess_reconstruction(reconstruction_path, target_num_points=target_num_points)
print(f"Reconstruction original points: {len(reconstruction.points)}")


#%%%%%%%%%%%%%%%%%% CREATE LANDMARK SPHERES FOR VISUALIZATION

landmark_spheres = fun.create_all_landmark_spheres(landmarks)

#%%%%%%%%%%%%%%%%%% SET TEMPLATE (PATIENT 13)

template_pcd = pointclouds[0]
template_landmarks = landmarks[0]
template_points = np.asarray(template_pcd.points)

template_pcd.paint_uniform_color([0, 1, 0])  # green for template

idx_points = [100,500]

#%%%%%%%%%%%%%%%%%% REGISTER TEMPLATE TO EACH PATIENT AND STORE DEFORMATIONS

registered_pointclouds = []
registered_TY = []

for i in range(1, len(patient_ids)):
    patient_pcd = pointclouds[i]
    patient_landmarks = landmarks[i]
    patient_points = np.asarray(patient_pcd.points)

    # Rigid registration of patient landmarks to template landmarks
    # Note: patient is moving, template is fixed
    T_rigid, aligned_patient_pcd, aligned_patient_landmarks = fun.perform_rigid_registration(
        patient_landmarks, template_landmarks, patient_pcd
    )
    # Create landmark spheres for template and aligned patient landmarks
    template_spheres = fun.create_landmark_spheres(template_landmarks, color=[0, 1, 0], radius=0.03)  # Green
    aligned_spheres = fun.create_landmark_spheres(aligned_patient_landmarks, color=[1, 0, 0], radius=0.03)  # Red
    aligned_patient_pcd.paint_uniform_color([1,0,0])
    # Visualize point clouds + landmark spheres
    o3d.visualization.draw_geometries(
        [template_pcd, aligned_patient_pcd] + template_spheres + aligned_spheres,
        window_name=f"Rigid Registration: Template vs Patient {patient_ids[i]}"
        )

    
    # Visual check correspondence of selected points after rigid registration
    idx_points = [0]  # example indices to check
    fun.visualize_corresponding_points(template_pcd, aligned_patient_pcd, idx_points)

    rms_error = fun.compute_rms_error(template_landmarks, aligned_patient_landmarks)
    print(f"RMS landmark error after rigid registration (patient {patient_ids[i]} to template): {rms_error:.6f}")

    # Non-rigid registration: deform template to the aligned patient (same frame)
    Y = np.asarray(template_pcd.points)  # static template points (same every time)
    X = np.asarray(aligned_patient_pcd.points)  # now in the template frame

    nonrigid_reg = DeformableRegistration(X=X, Y=Y, alpha=100)
    TY, _ = nonrigid_reg.register()
    registered_pointclouds.append(TY)

    # Visualization
    TY_pcd = o3d.geometry.PointCloud()
    TY_pcd.points = o3d.utility.Vector3dVector(TY)
    registered_TY.append(TY_pcd)
    
    TY_pcd.paint_uniform_color([1, 0, 0])  # red for deformed template

    aligned_patient_pcd.paint_uniform_color([0, 1, 0])  # green for aligned patient

    o3d.visualization.draw_geometries([aligned_patient_pcd, TY_pcd])


#%%%%%%%%%%%%%%%%%% BUILD SSM FROM DEFORMATION VECTORS

# Now calculate the average shape directly from registered warped point clouds:
mean_shape_points = fun.average_pointcloud(registered_pointclouds)

mean_shape_pcd = o3d.geometry.PointCloud()
mean_shape_pcd.points = o3d.utility.Vector3dVector(mean_shape_points)
mean_shape_pcd.paint_uniform_color([1, 0.5, 0])  # orange for mean shape

registered_TY[1].paint_uniform_color([0, 1, 0])  # green for aligned patient
o3d.visualization.draw_geometries([mean_shape_pcd])

idx_points=[0,1,3]
fun.visualize_corresponding_points(template_pcd, registered_TY[1], idx_points)

#%%%%%%%%%%%%%%%%%%%% PLAYING WITH THE RECONSTRUCTION

o3d.visualization.draw_geometries([aligned_patient_pcd])
print(len(reconstruction.points))

