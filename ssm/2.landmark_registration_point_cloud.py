import os
import numpy as np
import open3d as o3d
import copy
import ssm_functions as fun
from pycpd import DeformableRegistration

#%%%%%%%%%%%%%%%%%% SETTINGS

patient_ids = [13, 14]
base_path = r"H:\DATA\Afstuderen\3.Data\SSM"

#%%%%%%%%%%%%%%%%%% LOAD DATA

stl_paths, landmark_paths = fun.generate_paths(patient_ids, base_path)
pointclouds, landmarks = fun.load_meshes_and_landmarks(stl_paths, landmark_paths)

#%%%%%%%%%%%%%%%%%% CREATE LANDMARK SPHERES FOR VISUALIZATION

landmark_spheres = fun.create_all_landmark_spheres(landmarks)

#%%%%%%%%%%%%%%%%%% REGISTER ALL TO PATIENT 13 (index 0)

reference_landmarks = landmarks[0]
reference_pcd = pointclouds[0]

# Prepare geometries list with the fixed reference pointcloud painted green
reference_pcd.paint_uniform_color([0, 1, 0])  # green
geometries = [reference_pcd]  # show fixed landmarks as well

for i in range(1, len(patient_ids)):
    moving_landmarks = landmarks[i]
    moving_pcd = pointclouds[i]

    T_rigid, transformed_pcd, transformed_landmarks = fun.perform_rigid_registration(
        moving_landmarks, reference_landmarks, moving_pcd
    )
    
    # Calculate the RMS error of the rigid registration
    rms_error = fun.compute_rms_error(transformed_landmarks, reference_landmarks)
    print(f"RMS landmark error after registering patient {patient_ids[i]} to patient {patient_ids[0]}: {rms_error:.6f}")

    # Preparing the geomtries to be drawn
    transformed_pcd.paint_uniform_color([0, 0, 1])  # blue for moving patient
    transformed_lm_spheres = fun.create_landmark_spheres(transformed_landmarks, [0, 0, 1])
    
    geometries += [transformed_pcd]

#%%%%%%%%%%%%%%%%%% VISUALIZATION

# Perform non-rigid registration
X = np.asarray(pointclouds[0].points)  # fixed/reference point cloud
Y = np.asarray(transformed_pcd.points)  # moving point cloud (after rigid registration)

nonrigid_reg = DeformableRegistration(X=X, Y=Y)
TY, _ = nonrigid_reg.register()

TY_pcd = o3d.geometry.PointCloud()
TY_pcd.points = o3d.utility.Vector3dVector(TY)

# o3d.visualization.draw_geometries(geometries)
pointclouds[0].paint_uniform_color([0, 1, 0])  # green fixed
TY_pcd.paint_uniform_color([1, 0, 0])  # red moving after non-rigid deformation

o3d.visualization.draw_geometries([pointclouds[0], TY_pcd])
