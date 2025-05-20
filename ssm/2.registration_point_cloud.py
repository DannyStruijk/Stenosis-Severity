import os
import numpy as np
import open3d as o3d
import ssm_functions as fun
import copy

#%%%%%%%%%%%%%%%%%% LOADING THE SIMPLIFIED MESHES

# List of the paths to your simplified STL files
stl_paths = [
    r"H:\DATA\Afstuderen\3.Data\SSM\aos13\cusps\ncc_simplified_mesh_13.stl",
    r"H:\DATA\Afstuderen\3.Data\SSM\aos14\cusps\ncc_simplified_mesh_14.stl",
    r"H:\DATA\Afstuderen\3.Data\SSM\aos15\cusps\ncc_simplified_mesh_15.stl"
]

# Load triangle meshes and convert to point clouds
pointclouds = []

# load the STLs from the paths and convert to pointclouds. Also bring centroid to0 
for path in stl_paths:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        print(f"Warning: {path} failed to load or is empty.")
        continue
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=5000)
    pcd = fun.center_point_cloud(pcd)
    pointclouds.append(pcd)
    print(f"Loaded, sampled, and centered point cloud from: {path}")

#%%%%%%%%%%%%%%%%%%%%% ICP-ONLY REGISTRATION (mesh 14 to mesh 13)

# Define the source and the targetp oint cloud
source = pointclouds[1]  # mesh 14
target = pointclouds[0]  # mesh 13

# Scaling the pointclouds 
fun.normalize_scale(source)
fun.normalize_scale(target)

# Estimate voxel size, needed for global registration
voxel_size = fun.estimate_voxel_size(source)
print(voxel_size)

# Preprocessing the source and the target to get FPFH (features) vectors 
source, source_fpfh = fun.preprocess_point_cloud(source, voxel_size)
target, target_fpfh = fun.preprocess_point_cloud(target, voxel_size)

## Global registration using defined functions in ssm_functions
result_ransac=  fun.execute_global_registration(source, target,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
# Transform the source
source.transform(result_ransac.transformation)
source_copy = copy.deepcopy(source)

#%%%%%%%%%%%%%%%%%%%%%% FINE REGISTRATION

# # Registration by means of ICP (Iterative Closest Point)
# result_icp = fun.refine_registration(source, target,
#                                       voxel_size, result_ransac)

# # Apply the calculated transformation on the copy in order to compare copy to original 
# source_copy.transform(result_icp.transformation)

#%%%%%%%%%%%%%%%%%%%%%%% VISUALIZATION

o3d.visualization.draw_geometries([
    source.paint_uniform_color([1, 0, 0]),  # Red: registered source
    #source_copy.paint_uniform_color([0, 1, 0]),   # Green: target
    target.paint_uniform_color([0,0,1]) # blue 
])

print(result_ransac)

#%%%%%%%%%%%%%%%%%%%% SAVING THE RESULT

save_dir = r"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data"
fun.save_registered_pointclouds_as_stl(source_copy, target, save_dir)

# print(result_icp.transformation)
# print(result_ransac.transformation)
