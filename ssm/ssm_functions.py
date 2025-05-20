import os
import numpy as np
import open3d as o3d
import copy


# Helper function to center point clouds
def center_point_cloud(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    pcd.translate(-centroid)
    return pcd, centroid

def execute_global_registration(source, target, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000))
    return result

def estimate_voxel_size(pcd, k=10):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    distances = []

    for point in pcd.points:
        [_, idx, dists] = pcd_tree.search_knn_vector_3d(point, k)
        if len(dists) > 1:
            distances.append(np.sqrt(np.mean(dists[1:])))  # skip the first (distance to self)

    avg_dist = np.mean(distances)
    return avg_dist * 2  # slightly larger to downsample

def normalize_scale(pcd):
    points = np.asarray(pcd.points)
    max_range = np.max(np.linalg.norm(points, axis=1))
    pcd.scale(1.0 / max_range, center=(0, 0, 0))
    return pcd, max_range    
    

def get_rigid_transform(A, B):
    """
    Computes the best-fit rigid transform that maps A to B.
    A and B are Nx3 numpy arrays of corresponding points.
    Returns a 4x4 transformation matrix.
    """
    assert A.shape == B.shape

    # Centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # Covariance matrix
    H = AA.T @ BB

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection correction
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    # Compose 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def generate_paths(patient_ids, base_path):
    stl_paths = [
        os.path.join(base_path, f"aos{pid}", "cusps", f"ncc_simplified_mesh_{pid}.stl")
        for pid in patient_ids
    ]
    landmark_paths = [
        os.path.join(base_path, f"aos{pid}", "landmarks", f"landmarks_ncc_patient_{pid}.txt")
        for pid in patient_ids
    ]
    return stl_paths, landmark_paths

def load_meshes_and_landmarks(stl_paths, landmark_paths):
    pointclouds = []
    landmarks = []

    for pcd_path, lm_path in zip(stl_paths, landmark_paths):
        mesh = o3d.io.read_triangle_mesh(pcd_path)
        if mesh.is_empty():
            print(f"Warning: {pcd_path} failed to load or is empty.")
            continue

        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=5000)

        pcd, centroid = center_point_cloud(pcd)
        pcd, scale = normalize_scale(pcd)
        pointclouds.append(pcd)

        lm = np.loadtxt(lm_path)
        lm_centered = lm - centroid
        lm_scaled = lm_centered / scale
        landmarks.append(lm_scaled)

    return pointclouds, landmarks

def create_landmark_spheres(landmarks, color, radius=0.02):
    spheres = []
    for pt in landmarks:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    return spheres

def create_all_landmark_spheres(landmarks_list):
    spheres = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]  # Add more if needed
    for i, lm in enumerate(landmarks_list):
        color = colors[i % len(colors)]
        spheres += create_landmark_spheres(lm, color)
    return spheres

def perform_rigid_registration(lm_src, lm_tgt, pcd_src):
    trans_init = get_rigid_transform(lm_src, lm_tgt)
    print("Initial guess (from landmarks):\n", trans_init)

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(lm_src)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(lm_tgt)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        src, tgt, 1.0, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100000)
    )

    T = reg_p2p.transformation
    print("Rigid transformation matrix:\n", T)

    pcd_transformed = copy.deepcopy(pcd_src)
    pcd_transformed.transform(trans_init)

    lm_src_h = np.hstack((lm_src, np.ones((lm_src.shape[0], 1))))
    transformed_landmarks = (T @ lm_src_h.T).T[:, :3]

    return T, pcd_transformed, transformed_landmarks

def compute_rms_error(pred, gt):
    diff = pred - gt
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))