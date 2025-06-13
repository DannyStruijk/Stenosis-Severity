import os
import numpy as np
import open3d as o3d
import copy
import matplotlib as plt

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
        os.path.join(base_path, "patient_database", f"aos{pid}", "cusps", "rcc", f"rcc_simplified_mesh_aos{pid}.stl")
        for pid in patient_ids
    ]
    landmark_paths = [
        os.path.join(base_path, "patient_database", f"aos{pid}", "landmarks", f"landmarks_rcc_patient_{pid}.txt")
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

def vectorize_pointcloud(pcd):
    return np.asarray(pcd.points).flatten()

def pointwise_distances(pcd1, pcd2):
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    return np.linalg.norm(pts1 - pts2, axis=1)

def colorize_pointcloud_distance(pcd1, pcd2):
    distances = np.linalg.norm(np.asarray(pcd1.points) - np.asarray(pcd2.points), axis=1)
    # Normalize distances to [0,1] for color mapping
    distances_normalized = (distances - distances.min()) / (distances.max() - distances.min())

    # Use a colormap, e.g. 'jet'
    cmap = plt.cm.get_cmap('jet')
    colors = cmap(distances_normalized)[:, :3]  # RGB only

    pcd1.colors = o3d.utility.Vector3dVector(colors)
    return pcd1, distances

def average_pointcloud(pointclouds):
    """
    Calculate the average point cloud from a list of point clouds.

    Parameters:
    -----------
    pointclouds : list of np.ndarray
        List where each element is a numpy array of shape (n_points, 3),
        representing a point cloud with consistent point ordering.

    Returns:
    --------
    avg_pointcloud : np.ndarray
        The average point cloud, shape (n_points, 3).
    """
    if not pointclouds:
        raise ValueError("Input list is empty")

    # Check that all point clouds have the same shape
    n_points = pointclouds[0].shape[0]
    for pc in pointclouds:
        if pc.shape != (n_points, 3):
            raise ValueError("All point clouds must have the same shape and 3 coordinates per point")

    # Stack into a 3D array (n_samples, n_points, 3)
    stacked = np.stack(pointclouds, axis=0)

    # Compute mean along the samples axis
    avg_pointcloud = np.mean(stacked, axis=0)

    return avg_pointcloud

def visualize_corresponding_points(pcd1, pcd2, idx_points, sphere_radius=0.05):
    """
    Visualize two point clouds with corresponding points highlighted as spheres.

    Parameters:
    -----------
    pcd1, pcd2 : open3d.geometry.PointCloud
        The two point clouds to visualize.
    idx_points : list of int
        Indices of points to highlight in both point clouds.
    sphere_radius : float, optional
        Radius of the spheres highlighting the points (default: 0.005).
    """
    points_1 = np.asarray(pcd1.points)
    points_2 = np.asarray(pcd2.points)

    # Paint point clouds with distinct colors
    pcd1.paint_uniform_color([1, 0.7, 0.7])  # light red
    pcd2.paint_uniform_color([0.7, 0.7, 1])  # light blue

    highlight_spheres = []

    for idx in idx_points:
        sphere_1 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere_1.translate(points_1[idx])
        sphere_1.paint_uniform_color([1, 0, 0])  # red for pcd1
        highlight_spheres.append(sphere_1)

        sphere_2 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere_2.translate(points_2[idx])
        sphere_2.paint_uniform_color([0, 0, 1])  # blue for pcd2
        highlight_spheres.append(sphere_2)

    o3d.visualization.draw_geometries([pcd1, pcd2, *highlight_spheres])

def load_and_preprocess_reconstruction(path, lm_path, target_num_points=None):
    pcd = o3d.io.read_point_cloud(path)
    print(np.asarray(pcd.points))
    if pcd.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {path}")

    if target_num_points is not None:
        current_num_points = np.asarray(pcd.points).shape[0]
        if current_num_points > target_num_points:
            # Simple random downsampling of points
            points = np.asarray(pcd.points)
            indices = np.random.choice(current_num_points, target_num_points, replace=False)
            downsampled_points = points[indices]
            pcd.points = o3d.utility.Vector3dVector(downsampled_points)
        elif current_num_points < target_num_points:
            print(f"Warning: Point cloud has fewer points ({current_num_points}) than target ({target_num_points}). No upsampling done.")
    pcd, centroid = center_point_cloud(pcd)
    print("The centroid of the template is: ", centroid)
    pcd, scale = normalize_scale(pcd)
    print("The scaling factor that was used was: ", scale)
    
    lm = np.loadtxt(lm_path)
    lm_centered = lm - centroid
    lm_scaled = lm_centered / scale
    return pcd, lm_scaled, centroid, scale
