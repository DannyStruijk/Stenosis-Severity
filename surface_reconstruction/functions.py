import numpy as np
import vtk
from geomdl import fitting
import os
from scipy.ndimage import zoom
from scipy.ndimage import affine_transform
from skimage.draw import polygon2mask
from scipy.interpolate import splprep, splev
import pydicom
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops
from scipy.ndimage import binary_closing
from skimage.morphology import cube
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from skimage.filters import sobel
from scipy.ndimage import gaussian_filter
from skimage import exposure
import trimesh
from skimage import measure
from skimage.draw import line
from scipy.ndimage import binary_dilation
from geomdl import fitting


def calc_leaflet_landmarks(commissure_1, commissure_2, commissure_3, hinge_1, hinge_2, hinge_3):
    # Calculate midpoints between commissures
    mid_1_2 = np.mean([commissure_1, commissure_2], axis=0)
    mid_2_3 = np.mean([commissure_2, commissure_3], axis=0)
    mid_3_1 = np.mean([commissure_3, commissure_1], axis=0)

    # Calculate distances between midpoints and hinges
    dist = {
        ((commissure_1[0], commissure_1[1], commissure_1[2]), (commissure_2[0], commissure_2[1], commissure_2[2])): [np.linalg.norm(mid_1_2 - hinge_1), np.linalg.norm(mid_1_2 - hinge_2), np.linalg.norm(mid_1_2 - hinge_3)],
        ((commissure_2[0], commissure_2[1], commissure_2[2]), (commissure_3[0], commissure_3[1], commissure_3[2])): [np.linalg.norm(mid_2_3 - hinge_1), np.linalg.norm(mid_2_3 - hinge_2), np.linalg.norm(mid_2_3 - hinge_3)],
        ((commissure_3[0], commissure_3[1], commissure_3[2]), (commissure_1[0], commissure_1[1], commissure_1[2])): [np.linalg.norm(mid_3_1 - hinge_1), np.linalg.norm(mid_3_1 - hinge_2), np.linalg.norm(mid_3_1 - hinge_3)]
    }

    # Assign hinges to commissures based on minimum distance
    cusp_landmarks = [
        [commissure_pair[0], commissure_pair[1], [hinge_1, hinge_2, hinge_3][np.argmin(dists)]]
        for commissure_pair, dists in dist.items()
    ]

    return cusp_landmarks


def fit_spline_pts(point_1: list, point_2: list, arch_control: list):
    """
    Fit a B-spline for three points, in order to determine additional control points for the surface.
    This is done in order to broaden the control point grid from 3x3 to 5x5.

    Parameters:
        - point_1: First point.
        - point_2: Second point.
        - arch_control: The point that defines the arch and lies between the two points.
    
    Returns:
        - arch_1: The point that lies between the arch control and point_1.
        - arch_2: The point that lies between the arch control and point_2.
    """
    # Create the B-spline curve using the points
    points = [point_1, arch_control, point_2]
    curve = fitting.interpolate_curve(points, degree=2)
    
    # Evaluate the curve at different parameter values (u values from 0 to 1)
    u_vals = np.linspace(0, 1, 100)  # Evaluate at 100 points along the curve
    eval_pts = [curve.evaluate_single(u) for u in u_vals]
    
    # Define the points where we want to show the additional control points
    arch_1, arch_2 = eval_pts[25], eval_pts[75]
    
    return arch_1, arch_2


def save_surface_evalpts(surface, save_dir = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity-backup\temp", filename="lcc"):
    """
    Saves the evaluated points (evalpts) of a NURBS surface to both .npy and .txt files.

    Parameters:
        surface : geomdl.NURBS.Surface
            The surface object after evaluation.
        filename : str
            Base filename (without extension). Files will be saved in the temp folder.
    """
    if not surface.evalpts:
        raise ValueError("Surface has not been evaluated. Run surface.evaluate() first.")

    # Convert evalpts to NumPy array
    evalpts_array = np.array(surface.evalpts)

    # Define save path
    os.makedirs(save_dir, exist_ok=True)

    # Full paths
    npy_path = os.path.join(save_dir, filename + ".npy")
    txt_path = os.path.join(save_dir, filename + ".txt")

    # Save
    np.save(npy_path, evalpts_array)
    np.savetxt(txt_path, evalpts_array)

    print(f"Surface evalpts saved as:\n- {npy_path}\n- {txt_path}")

    

def save_ordered_landmarks(cusp_landmarks, center, base_folder):
    """
    Saves landmarks of each leaflet (NCC, LCC, RCC) separately, 
    each containing its 3 landmarks (2 commissures and 1 hinge) plus the center point,
    into separate text files under the specified base folder.
    
    The leaflets are assigned based on:
    - RCC: Least posterior (smallest P)
    - LCC: Most left (largest L)
    - NCC: The remaining one
    
    Args:
        cusp_landmarks (list): List of 3 cusp landmarks, each with 3 points (commissure_1, commissure_2, hinge)
        center (list): The center point (3D)
        base_folder (str): Folder path where the landmarks are saved
    """
    
    # Extract hinge positions for each cusp (x, y, z)
    hinges = [cusp[2] for cusp in cusp_landmarks]
    
    # Get the Left (L) and Posterior (P) values for each hinge
    L_values = [h[0] for h in hinges]  # Left-right position
    P_values = [h[1] for h in hinges]  # Anterior-posterior position

    # Initialize a list to store which cusp is already assigned
    used_points = []

    # 1. Assign RCC (most anterior, i.e., least posterior)
    rcc_idx = P_values.index(min(P_values))  # Index of the smallest P (most anterior)
    used_points.append(rcc_idx)  # Mark RCC as used

    # 2. Assign LCC (most left, i.e., largest L)
    remaining_points = [i for i in range(3) if i not in used_points]  # Get remaining points
    lcc_idx = max(remaining_points, key=lambda i: L_values[i])  # Index of the largest L
    used_points.append(lcc_idx)  # Mark LCC as used

    # 3. The remaining point is NCC
    ncc_idx = [i for i in range(3) if i not in used_points][0]  # The remaining index for NCC

    # Map indices to leaflet names
    leaflet_indices = {
        "rcc": rcc_idx,
        "lcc": lcc_idx,
        "ncc": ncc_idx
    }

    # Ensure the output folder exists
    os.makedirs(base_folder, exist_ok=True)

    # Process and save landmarks for each leaflet
    for leaflet_name, idx in leaflet_indices.items():
        commissure_1, commissure_2, hinge = cusp_landmarks[idx]

        # Order commissures based on the leaflet type
        if leaflet_name == "rcc":
            ordered_commissures = sorted([commissure_1, commissure_2], key=lambda x: x[0])  # Lowest L first
        elif leaflet_name == "lcc":
            ordered_commissures = sorted([commissure_1, commissure_2], key=lambda x: x[1])  # Lowest P first
        elif leaflet_name == "ncc":
            ordered_commissures = sorted([commissure_1, commissure_2], key=lambda x: x[1], reverse=True)  # Highest P first

        # Save the ordered landmarks
        leaflet_landmarks = ordered_commissures + [hinge, center]
        output_path = os.path.join(base_folder, f"{leaflet_name}_template_landmarks.txt")
        np.savetxt(output_path, np.array(leaflet_landmarks), fmt="%.6f")

        print(f"Saved {leaflet_name} landmarks to: {output_path}")

        


def load_leaflet_landmarks(file_path):
    """
    Loads 4 landmarks from a leaflet landmark .txt file.
    
    The file is expected to contain 4 rows, each with 3 float values:
        - commissure 1
        - commissure 2
        - hinge
        - center

    Args:
        file_path (str): Path to the leaflet .txt file.

    Returns:
        np.ndarray: A (4, 3) array of landmark coordinates.
    """
    try:
        landmarks = np.loadtxt(file_path)
        if landmarks.shape != (4, 3):
            raise ValueError(f"Unexpected shape {landmarks.shape} in {file_path}; expected (4, 3).")
        return landmarks
    except Exception as e:
        print(f"Error reading landmarks from {file_path}: {e}")
        return None


def get_annular_normal(patient_number, base_path="H:/DATA/Afstuderen/3.Data/SSM/patient_database"):
    """
    Computes the perpendicular vector to the aortic annulus plane
    defined by the hinge points of the three aortic cusps.
    
    Parameters:
    - patient_number: string or int, e.g., "aos14"
    - base_path: base folder containing patient landmark subfolders
    
    Returns:
    - normal_vector: 3-element numpy array, unit vector perpendicular to the plane
    """
    
    # Construct paths to landmark files
    patient_folder = os.path.join(base_path, str(patient_number), "landmarks")
    lcc_file = os.path.join(patient_folder, "lcc_template_landmarks.txt")
    ncc_file = os.path.join(patient_folder, "ncc_template_landmarks.txt")
    rcc_file = os.path.join(patient_folder, "rcc_template_landmarks.txt")
    
    # Load hinge points (third row in each file)
    lcc_hinge = np.loadtxt(lcc_file)[2, :]
    ncc_hinge = np.loadtxt(ncc_file)[2, :]
    rcc_hinge = np.loadtxt(rcc_file)[2, :]
    
    # Define two vectors on the plane
    v1 = ncc_hinge - lcc_hinge
    v2 = rcc_hinge - lcc_hinge
    
    # Compute perpendicular vector via cross product
    normal_vector = np.cross(v1, v2)
    
    print("lcc_hinge is: ", lcc_hinge)
    print("ncc_hinge is: ", ncc_hinge)
    print("rcc_hinge is: ", rcc_hinge)
    
    print("normal_vector is: ", normal_vector)
    
    # Normalize
    normal_vector /= np.linalg.norm(normal_vector)
    
    return normal_vector


def space_volume(volume: np.ndarray, spacing: tuple):
    """
    Rescale a 3D volume so that voxel spacing becomes isotropic (1 mm per voxel).

    Parameters
    ----------
    volume : np.ndarray
        3D volume (z, y, x)
    spacing : tuple of floats
        Pixel spacing in mm: (spacing_y, spacing_x, slice_thickness)

    Returns
    -------
    rescaled_volume : np.ndarray
        Volume rescaled to isotropic voxel spacing
    """
    # Ensure spacing is in the same order as volume axes (z, y, x)
    z_spacing, y_spacing, x_spacing = spacing

    # Compute zoom factors: target spacing = 1 mm
    zoom_factors = (z_spacing, y_spacing, x_spacing)

    # Apply zoom (rescale)
    rescaled_volume = zoom(volume, zoom=zoom_factors, order=1)  # linear interpolation

    return rescaled_volume


def reorient_volume(volume, annular_normal, dicom_origin, spacing):
    """
    Reorient a 3D volume so that the annular_normal aligns with the volume z-axis,
    rotating around the DICOM origin instead of the volume center.

    Parameters
    ----------
    volume : np.ndarray
        3D numpy array in (z, y, x) order.
    annular_normal : array-like
        Unit vector normal to the annular plane, in physical coordinates (x, y, z).
    dicom_origin : array-like
        Origin of the DICOM volume in physical coordinates (x, y, z).
    spacing : array-like
        Voxel spacing (dz, dy, dx).

    Returns
    -------
    reoriented_volume : np.ndarray
        Rotated 3D volume.
    R : np.ndarray
        3x3 rotation matrix applied to the volume.
    """

    # Convert normal from physical (x,y,z) → array axes (z,y,x)
    normal = np.array([annular_normal[2], annular_normal[1], annular_normal[0]])
    normal /= np.linalg.norm(normal)
    normal = -normal
    print("Normal is: ", normal)

    # Target z-axis in array coordinates (first axis)
    target_z = np.array([1, 0, 0])  # +Z in (z, y, x) corresponds to first axis

    # Compute rotation axis and angle
    axis = np.cross(normal, target_z)
    axis_norm = np.linalg.norm(axis)
    
    print("Rotation axis is:", axis)
    
    if axis_norm < 1e-8:  # already aligned
        return volume.copy(), np.eye(3)
    axis /= axis_norm
    angle = np.arccos(np.clip(np.dot(normal, target_z), -1.0, 1.0))
    
    print("Angle is: ", angle)

    # Rodrigues rotation matrix
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    # print("Rotation matrix: ", R)
    M = np.linalg.inv(R)

    # Offset so rotation is around this point
    center = np.array(volume.shape) / 2
    offset = center - M @ center
    
    reoriented_volume = affine_transform(volume, M, offset=offset, order=1)

    # Rotation matrix is returned so that it can be used for the inverse rotation
    return reoriented_volume, R, center

def reorient_volume_back(volume, dicom_origin, rotation_matrix):
    """
    Reorient a 3D volume back to its original orientation by applying the inverse of the
    rotation matrix used for the original reorientation.

    Parameters
    ----------
    volume : np.ndarray
        3D numpy array in (z, y, x) order.
    dicom_origin : array-like
        Origin of the DICOM volume in physical coordinates (x, y, z).
    rotation_matrix : np.ndarray
        The 3x3 rotation matrix used during the original reorientation.

    Returns
    -------
    reoriented_back_volume : np.ndarray
        The reoriented volume after reversing the original rotation.
    """

    # Apply the inverse of the rotation matrix
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)

    # Offset so rotation is around the DICOM origin (reverse the original transformation)
    center = np.array(volume.shape) / 2
    offset = center - rotation_matrix @ center

    reoriented_back_volume = affine_transform(volume, rotation_matrix, offset=offset, order=1)

    return reoriented_back_volume

def vtk_to_volume_space(surface, origin, spacing):
    """
    Transform VTK PolyData points from DICOM patient coordinates to NumPy volume voxel space.

    Parameters
    ----------
    surface : vtk.vtkPolyData
        VTK surface in DICOM patient coordinates.
    origin : array-like of shape (3,)
        DICOM ImagePositionPatient of the first slice.
    spacing : array-like of shape (3,)
        Voxel spacing (dz, dy, dx) of the volume.

    Returns
    -------
    vtk.vtkPolyData
        The modified surface in volume voxel space.
    """
    points = surface.GetPoints()
    n_points = points.GetNumberOfPoints()
    
    for i in range(n_points):
        x, y, z = points.GetPoint(i)
        vx = (z - origin[0])/spacing[0]
        vy = (y - origin[1])/spacing[1]
        vz = (x - origin[2])/spacing[2] 
        points.SetPoint(i, vx, vy, vz)
    
    surface.Modified()
    return surface

def landmarks_to_voxel(txt_file, origin, pixel_spacing):
    """
    Converts landmark coordinates from a text file into voxel coordinates.
    
    Parameters:
        txt_file (str): Path to the text file containing landmarks (LPS/RAS coordinates).
        origin (np.ndarray or list): Origin of the volume (LPS coordinates of voxel [0,0,0]).
        spacing (np.ndarray or list): Voxel spacing in mm for (x, y, z) directions.
    
    Returns:
        np.ndarray: N x 3 array of voxel coordinates.
    """
    spacing = pixel_spacing[::-1]

    landmarks_lps = np.loadtxt(txt_file)
    print("Coordinates: ", landmarks_lps)
    voxel_coords = (landmarks_lps-origin)/spacing
    voxel_coords = np.round(voxel_coords).astype(int)
    
    return voxel_coords

def reorient_landmarks(landmarks_zyx, R, dicom_origin, spacing, output_shape):
    """
    Reorient landmarks to match affine_transform applied to the volume.
    Preserves all landmarks and matches the original backward mapping.

    Parameters
    ----------
    landmarks_zyx : (N,3) array
        Landmarks in voxel coordinates (z,y,x) of the original volume.
    R : (3,3) array
        Rotation matrix used for the volume (zyx convention).
    dicom_origin : array-like
        Not used, kept for API.
    spacing : array-like
        Not used, kept for API.
    output_shape : tuple
        Shape of the reoriented volume (z,y,x)

    Returns
    -------
    rotated_landmarks_voxel : (N,3) array
        Landmark coordinates in the reoriented volume grid (z,y,x order).
    """
    # Center of the volume
    center = np.array(output_shape) / 2

    # Affine mapping used by scipy: M = inv(R), offset = center - M @ center
    M = np.linalg.inv(R)
    offset = center - M @ center

    # Compute output coordinates corresponding to input landmarks
    rotated_landmarks = np.linalg.inv(M) @ (landmarks_zyx.T - offset[:, None])
    rotated_landmarks = rotated_landmarks.T

    # Round to integer voxel coordinates
    rotated_landmarks_voxel = np.round(rotated_landmarks).astype(int)

    # Debug info
    print(f"Number of landmarks before affine transform: {landmarks_zyx.shape[0]}")
    print(f"Number of landmarks after affine transform: {rotated_landmarks_voxel.shape[0]}")

    return rotated_landmarks_voxel


def rotate_vtk_landmarks(vtk_points, rotation_matrix, rotation_offset):
    """
    Rotate VTK landmarks (or any 3D points) using a known rotation matrix
    and the same rotation pivot as the volume.

    Parameters
    ----------
    vtk_points : (N,3) array
        VTK points to rotate (z, y, x order or voxel coordinates).
    rotation_matrix : (3,3) array
        Rotation matrix applied to the volume.
    rotation_offset : (3,) array-like
        The pivot (center) used when rotating the volume in voxel coordinates
        (the same as used to compute the volume offset in affine_transform).

    Returns
    -------
    rotated_points : (N,3) array
        Rotated VTK points, same shape as input.
    """
    vtk_points = np.array(vtk_points, dtype=float)
    rotation_offset = np.array(rotation_offset, dtype=float)

    # Rotate around the center (rotation_offset acts as the pivot)
    rotated_points = (rotation_matrix @ (vtk_points - rotation_offset).T).T + rotation_offset

    return rotated_points

def order_points_by_angle(points):
    """
    Order a set of 2D points to form a closed loop by sorting them
    according to their angle relative to the centroid.
    
    Parameters
    ----------
    points : ndarray of shape (N, 2)
        Points in (row, col) = (y, x) format.

    Returns
    -------
    ordered_points : ndarray of shape (N, 2)
        Points ordered to form a smooth closed loop.
    """
    # Compute centroid
    centroid = points.mean(axis=0)
    
    # Compute angles relative to centroid
    angles = np.arctan2(points[:,0] - centroid[0], points[:,1] - centroid[1])
    
    # Sort points by angle
    sort_idx = np.argsort(angles)
    ordered_points = points[sort_idx]
    
    return ordered_points


def order_points_nn(points):
    """
    Order points along a 2D open contour using a greedy nearest-neighbor approach.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        Points in (y, x) format.

    Returns
    -------
    ordered_points : ndarray of shape (N, 2)
        Points ordered along the contour.
    """
    if len(points) == 0:
        return points

    points = points.copy()
    # Start from the first point (or pick any)
    ordered = [points[0]]
    points = np.delete(points, 0, axis=0)

    while len(points) > 0:
        last = ordered[-1]
        dists = cdist([last], points)[0]
        idx = np.argmin(dists)
        ordered.append(points[idx])
        points = np.delete(points, idx, axis=0)

    return np.array(ordered)


def find_closest_point(points, reference_point):
    """
    Finds the point in `points` closest to `reference_point`.

    Parameters
    ----------
    points : ndarray of shape (N, D)
        Array of points (2D or 3D).
    reference_point : array-like of shape (D,)
        The reference point to compare distances to.

    Returns
    -------
    closest_idx : int
        Index of the point in `points` closest to `reference_point`.
    closest_point : ndarray
        Coordinates of the closest point.
    """
    points = np.array(points)
    reference_point = np.array(reference_point)
    distances = np.linalg.norm(points - reference_point, axis=1)
    closest_idx = np.argmin(distances)
    closest_point = points[closest_idx]
    return closest_idx, closest_point


def circle_through_commissures(points, n_points=25):
    """
    Fit a circle through 3 commissure landmarks in 3D and return points on the circle.

    Parameters
    ----------
    points : ndarray of shape (3, 3)
        Commissure coordinates (x, y, z).
    n_points : int
        Number of points to sample along the circle.

    Returns
    -------
    circle_points : ndarray of shape (n_points, 3)
        Points lying on the fitted circle in 3D.
    center : ndarray of shape (3,)
        Circle center in 3D.
    radius : float
        Circle radius.
    normal : ndarray of shape (3,)
        Normal vector of the plane containing the circle.
    """
    if points.shape != (3, 3):
        raise ValueError("Input must be a (3,3) array of commissure points in 3D")
    points = points.astype(float)
    A, B, C = points

    # Vectors
    AB = B - A
    AC = C - A

    # Normal vector of the circle's plane
    normal = np.cross(AB, AC)
    print('normal is ----', normal)
    normal /= np.linalg.norm(normal)

    # Midpoints
    mid_AB = (A + B) / 2
    mid_AC = (A + C) / 2

    # Directions of perpendicular bisectors (in plane)
    dir_AB = np.cross(normal, AB)
    dir_AC = np.cross(normal, AC)

    # Solve for intersection of bisectors -> circle center
    M = np.column_stack((dir_AB, -dir_AC))
    rhs = mid_AC - mid_AB
    t, s = np.linalg.lstsq(M, rhs, rcond=None)[0]
    center = mid_AB + t * dir_AB

    # Circle radius
    radius = np.linalg.norm(center - A)

    # Build orthonormal basis in the circle plane
    u = AB / np.linalg.norm(AB)              # first in-plane axis
    v = np.cross(normal, u)                  # second in-plane axis (orthogonal)

    # Parametric circle in plane
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    circle_points = np.array([center + radius * (np.cos(theta) * u + np.sin(theta) * v)
                              for theta in angles])
    circle_points=circle_points.astype(int)
    return circle_points


def midpoint_xy(points):
    """
    Calculate the midpoint in XY plane from three 3D points.
    
    Parameters:
        points (list or array): List of three points [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
        
    Returns:
        tuple: (mid_x, mid_y)
    """
    points = np.array(points)
    
    if points.shape != (3, 3):
        raise ValueError("Input must be three 3D points (3x3 array or list).")
    
    mid_x = (np.mean(points[:, 1])).astype(int)
    mid_y = np.mean(points[:, 2]).astype(int)

    return tuple((mid_x, mid_y))

def closest_contour_point(point_yx, contour):
    """Return index of contour point closest to given (y,x) point."""
    dists = np.linalg.norm(contour - point_yx, axis=1)
    return int(np.argmin(dists))


def contour_segment(contour, i_start, i_end, debug_image=None):
    """
    Extract the shortest segment along a closed contour using geometric distance.
    Optionally visualizes the start/end points on the given image.
    """

    # --- Compute both forward and reverse segments ---
    if i_start <= i_end:
        seg_fwd = contour[i_start:i_end]
    else:
        seg_fwd = np.vstack([contour[i_start:], contour[:i_end]])
    dist_fwd = np.sum(np.linalg.norm(np.diff(seg_fwd, axis=0), axis=1))

    if i_end <= i_start:
        seg_rev = contour[i_end:i_start][::-1]
    else:
        seg_rev = np.vstack([contour[i_end:], contour[:i_start]])[::-1]
    dist_rev = np.sum(np.linalg.norm(np.diff(seg_rev, axis=0), axis=1))

    # --- Choose shorter segment ---
    seg = seg_fwd if dist_fwd <= dist_rev else seg_rev

    return seg


    
def create_boundary_mask(center, contour, hinge_idx_1, hinge_idx_2, segment, upscaled, inner_skeleton):
    """
    Create a leaflet boundary mask and masked skeleton region between two hinges.

    Parameters
    ----------
    center : np.ndarray
        (y, x) coordinates of the leaflet center.
    contour : np.ndarray
        Full aortic wall contour (Nx2).
    hinge_idx_1, hinge_idx_2 : int
        Indices of the two hinges on the contour.
    segment : np.ndarray
        Contour segment between hinge_idx_1 and hinge_idx_2.
    upscaled : np.ndarray
        The upscaled DICOM image for visualization/mask sizing.
    inner_skeleton : np.ndarray
        The skeleton image to extract leaflet boundaries from.

    Returns
    -------
    mask : np.ndarray
        Binary mask for this leaflet region.
    boundary : np.ndarray
        Masked portion of the skeleton corresponding to this boundary.
    polygon : np.ndarray
        Closed polygon coordinates defining the region.
    """

    polygon = np.vstack([
        center,
        contour[hinge_idx_1],
        segment,
        contour[hinge_idx_2],
        center
    ])

    mask = polygon2mask(upscaled.shape, polygon)
    # mask = dilation(mask, disk(3)) optional dilation if you want it to be robuster
    boundary = inner_skeleton * mask

    return mask, boundary


def resample_closed_contour(contour, n_points=100):
    """
    Resample a closed 2D contour to have n_points (arc-length parameterization).
    """
    x, y = contour[:, 1], contour[:, 0]
    # Ensure the contour is closed
    if not np.allclose(contour[0], contour[-1]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    # Fit a periodic spline
    tck, _ = splprep([x, y], s=0, per=True)
    u_new = np.linspace(0, 1, n_points)
    x_new, y_new = splev(u_new, tck)
    return np.vstack([y_new, x_new]).T  # return in (row, col) order



def find_all_boundary_intersections(
    upscaled,
    seg_lcc_ncc,
    seg_rcc_lcc,
    seg_ncc_rcc,
    lcc_ncc_boundary,
    rcc_lcc_boundary,
    ncc_rcc_boundary,
    slice_idx,
    mercedes_mask,
    plot=False,
):

    boundaries = {
        "lcc_ncc": (lcc_ncc_boundary, seg_lcc_ncc, "cyan"),
        "rcc_lcc": (rcc_lcc_boundary, seg_rcc_lcc, "magenta"),
        "ncc_rcc": (ncc_rcc_boundary, seg_ncc_rcc, "green"),
    }

    intersections = {}
    line_data = {}  # store line points for plotting

    for name, (boundary, wall_segment, color) in boundaries.items():
        if boundary is None or wall_segment is None or len(wall_segment) < 3:
            intersections[name] = None
            continue

        y_true, x_true = boundary[:, 0], boundary[:, 1]
        # --- Compute total Euclidean length along the boundary ---
        boundary_points = boundary
        if len(boundary_points) < 4:
            total_length = 0
        else:
            total_length = compute_max_pairwise_distance(boundary)  
            # print(total_length)
            print(f"{name} total length: {total_length}")

        # Skip if boundary is too short
        if total_length < 5:  # adjust threshold in pixels
            intersections[name] = None
            continue

        try:
            # Fit line y = m*x + b
            m, b = np.polyfit(x_true, y_true, 1)
            x_line = np.linspace(0, upscaled.shape[1] - 1, 500)
            y_line = m * x_line + b
            line_points = np.vstack((y_line, x_line)).T
            line_data[name] = (x_line, y_line)

            # Find closest intersection with wall
            seg_y, seg_x = wall_segment[:, 0], wall_segment[:, 1]
            seg_points = np.vstack((seg_y, seg_x)).T

            distances = np.linalg.norm(seg_points[:, None, :] - line_points[None, :, :], axis=2)
            idx_seg, idx_line = np.unravel_index(np.argmin(distances), distances.shape)
            intersection = seg_points[idx_seg]
            
            # Check if the intersection is inside the Mercedes mask
            if mercedes_mask is not None:
                if not mercedes_mask[int(intersection[0]), int(intersection[1])]:  # Check if the point is inside the mask
                    # print(f"Intersection for {name} is outside the mercedes mask. Using previous intersection.")
                    intersections[name] = None
                else:
                    intersections[name] = tuple(intersection)
            else:
                intersections[name] = tuple(intersection)

        except Exception as e:
            print(f"[Warning] Could not compute intersection for {name}: {e}")
            intersections[name] = None
            line_data[name] = None

    return intersections


def compute_max_pairwise_distance(points):
    """
    Compute distance between the two points that are furthest apart.

    This is the diameter of the point set.

    Parameters
    ----------
    points : (N,3) or (N,2) array-like

    Returns
    -------
    float
        Maximum Euclidean distance between any two points.
    """

    if points is None:
        return 0

    points = np.asarray(points)

    if len(points) < 2:
        return 0

    # Pairwise distance matrix
    dist_matrix = cdist(points, points)

    # Maximum distance (diameter of point cloud)
    max_distance = np.max(dist_matrix)

    return float(max_distance)

def compute_boundary_to_com_distance(boundary_curve, com_point):
    """
    Compute minimum Euclidean distance from boundary curve to commissure point.
    Safe against empty arrays.
    """

    if boundary_curve is None or com_point is None:
        return None

    boundary_curve = np.asarray(boundary_curve)

    if boundary_curve.size == 0:
        return None

    com_point = np.asarray(com_point)

    distances = np.linalg.norm(boundary_curve - com_point, axis=1)

    if distances.size == 0:
        return None

    return float(np.min(distances))

def clean_boundary_from_mask(mask, aortic_wall_points, min_dist, commissures, center):
    """
    Extract the largest connected component from a binary mask 
    and return its coordinates in (y, x) order.
    
    First filters according to the Mercedes mask, then removes points that are too close to the aortic wall
    based on Euclidean distance.
    
    Parameters
    ----------
    mask : 2D array
        Binary mask of the region (leaflet).
    aortic_wall_points : Nx2 array of (y, x), optional
        Coordinates of aortic wall points.
    min_dist : float
        Minimum Euclidean distance (in pixels) from the aortic wall.
     center : tuple
        Coordinates of the center point (x, y).
    
    Returns
    -------
    coords : Nx2 array of (y, x)
        Filtered boundary points.
    """
    # Create the Mercedes mask
    mercedes_mask = create_mercedes_mask(mask, commissures, center)
    
    # Print some info
    # print(f"[Debug]  Mercedes mask shape = {mercedes_mask.shape}")
    # print(f"[Debug]  Mercedes mask min = {mercedes_mask.min()}, max = {mercedes_mask.max()}")
    # print(f"[Debug] Non-zero pixels = {np.count_nonzero(mercedes_mask)}")
    # Apply the Mercedes mask to the input mask
    filtered_mask = (mask > 0) & (mercedes_mask > 0)
    
    # Label the connected components
    labeled = label(filtered_mask)
    if labeled.max() == 0:
        # print("[Info] Mask is empty after applying the Mercedes mask, returning empty array.")
        return np.zeros((0, 2), dtype=int), mask

    # Largest connected component
    regions = regionprops(labeled)
    largest_region = max(regions, key=lambda r: r.area)
    coords = largest_region.coords
    # Final hard Mercedes filtering (row, col indexing!)
    coords = coords[mercedes_mask[coords[:, 0], coords[:, 1]] > 0]

    # print(f"[Info] Largest component has {len(coords)} points before filtering.")

    # # Filter based on distance to aortic wall points
    if aortic_wall_points is not None and len(aortic_wall_points) > 0:
        tree = cKDTree(aortic_wall_points)
        dists, _ = tree.query(coords)
        mask_indices = dists < min_dist
        removed_count = np.sum(mask_indices)
        coords = coords[~mask_indices]
        # print(f"[Info] Removed {removed_count} points too close to aortic wall.")
        # print(f"[Info] Remaining points after filtering: {len(coords)}")

    return coords, mercedes_mask



def compute_edge_volume(raw_volume_hu,
                        hu_window=(0, 400),
                        sigma=2,
                        post_sigma=0.5,
                        clahe_clip=0.03,
                        normalize=True,
                        visualize=False,
                        slice_idx=None):
    """
    Compute per-slice edge magnitude and optionally visualize intermediate steps.
    """

    # 0. Slice selection for visualization
    if slice_idx is None:
        slice_idx = raw_volume_hu.shape[0] // 2  # middle slice

    # Clip to HU window
    clipped = np.clip(raw_volume_hu, hu_window[0], hu_window[1])

    smoothed_vol = np.zeros_like(clipped, dtype=np.float32)
    norm_vol = np.zeros_like(clipped, dtype=np.float32)
    gradient_vol = np.zeros_like(clipped, dtype=np.float32)

    # Process each slice
    for i in range(raw_volume_hu.shape[0]):

        # 1. Gaussian pre-smoothing
        smoothed = gaussian_filter(clipped[i], sigma=sigma)
        smoothed_vol[i] = smoothed

        # 2. Normalize to [0,1]
        norm = exposure.rescale_intensity(smoothed, in_range="image", out_range=(0, 1))
        norm_vol[i] = norm
        
        # 3. Sobel edge detection
        grad = sobel(norm)
        gradient_vol[i] = grad

    # 5. Post-smoothing
    if post_sigma > 0:
        gradient_vol = gaussian_filter(gradient_vol, sigma=post_sigma)

    # 6. Optional normalization to 0–255
    if normalize:
        gradient_vol = exposure.rescale_intensity(gradient_vol, out_range=(0, 255))

    return gradient_vol


def compute_slice_spacing(dicom_list):
    """
    dicom_list: list of tuples (filepath, z_position)
    """

    positions = []

    for item in dicom_list:
        # item = (filepath, z_pos)
        filepath = item[0]
        ds = pydicom.dcmread(filepath)

        if hasattr(ds, "ImagePositionPatient"):
            positions.append(float(ds.ImagePositionPatient[2]))
        elif hasattr(ds, "SliceLocation"):
            positions.append(float(ds.SliceLocation))
        else:
            raise ValueError("DICOM slice missing both ImagePositionPatient and SliceLocation.")

    # Sort by position
    positions = np.array(sorted(positions))
    diffs = np.diff(positions)

    if len(diffs) == 0:
        raise ValueError("Cannot compute slice spacing from a single slice.")

    return round(float(abs(np.median(diffs))), 3)


def rotate_segmentation_back(seg_rotated, R, rotation_center, upscaling_factor):
    """
    Rotate a segmentation from rotated/annular-aligned space
    back to original DICOM space.

    Parameters
    ----------
    seg_rotated : np.ndarray
        3D segmentation in rotated space (binary or labels)
    R : np.ndarray
        3x3 rotation matrix used to align original volume

    Returns
    -------
    seg_original : np.ndarray
        Segmentation in original DICOM space
    """
    # Center of rotated volume - since we are working with upsampled image, center also needs to be upsampled
    center = rotation_center * upscaling_factor

    # Rotation matrix: rotated -> original
    M_back = R

    # Offset for rotation around center
    offset_back = center - M_back @ center

    # Apply affine transform
    seg_original = affine_transform(
        seg_rotated,
        M_back,
        offset=offset_back,
        order=0       # nearest neighbor for masks
    )

    return seg_original

def create_mercedes_mask(image, commissure_points, center_point, line_thickness=8):
    """
    Creates a mask of lines connecting the commissures to the center with specified thickness.
    
    Parameters:
    - image: The image in which to draw the lines (used for shape).
    - commissure_points: A list of tuples where each tuple contains the coordinates (x, y) of a commissure.
    - center_point: A tuple (x, y) for the center point.
    - line_thickness: Thickness of the lines to be drawn.
    
    Returns:
    - A mask (binary image) with the thickened lines.
    """
    # Initialize the mask with zeros (black)
    mask = np.zeros(image.shape, dtype=np.uint8)

    # Loop through each commissure and draw a thick line
    for commissure_coords in commissure_points:
        # print(commissure_coords)
        x1, y1 = commissure_coords  # Swap x and y
        x2, y2 = center_point  # Swap x and y
        
        # Use skimage's line function to get the line coordinates
        rr, cc = line(int(x1), int(y1), int(x2), int(y2))
        
        # Loop through the line points and thicken the line
        for i in range(-line_thickness//2, line_thickness//2 + 1):
            for j in range(-line_thickness//2, line_thickness//2 + 1):
                rr_offset = rr + i
                cc_offset = cc + j
                # Ensure the points are within image bounds (for both rr and cc)
                valid_indices = (rr_offset >= 0) & (rr_offset < mask.shape[0]) & (cc_offset >= 0) & (cc_offset < mask.shape[1])
                mask[rr_offset[valid_indices], cc_offset[valid_indices]] = 1

    return mask



def save_volume_as_stl(calc_volume, output_path, patient_nr, file_type):
    """Save the calcification volume as an STL file."""
    # Use skimage to find contours (isosurface) from the calcification mask
    verts, faces, _, _ = measure.marching_cubes(calc_volume, level=0.5)
    
    # Create a VTK polydata object
    polydata = vtk.vtkPolyData()

    # Create VTK points and add them to the polydata
    points = vtk.vtkPoints()
    for vert in verts:
        points.InsertNextPoint(vert)
    polydata.SetPoints(points)

    # Create VTK cells (faces) and add them to the polydata
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(3, face.astype(int))  # faces need to be integer type
    polydata.SetPolys(cells)

    # Write the polydata to an STL file
    stl_writer = vtk.vtkSTLWriter()
    output_filename = os.path.join(output_path, f"{patient_nr}_{file_type}.stl")
    stl_writer.SetFileName(output_filename)
    stl_writer.SetInputData(polydata)
    stl_writer.Write()

    print(f"{file_type} volume saved as STL: {output_filename}")

def create_3d_mask_from_boundary_points(boundary_data, calc_volume_shape, which_boundary):
    """Create a 3D binary mask from the boundary points."""
    # Initialize a 3D mask with the same shape as calc_volume, all set to False
    mask_3d = np.zeros(calc_volume_shape, dtype=bool)

    # Iterate over each slice in the boundary data
    for slice_nr, data in boundary_data.items():
        boundary_points = data[which_boundary]  # Boundary points (Nx2 array)

        # Convert boundary points to slice coordinates (row, col)
        for point in boundary_points:
            y, x = point.astype(int)  # boundary points are (y, x), mapping to (row, col)
            
            # Make sure the coordinates are within bounds
            if 0 <= y < calc_volume_shape[1] and 0 <= x < calc_volume_shape[2]:
                mask_3d[slice_nr, y, x] = True  # Set the corresponding voxel to True

    return mask_3d

def downsample_and_rescale(volume, downsample_factor, inverse_zoom):
    """
    Downsamples a volume by a factor and then rescales it using inverse_zoom.
    
    Parameters
    ----------
    volume : np.ndarray
        Input 3D volume.
    downsample_factor : float or tuple of 3 floats
        Factor to downsample the volume. Can be a single float or per-axis.
    inverse_zoom : tuple of 3 floats
        Zoom factors to restore voxel spacing after processing.
    is_mask : bool, default False
        If True, use nearest-neighbor interpolation (order=0), otherwise linear interpolation (order=1).
    
    Returns
    -------
    np.ndarray
        Processed volume.
    """
    # Step 1: Downsample
    volume_down = zoom(volume, downsample_factor, order=0)
    
    # Step 2: Rescale using inverse_zoom
    volume_rescaled = zoom(volume, inverse_zoom, order=0)
    
    return volume_rescaled

def save_volume_as_stl_patient_space(volume, output_path, patient_nr, file_type,
                                     zoom_x, zoom_y, zoom_z,
                                     dicom_origin):
    """
    Save a 3D binary calcification volume as an STL file in patient (DICOM) space.
    
    Parameters
    ----------
    calc_volume : np.ndarray
        3D binary mask of calcification (shape: z, y, x)
    output_path : str
        Folder where the STL will be saved
    patient_nr : str or int
        Patient identifier for filename
    file_type : str
        Type of volume, e.g., 'calcification'
    pixel_spacing_x : float
        Voxel size in x direction (mm)
    pixel_spacing_y : float
        Voxel size in y direction (mm)
    slice_thickness : float
        Voxel size in z direction (mm)
    dicom_origin : np.ndarray
        Origin of the volume in patient coordinates (x0, y0, z0)
    """
    
    verts, faces, normals, values = measure.marching_cubes(
        volume,  # convert to smaller type
        level=0.5,
        spacing=(zoom_z, zoom_y, zoom_x)
    )
    
    # Convert verts from (z, y, x) -> (x, y, z)
    verts = verts[:, ::-1]
    
    # Shift to patient (DICOM) space using the origin
    verts += dicom_origin
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Save STL
    output_filename = os.path.join(output_path, f"{patient_nr}_{file_type}.stl")
    mesh.export(output_filename)
    
    print(f"[OK] {file_type} volume saved as STL in patient space: {output_filename}")

def bresenham_line_2d(p0, p1):
    """
    Bresenham line in 2D between points p0=(y0,x0) and p1=(y1,x1)
    Returns list of (y,x) coordinates along the line
    """
    y0, x0 = p0
    y1, x1 = p1
    points = []

    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy

    while True:
        points.append((y0, x0))
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def grow_boundary_until_wall(
    boundary_mask,
    wall_mask,
    z_center,
    line_dilate=2,
    keep_below_center=True
):
    """
    Slice-wise boundary growth toward wall.

    - Growth only happens for z <= z_center
    - Boundary preservation below z_center is optional
    """
    assert boundary_mask.shape == wall_mask.shape
    Z, Y, X = boundary_mask.shape

    expanded_full = np.zeros_like(boundary_mask, dtype=bool)
    selem = np.ones((3, 3), dtype=bool)

    for z in range(Z):

        #No growth below center plane
        if z > z_center:
            continue

        wall_slice = wall_mask[z]
        boundary_slice = boundary_mask[z]

        if not np.any(boundary_slice) or not np.any(wall_slice):
            continue

        boundary_coords = np.array(np.where(boundary_slice)).T
        wall_coords = np.array(np.where(wall_slice)).T

        tree = cKDTree(wall_coords)
        distances, indices = tree.query(boundary_coords)

        min_idx = np.argmin(distances)
        closest_boundary = boundary_coords[min_idx]
        closest_wall = wall_coords[indices[min_idx]]

        line_voxels = bresenham_line_2d(
            tuple(closest_boundary),
            tuple(closest_wall)
        )

        for y, x in line_voxels:
            expanded_full[z, y, x] = True

        if line_dilate > 0:
            expanded_full[z] = binary_dilation(
                expanded_full[z],
                structure=selem,
                iterations=line_dilate
            )

    # ---------------- Boundary preservation ----------------

    if keep_below_center:
        # Keep boundary everywhere above AND below center
        expanded_full |= boundary_mask
    else:
        # Keep boundary only above center plane
        expanded_full[:z_center + 1] |= boundary_mask[:z_center + 1]

    return expanded_full



def grow_boundary(boundary_mask, wall_mask, center_height,
                  line_dilate=2, gaussian_blur=0, keep_below_center = True):
    """
    Grow boundary toward wall only for slices z >= center_height.

    Returns:
        expanded_full : 3D bool array
    """
    combined_union = boundary_mask | wall_mask
    coords = np.array(np.where(combined_union))
    z_min, y_min, x_min = coords.min(axis=1)
    z_max, y_max, x_max = coords.max(axis=1)

    boundary_crop = boundary_mask[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    wall_crop = wall_mask[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

    center_height_crop = center_height - z_min
    center_height_crop = np.clip(
        center_height_crop, 0, boundary_crop.shape[0] - 1
    )

    expanded_crop = grow_boundary_until_wall(
        boundary_crop,
        wall_crop,
        z_center=center_height_crop,
        line_dilate=line_dilate,
        keep_below_center=keep_below_center
    )

    expanded_full = np.zeros_like(boundary_mask, dtype=bool)
    expanded_full[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = expanded_crop

    if gaussian_blur > 0:
        expanded_full = gaussian_filter(
            expanded_full.astype(np.float32),
            sigma=gaussian_blur
        )


    # Also export the lowest slice, this is needed to close the aortic leaflet
    return expanded_full, expanded_full[center_height] 

def load_stl_points(stl_path):
    """
    Load an STL file using VTK and return its surface points as a NumPy array.
    """
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()

    polydata = reader.GetOutput()
    vtk_points = polydata.GetPoints()

    points = np.array([
        vtk_points.GetPoint(i)
        for i in range(vtk_points.GetNumberOfPoints())
    ])

    return points

def extract_lower_boundary(points, band_thickness=0.4):
    """
    Extract a thin band of points near the lowest Z-value.
    This represents the base of the leaflet.
    """
    z_vals = points[:, 2]
    z_min = z_vals.min()

    boundary_points = points[z_vals < z_min + band_thickness]

    return boundary_points

def mask_to_pointcloud(mask, height):
    """
    Convert a 2D binary mask (512x512) into a 3D point cloud
    with fixed height (z-coordinate).
    
    Parameters
    ----------
    mask : (H, W) ndarray
        Binary mask (0/1)
    height : int or float
        Slice height (z value)

    Returns
    -------
    points : (N, 3) ndarray
        3D coordinates [z, y, x]
    """
    # Get indices where mask is 1
    y_idx, x_idx = np.where(mask > 0)

    # Create z coordinate
    z_idx = np.full_like(y_idx, height)

    # Stack into (N,3)
    points = np.column_stack((z_idx, y_idx, x_idx))

    return points

def fit_spline(points, n_samples=20, smoothing=2):
    """
    Fit a smooth 3D spline through points and return sampled points.
    
    Parameters
    ----------
    points : (N,3) array-like
        3D points to fit the spline through. Should be ordered along the curve.
    n_samples : int
        Number of points to sample along the spline.
    smoothing : float
        Smoothing factor. Larger = smoother, smaller = closer to original points.
        
    Returns
    -------
    sampled_points : (n_samples,3) ndarray
        Points sampled along the fitted spline.
    """
    points = np.asarray(points)
    if points.shape[1] != 3:
        raise ValueError("Input points must have shape (N,3)")
    
    # Fit spline
    tck, u = splprep(points.T, s=smoothing,k=2)
    
    # Sample along spline
    u_new = np.linspace(0, 1, n_samples)
    sampled_points = np.array(splev(u_new, tck)).T
    
    return sampled_points


def create_3d_mask_from_points(points, volume_shape, thickness_voxels=1):
    """
    Create a 3D binary mask from a point cloud and optionally give it some thickness.

    Parameters
    ----------
    points : np.ndarray, shape (N,3)
        Points in (z, y, x) voxel coordinates
    volume_shape : tuple of int
        Shape of the target 3D volume (z, y, x)
    thickness_voxels : int
        Number of voxels to dilate the points for thickness
    """
    mask = np.zeros(volume_shape, dtype=bool)

    # Round to nearest voxel
    indices = np.round(points).astype(int)

    # Clip to volume boundaries
    indices[:,0] = np.clip(indices[:,0], 0, volume_shape[0]-1)
    indices[:,1] = np.clip(indices[:,1], 0, volume_shape[1]-1)
    indices[:,2] = np.clip(indices[:,2], 0, volume_shape[2]-1)

    # Fill mask
    mask[indices[:,0], indices[:,1], indices[:,2]] = True

    # Add thickness
    if thickness_voxels > 0:
        mask = binary_dilation(mask, iterations=thickness_voxels)

    return mask


def get_leaflet_attachment_curve(
        aortic_wall_mask,
        commissure1,
        hinge,
        commissure2,
        n_points=20,
        snap_to_wall=True):
    """
    Computes smooth attachment curve between commissure1 → hinge → commissure2.
    """

    # Extract wall surface voxels
    wall_surface = aortic_wall_mask & ~binary_erosion(aortic_wall_mask)
    wall_coords = np.array(np.where(wall_surface)).T

    tree = cKDTree(wall_coords)

    # Snap landmarks to wall
    snapped = []
    for lm in [commissure1, hinge, commissure2]:
        _, idx = tree.query(lm)
        snapped.append(wall_coords[idx])

    comm1, hinge, comm2 = snapped

    # Fit spline through 3 landmarks
    landmarks_array = np.array([comm1, hinge, comm2])
    coords = landmarks_array.T

    tck, _ = splprep(coords, s=0, k=2)
    u_new = np.linspace(0, 1, n_points)
    smooth_curve = np.array(splev(u_new, tck)).T

    if snap_to_wall:
        _, idx = tree.query(smooth_curve)
        curve_on_wall = wall_coords[idx]
    else:
        curve_on_wall = smooth_curve.copy()

    return smooth_curve, curve_on_wall

def create_leaflet_surface_from_curves(curves,
                                       degree_u=2,
                                       degree_v=3,
                                       delta=0.01):
    """
    Creates interpolated NURBS surface from 3 boundary curves:
    [free_edge_1, wall_curve, free_edge_2]

    The function automatically reorders the free edge curves such that
    their LAST point is closest to the wall curve.

    Parameters
    ----------
    curves : list of (N,3) arrays
        [boundary_curve_1, boundary_curve_2, boundary_curve_3]
        Each curve must have the same number of points.
    degree_u : int
    degree_v : int
    delta : float
        Surface evaluation density.

    Returns
    -------
    surf : geomdl surface object
    eval_pts : (M,3) evaluated surface points
    """

    # --- Convert safely to numpy arrays ---
    curves = [np.array(c) for c in curves]

    # --- Basic safety check ---
    num_pts = curves[0].shape[0]
    for c in curves:
        if c.shape[0] != num_pts:
            raise ValueError("All curves must have same number of points")

    # -------------------------------------------------
    # Reorder logic:
    # Make sure free edge curves end at wall curve
    # -------------------------------------------------

    # Assume middle curve is the wall curve
    free_1, wall_curve, free_2 = curves

    wall_start = wall_curve[0]
    wall_end = wall_curve[-1]

    def reorder_curve(curve):
        """
        Ensures last point of curve is closest to wall.
        If first point is closer to wall than last point,
        reverse the curve.
        """

        # Distance of first point to wall
        d_first = min(
            np.linalg.norm(curve[0] - wall_start),
            np.linalg.norm(curve[0] - wall_end)
        )

        # Distance of last point to wall
        d_last = min(
            np.linalg.norm(curve[-1] - wall_start),
            np.linalg.norm(curve[-1] - wall_end)
        )

        # If first point is closer → reverse curve
        if d_first < d_last:
            return curve[::-1]

        return curve

    free_1 = reorder_curve(free_1)
    free_2 = reorder_curve(free_2)

    curves = np.array([free_1, wall_curve, free_2])

    # -------------------------------------------------
    # Surface fitting
    # -------------------------------------------------

    num_curves = curves.shape[0]

    surf_flat = curves.reshape(-1, 3)

    surf = fitting.interpolate_surface(
        surf_flat,
        size_u=num_curves,
        size_v=num_pts,
        degree_u=degree_u,
        degree_v=degree_v
    )

    # --- Evaluate surface ---
    surf.delta = delta
    surf.evaluate()

    eval_pts = np.array(surf.evalpts)

    return surf, eval_pts




def surface_points_to_stl(
        eval_pts,
        volume_shape,
        pixel_spacing,
        dicom_origin,
        output_path,
        patient_nr,
        file_type,
        thickness_voxels=1,
        smooth_sigma=1.5):

    closing_structure = cube(3)

    mask_3d = create_3d_mask_from_points(
        eval_pts,
        volume_shape,
        thickness_voxels=thickness_voxels
    )

    mask_3d = binary_closing(mask_3d, structure=closing_structure)
    # mask_3d = binary_dilation(mask_3d, closing_structure)
    mask_3d = gaussian_filter(mask_3d.astype(np.float32),
                              sigma=smooth_sigma)

    save_volume_as_stl_patient_space(
        volume=mask_3d,
        output_path=output_path,
        patient_nr=patient_nr,
        file_type=file_type,
        zoom_x=pixel_spacing[2],
        zoom_y=pixel_spacing[1],
        zoom_z=pixel_spacing[0],
        dicom_origin=dicom_origin
    )

    return mask_3d



def debug_plot_curves_2d(curves,
                         labels=None,
                         title="2D Projection of Curves (XY plane)",
                         show_order=True):
    """
    Debug visualization of boundary curves in XY plane.

    Parameters
    ----------
    curves : list of (N,3) arrays
        Curves to visualize.
    labels : list of str, optional
        Labels for legend.
    title : str
        Plot title.
    show_order : bool
        If True, shows start (circle) and end (cross) markers.
    """

    if labels is None:
        labels = [f"curve_{i}" for i in range(len(curves))]

    colors = ["blue", "red", "green", "orange", "purple"]

    plt.figure(figsize=(6, 6))

    for i, curve in enumerate(curves):
        curve = np.array(curve)

        x = curve[:, 2]  # X
        y = curve[:, 1]  # Y

        color = colors[i % len(colors)]

        # ✅ Plot points instead of line
        plt.scatter(x, y, color=color, s=20, label=labels[i])

        if show_order and len(x) > 1:
            # Start point
            plt.scatter(x[0], y[0],
                        color=color, marker='o', s=100)

            # End point
            plt.scatter(x[-1], y[-1],
                        color=color, marker='x', s=100)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()


def build_leaflet_surface(
        aortic_wall_mask,
        commissure1,
        hinge,
        commissure2,
        boundary_curve_1,
        boundary_curve_2,
        volume_shape,
        pixel_spacing,
        dicom_origin,
        output_path,
        patient_nr,
        file_type,
        n_attachment_pts=20):
    """
    Full pipeline:
    wall → attachment curve → surface → STL
    """


    # 1. Attachment curve
    _, curve_on_wall = get_leaflet_attachment_curve(
        aortic_wall_mask,
        commissure1,
        hinge,
        commissure2,
        n_points=n_attachment_pts,
        snap_to_wall=True
    )

    # 2. Surface interpolation
    curves = [boundary_curve_1,
              boundary_curve_2,
              curve_on_wall]
    
    # Look at how the curves go to determine the fualt
    debug_plot_curves_2d(
    curves,
    labels=["curve1", "curve2", "attachment"]
    )
    surf, eval_pts = create_leaflet_surface_from_curves(curves)

    # 3. STL creation
    mask_3d = surface_points_to_stl(
        eval_pts,
        volume_shape,
        pixel_spacing,
        dicom_origin,
        output_path,
        patient_nr,
        file_type
    )

    return surf, eval_pts, mask_3d

def update_boundary_shrink_state(boundary, state_dict, name, slice_idx, threshold=20):
    """
    Detect first transition from > threshold to < threshold.
    Updates state_dict in place.
    """

    if boundary is None or len(boundary) < 2:
        return None

    length = compute_max_pairwise_distance(boundary)

    # If already detected → stop
    if state_dict[name]["event_slice"] is not None:
        return length

    # Mark that we have seen it open
    if length > threshold:
        state_dict[name]["was_above"] = True

    # Detect crossing
    if length < threshold and state_dict[name]["was_above"]:
        state_dict[name]["event_slice"] = slice_idx
        print(f"[EVENT] {name} shrank below {threshold} at slice {slice_idx}")

    return length

def find_shrink_slice_consecutive(data_dict, shrink_ratio=0.7, consecutive=2):
    """
    Finds first slice AFTER the maximum length slice where boundary length
    drops below shrink_ratio * max_length for N consecutive slices.

    Returns
    -------
    event_slice : int or None
        First slice of the consecutive shrink event
    threshold : float
    max_slice : int
        Slice where maximum occurred
    """

    sorted_slices = sorted(data_dict.keys())

    # Extract valid (slice, length) pairs
    slice_length_pairs = [
        (s, data_dict[s]["boundary_length"])
        for s in sorted_slices
        if data_dict[s]["boundary_length"] is not None
    ]

    if not slice_length_pairs:
        return None, None, None

    # Determine maximum
    max_slice, max_length = max(slice_length_pairs, key=lambda x: x[1])
    threshold = shrink_ratio * max_length

    counter = 0
    start_slice = None

    for s, length in slice_length_pairs:

        # Only evaluate AFTER maximum slice
        if s <= max_slice:
            continue

        if length <= threshold:
            if counter == 0:
                start_slice = s
            counter += 1

            if counter >= consecutive:
                return start_slice, threshold, max_slice
        else:
            counter = 0
            start_slice = None

    return None, threshold, max_slice


def keep_largest_component_3d(volume_mask):
    """
    Keeps only the largest 3D connected component in a binary volume.
    
    Parameters
    ----------
    volume_mask : 3D numpy array (Z, Y, X)
        Binary mask.
        
    Returns
    -------
    largest_component_mask : 3D numpy array
        Binary mask with only the largest component kept.
    """

    labeled = label(volume_mask, connectivity=3)  # full 3D connectivity

    if labeled.max() == 0:
        print("[Info] No connected components found.")
        return np.zeros_like(volume_mask)

    regions = regionprops(labeled)
    largest_region = max(regions, key=lambda r: r.area)

    largest_label = largest_region.label

    largest_component_mask = (labeled == largest_label)

    print(f"[Info] Largest 3D component size: {largest_region.area} voxels")

    return largest_component_mask

def assign_calcifications_3D_voxelwise(calc_mask, LCC_mask, RCC_mask, NCC_mask):
    """
    Assign each calcification voxel to the closest leaflet cap in 3D
    using voxelwise nearest-neighbor assignment (physical distances).

    Parameters
    ----------
    calc_mask : 3D binary array
        Binary mask of all calcifications
    LCC_mask, RCC_mask, NCC_mask : 3D binary arrays
        Binary masks of each leaflet cap
    pixel_spacing : tuple of 3 floats
        Voxel spacing in (z, y, x) order for physical distances

    Returns
    -------
    LCC_calc, RCC_calc, NCC_calc : 3D binary arrays
        Assigned calcifications per leaflet
    """
    # Ensure boolean
    calc_mask = calc_mask.astype(bool)
    LCC_mask = LCC_mask.astype(bool)
    RCC_mask = RCC_mask.astype(bool)
    NCC_mask = NCC_mask.astype(bool)

    # Output volumes
    LCC_calc = np.zeros_like(calc_mask, dtype=np.uint8)
    RCC_calc = np.zeros_like(calc_mask, dtype=np.uint8)
    NCC_calc = np.zeros_like(calc_mask, dtype=np.uint8)

    # Get voxel coordinates
    calc_coords = np.column_stack(np.nonzero(calc_mask))
    LCC_coords = np.column_stack(np.nonzero(LCC_mask))
    RCC_coords = np.column_stack(np.nonzero(RCC_mask))
    NCC_coords = np.column_stack(np.nonzero(NCC_mask))

    if len(calc_coords) == 0:
        print("No calcification voxels found!")
        return LCC_calc, RCC_calc, NCC_calc

    # Build KD-trees
    tree_LCC = cKDTree(LCC_coords)
    tree_RCC = cKDTree(RCC_coords)
    tree_NCC = cKDTree(NCC_coords)

    # Query distances for all calc voxels
    d_LCC, _ = tree_LCC.query(calc_coords)
    d_RCC, _ = tree_RCC.query(calc_coords)
    d_NCC, _ = tree_NCC.query(calc_coords)

    # Assign each voxel to nearest leaflet
    nearest_leaflet = np.argmin(np.stack([d_LCC, d_RCC, d_NCC], axis=1), axis=1)

    LCC_calc[tuple(calc_coords[nearest_leaflet==0].T)] = 1
    RCC_calc[tuple(calc_coords[nearest_leaflet==1].T)] = 1
    NCC_calc[tuple(calc_coords[nearest_leaflet==2].T)] = 1

    # Optional: print summary
    print(f"Assigned calcifications: LCC={np.count_nonzero(LCC_calc)}, "
          f"RCC={np.count_nonzero(RCC_calc)}, NCC={np.count_nonzero(NCC_calc)}, "
          f"Total={np.count_nonzero(LCC_calc)+np.count_nonzero(RCC_calc)+np.count_nonzero(NCC_calc)}")

    return LCC_calc, RCC_calc, NCC_calc


def find_max_length_slice(data_dict):
    """
    Finds the slice where the boundary length is maximal.

    Parameters:
        data_dict (dict): e.g. LCC_data, RCC_data, NCC_data

    Returns:
        max_slice (int): slice index of maximum boundary length
        max_length (float): corresponding length
    """
    max_length = -np.inf
    max_slice = None

    for slice_nr, data in data_dict.items():
        length = data.get("boundary_length", None)

        if length is None:
            continue

        if length > max_length:
            max_length = length
            max_slice = slice_nr

    return max_slice, max_length



def coords_from_mask(mask):
    """Return pixel coordinates of all nonzero pixels in (row, col) format."""
    return np.column_stack(np.nonzero(mask))


from skimage.measure import find_contours

def ordered_coords_from_mask(mask, start_point=None):
    """
    Extract ordered coordinates from a 2D binary mask using skimage.find_contours.
    start_point: optional (row, col) to start ordering from.
    """
    # find_contours returns coordinates in (row, col) as floats
    contours = find_contours(mask.astype(float), 0.5)
    if len(contours) == 0:
        return np.empty((0,2))
    
    # Take the longest contour
    contour = max(contours, key=lambda c: len(c))
    
    # Optionally reorder to start near start_point
    if start_point is not None and contour.shape[0] > 0:
        dists = np.linalg.norm(contour - np.array(start_point), axis=1)
        idx = np.argmin(dists)
        contour = np.vstack([contour[idx:], contour[:idx]])
    
    # Round to integer pixel coordinates
    contour = np.round(contour).astype(int)
    return contour

def build_leaflet_mask(slice_shape, wall_coords, comA, comB, center):
    """
    Build a leaflet mask using wall mask and straight lines to center.
    
    Polygon path:
    comA -> wall -> comB -> center -> comA
    """
    # Extract wall coordinates
    wall_coords = np.round(wall_coords).astype(int)
    
    # Combine polygon points
    polygon_points = [comA]
    if wall_coords.size > 0:
        polygon_points.append(wall_coords)
    polygon_points.append(comB)
    polygon_points.append(center)
    polygon_points.append(comA)  # close polygon
    for i, arr in enumerate(polygon_points):
        print(f"Index {i}: shape = {np.shape(arr)}")
    polygon_points = np.vstack(polygon_points)
    
    # Create mask
    mask = polygon2mask(slice_shape, polygon_points)
    return mask

