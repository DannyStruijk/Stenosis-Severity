from geomdl import BSpline
import numpy as np
import vtk
from geomdl import fitting
import os
import pyvista as pv
import matplotlib.pyplot as plt
import pydicom
from scipy.ndimage import zoom
from scipy.ndimage import affine_transform

def export_vtk(surface, filename="surface.vtk"):
    """
    Save a B-Spline surface to a VTK file for visualization in ParaView.
    
    Parameters:
    - surface: geomdl BSpline.Surface object
    - filename: Name of the output VTK file (default: 'surface.vtk')
    """
    # Ensure surface is evaluated
    surface.evaluate()

    # Get surface points
    points = np.array(surface.evalpts)  # List of (x, y, z) tuples

    # Create VTK points array
    vtk_points = vtk.vtkPoints()
    for p in points:
        vtk_points.InsertNextPoint(p)

    # Create VTK polygonal data
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    # Create a triangulation of the points
    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(polydata)
    delaunay.Update()

    # Write to VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(delaunay.GetOutput())
    writer.Write()
    
    print(f"Surface saved as {filename}")
    
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


def calc_ctrlpoints(cusp_landmarks, leaflet_tip, delta=1e-3):
    """
    Function to calculate control points for a B-spline surface,
    estimating intermediate points by averaging instead of spline fitting.

    Parameters:
        cusp_landmarks: List of three points [commissure_1, commissure_2, hinge]
        leaflet_tip: The leaflet tip coordinate [x, y, z]
        delta: Small value used to differentiate the leaflet tip points

    Returns:
        A 5x3 matrix representing control points for surface interpolation.
    """
    # Extract the relevant annotations for one cusp
    commissure_1 = np.array(cusp_landmarks[0])
    commissure_2 = np.array(cusp_landmarks[1])
    hinge = np.array(cusp_landmarks[2])
    leaflet_tip = np.array(leaflet_tip)
    
    # Center point between hinge and leaflet tip
    center = (hinge + leaflet_tip) / 2

    # Arch control points between leaflet tip and commissures
    arch_control_1 = (leaflet_tip + commissure_1) / 2
    arch_control_2 = (leaflet_tip + commissure_2) / 2

    # Small perturbations around the leaflet tip
    leaflet_tip_1 = leaflet_tip + np.array([delta,     delta / 10, 0])
    leaflet_tip_2 = leaflet_tip - np.array([delta,     delta / 10, 0])
    leaflet_tip_3 = leaflet_tip + np.array([2*delta,   delta / 10, 0])
    leaflet_tip_4 = leaflet_tip - np.array([2*delta,   delta / 10, 0])

    # Estimate intermediate points as averages instead of spline fitting
    hinge_arch_1 = (commissure_1 + hinge) / 2
    hinge_arch_2 = (commissure_2 + hinge) / 2

    center_left = (arch_control_1 + center) / 2
    center_right = (arch_control_2 + center) / 2

    arch_left_1 = (commissure_1 + arch_control_1) / 2
    arch_left_2 = (arch_control_1 + leaflet_tip) / 2

    arch_right_1 = (commissure_2 + arch_control_2) / 2
    arch_right_2 = (arch_control_2 + leaflet_tip) / 2

    left_hinge_1 = (hinge_arch_1 + center_left) / 2
    left_hinge_2 = (center_left + leaflet_tip_1) / 2

    right_hinge_1 = (hinge_arch_2 + center_right) / 2
    right_hinge_2 = (center_right + leaflet_tip_3) / 2

    center_mid_1 = (hinge + center) / 2
    center_mid_2 = (center + leaflet_tip_2) / 2

    # Final 5x3 control grid
    control_points = np.array([
        [commissure_1, arch_left_1, arch_control_1, arch_left_2, leaflet_tip],
        [hinge_arch_1, left_hinge_1, center_left,   left_hinge_2, leaflet_tip_1],
        [hinge,       center_mid_1,  center,        center_mid_2,  leaflet_tip_2],
        [hinge_arch_2, right_hinge_1, center_right, right_hinge_2, leaflet_tip_3],
        [commissure_2, arch_right_1, arch_control_2, arch_right_2, leaflet_tip_4]
    ], dtype=np.float64)

    return control_points


def reconstruct_surface(control_points, degree_u=2, degree_v=2, knotvector_u=None, knotvector_v=None, delta=0.01):
    """
    Constructs and evaluates a NURBS surface from given control points.

    Parameters:
        control_points (list): 2D grid of 3D control points.
        degree_u (int): Degree in the U direction (default: 2).
        degree_v (int): Degree in the V direction (default: 2).
        knotvector_u (list, optional): Knot vector for U direction.
        knotvector_v (list, optional): Knot vector for V direction.
        delta (float): Surface evaluation resolution (default: 0.05).

    Returns:
        BSpline.Surface: The reconstructed and evaluated NURBS surface.
    """

    # Create the NURBS surface
    surf = BSpline.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    surf.ctrlpts2d = control_points

    # Define default knot vectors if not provided
    if knotvector_u is None:
        knotvector_u = [0, 0, 1, 1]
    if knotvector_v is None:
        knotvector_v = [0, 0, 1, 1]

    surf.knotvector_u = knotvector_u
    surf.knotvector_v = knotvector_v
    surf.delta = delta

    # Evaluate the surface
    surf.evaluate()
    return surf


def interpolate_surface(interp_points):
    """
    Constructs a NURBS surface from interp_points and prints debug info.
    """

    print(f"Input interp_points type: {type(interp_points)}")
    print(f"Input interp_points shape: {getattr(interp_points, 'shape', 'no shape attribute')}")

    size_u = interp_points.shape[0]
    size_v = interp_points.shape[1]
    print(f"Size_u: {size_u}, Size_v: {size_v}")

    # Convert points to a flat tuple of 3D tuples matching the working example format
    interp_points_2d = tuple(tuple(pt) for pt in interp_points.reshape(-1, 3))

    print(f"Reshaped interp_points_2d type: {type(interp_points_2d)}")
    print(f"Number of points: {len(interp_points_2d)}")
    print(interp_points_2d)

    degree_u = 2
    degree_v = 2
    print(f"Degree_u: {degree_u}, Degree_v: {degree_v}")

    surf = fitting.approximate_surface(interp_points_2d, size_u, size_v, degree_u, degree_v)

    surf.evaluate()
    return surf



def save_surface_evalpts(surface, save_path):
    """
    Saves the evaluated points (evalpts) of a NURBS surface to a .npy file.

    Parameters:
        surface : geomdl.NURBS.Surface
            The surface object after evaluation.
        save_path : str
            Path to save the .npy file.
    """
    if not surface.evalpts:
        raise ValueError("Surface has not been evaluated. Run surface.evaluate() first.")

    evalpts_array = np.array(surface.evalpts)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savetxt(save_path, evalpts_array)
    print(f"Surface evalpts saved to {save_path}")
    

def save_ordered_landmarks(cusp_landmarks, center, base_folder):
    """
    Saves landmarks of each leaflet (NCC, LCC, RCC) separately, 
    each containing its 3 landmarks (2 commissures and 1 hinge) plus the center point,
    into separate text files under the specified base folder.

    Commissures are saved in a fixed clockwise order, with rules defined per cusp:
    
    - RCC (most anterior cusp): save commissure with lower L (rightmost) first
    - LCC (most left cusp): save commissure with lower P (most anterior) first
    - NCC (remaining cusp): save commissure with higher P (most posterior) first

    Args:
        cusp_landmarks (list or np.ndarray): List of 3 cusp landmarks, each with 3 points (each point is length 3).
                                             Each cusp: [commissure_1, commissure_2, hinge]
        center (list or np.ndarray): Center landmark (length 3).
        base_folder (str): Folder path where landmark files are saved.
    """
    hinges = [cusp[2] for cusp in cusp_landmarks]
    L_values = [h[0] for h in hinges]
    P_values = [h[1] for h in hinges]

    rcc_idx = np.argmin(P_values)  # most anterior (smallest P)
    lcc_idx = np.argmax(L_values)  # most left (largest L)
    ncc_idx = list({0, 1, 2} - {rcc_idx, lcc_idx})[0]  # remaining index

    leaflet_indices = {
        "rcc": rcc_idx,
        "lcc": lcc_idx,
        "ncc": ncc_idx
    }

    os.makedirs(base_folder, exist_ok=True)

    for leaflet_name, idx in leaflet_indices.items():
        commissure_1, commissure_2, hinge = cusp_landmarks[idx]

        if leaflet_name == "rcc":
            # Rightmost commissure (lowest L) first
            if commissure_1[0] < commissure_2[0]:
                ordered_commissures = [commissure_1, commissure_2]
            else:
                ordered_commissures = [commissure_2, commissure_1]

        elif leaflet_name == "lcc":
            # Most anterior commissure (lowest P) first
            if commissure_1[1] < commissure_2[1]:
                ordered_commissures = [commissure_1, commissure_2]
            else:
                ordered_commissures = [commissure_2, commissure_1]

        elif leaflet_name == "ncc":
            # Most posterior commissure (highest P) first
            if commissure_1[1] > commissure_2[1]:
                ordered_commissures = [commissure_1, commissure_2]
            else:
                ordered_commissures = [commissure_2, commissure_1]

        leaflet_landmarks = ordered_commissures + [hinge, center]

        arr_to_save = np.array(leaflet_landmarks)
        output_path = os.path.join(base_folder, f"{leaflet_name}_template_landmarks_test.txt")
        np.savetxt(output_path, arr_to_save, fmt="%.6f")

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


def plot_control_points(control_points):
    """
    Plots control points with PyVista, including labels and connecting lines.
    
    Parameters:
        control_points: List or 2D array of control points with shape (rows, cols, 3)
    """
    control_points = np.array(control_points)
    rows, cols = control_points.shape[:2]

    # Flatten points for plotting
    points_flat = control_points.reshape(-1, 3)

    # Create PyVista point cloud
    point_cloud = pv.PolyData(points_flat)

    # Create plotter
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color='red', point_size=10, render_points_as_spheres=True)

    # Add labels for each control point as (row, col)
    for r in range(rows):
        for c in range(cols):
            point = control_points[r, c]
            label = f"({r},{c})"
            plotter.add_point_labels(point, [label], font_size=12, text_color='black', point_color='yellow', shape_opacity=0.6)


    plotter.add_axes()
    plotter.show()


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
    
    print("Rotation matrix: ", R)

    M = np.linalg.inv(R)

    # Offset so rotation is around this point
    center = np.array(volume.shape) / 2
    offset = center - M @ center
    
    reoriented_volume = affine_transform(volume, M, offset=offset, order=1)

    return reoriented_volume, R, center


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
    print("Spacing" , spacing)
    
    landmarks_lps = np.loadtxt(txt_file)
    voxel_coords = (landmarks_lps-origin)/spacing
    voxel_coords = np.round(voxel_coords).astype(int)
    
    return voxel_coords

import numpy as np

def reorient_landmarks(landmarks_zyx, R, dicom_origin, spacing, output_shape):
    """
    Reorient landmarks using the same affine transform as the volume.
    The grid is the size of the original volume; output is in the rotated volume grid.

    Parameters
    ----------
    landmarks_voxel : (N,3) array
        Landmarks in voxel coordinates (x,y,z) of the original volume.
    R : (3,3) array
        Rotation matrix used for the volume (zyx convention).
    dicom_origin : array-like
        ImagePositionPatient of the first voxel (LPS), in z,y,x order.
    spacing : array-like
        Voxel spacing (dz,dy,dx), matching z,y,x order.
    orig_shape : tuple
        Shape of the original volume (z,y,x)
    output_shape : tuple
        Shape of the reoriented volume (z,y,x)

    Returns
    -------
    rotated_landmarks_voxel : (M,3) array
        Landmark coordinates in the reoriented volume grid (z,y,x order).
    """
    # Create empty grid of the original volume size
    landmark_grid = np.zeros(output_shape, dtype=np.uint8)
    # Place landmarks in the grid
    for z, y, x in landmarks_zyx:
        landmark_grid[z, y, x] = 1
        
    # Check: how many 1s before rotation
    n_before = np.count_nonzero(landmark_grid)
    print(f"Number of landmarks before affine transform: {n_before}")
    
    # Inverse matrix as affine_transform is backwards mapping
    M = np.linalg.inv(R)

    # Offset so rotation is around this point, using the center of the image
    center = np.array(output_shape) / 2
    offset = center - M @ center

    # Apply affine transform to match rotated volume
    rotated_grid = affine_transform(landmark_grid, M, offset=offset, order=0)
    
    # Check: how many 1s before rotation
    n_before = np.count_nonzero(rotated_grid)
    print(f"Number of landmarks after affine transform: {n_before}")

    # Extract coordinates of landmarks in the rotated volume
    rotated_landmarks = np.argwhere(rotated_grid > 0)  # z,y,x

    return rotated_landmarks


def vtk_to_pointcloud(filename, origin, pixel_spacing):
    """
    Load a VTK PolyData surface and return points in voxel coordinates
    aligned with a DICOM volume grid.

    Parameters
    ----------
    filename : str
        Path to VTK PolyData file.
    origin : tuple of float
        DICOM origin (LPS coordinates of voxel 0,0,0).
    spacing : tuple of float
        DICOM voxel spacing (dx, dy, dz).

    Returns
    -------
    voxel_points : ndarray
        Array of shape (N,3) containing points in voxel coordinates (z,y,x order if desired).
    """
    # Load VTK
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()

    spacing = pixel_spacing[::-1]
    
    # Extract points
    points_vtk = polydata.GetPoints()
    points = np.array([points_vtk.GetPoint(i) for i in range(points_vtk.GetNumberOfPoints())])

    # Convert LPS coordinates to voxel coordinates
    voxel_points = np.array(points - np.array(origin)) / np.array(spacing)

    return voxel_points


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

from scipy.spatial.distance import cdist

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


import networkx as nx
from scipy.spatial import distance_matrix

def mst_backbone_path(point_cloud, start_idx, end_idx):
    """
    Extract the backbone from a 2D point cloud as the unique path in MST
    between start and end nodes.
    
    Parameters:
        point_cloud (np.ndarray): N x 2 array of points
        start_idx (int): index of start point
        end_idx (int): index of end point
    
    Returns:
        ordered_points (np.ndarray): points along the backbone from start to end
    """
    N = point_cloud.shape[0]
    
    # Build fully connected graph
    dist_mat = distance_matrix(point_cloud, point_cloud)
    G = nx.Graph()
    for i in range(N):
        for j in range(i+1, N):
            G.add_edge(i, j, weight=dist_mat[i, j])
    
    # Compute MST
    mst = nx.minimum_spanning_tree(G)
    
    # Get the unique path between start and end in the MST
    path_indices = nx.shortest_path(mst, source=start_idx, target=end_idx, weight='weight')
    
    # Extract points
    ordered_points = point_cloud[path_indices]
    return ordered_points


def circle_through_commissures(points, n_points=30):
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
