from geomdl import BSpline
from geomdl.visualization import VisVTK
import numpy as np
import vtk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares



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
    

def calc_surface_ctrlpts(commissure_1: list, commissure_2: list, leaflet_tip: list):
    """
    Calculate the control points of the boundaries of a leaflet tip based on landmarks
    
    Parameters:
        Commissure_1, representing one commisure of the leaflet
        Commissure_2, representing the other commisure of the leaflet
        leaflet_tip, which is the tip of the leaflet
        hinge_point, this is the hinge points, the lowest point of the leaflet between the commissures
    Note that the function now only works for a single leaflet. The way that the
    curvature of the arch is now calculate is based on hard-coding
    
    Returns:
        grid of control points representing the control points reconstructing the 
        boundaries of the leaflet.
    """

    annulus_midpoint = midpoint_on_annulus(commissure_1, commissure_2, leaflet_tip)
    
    center = [
    (commissure_1[0] + commissure_2[0] + leaflet_tip[0]) / 3,
    (commissure_1[1] + commissure_2[1] + leaflet_tip[1]) / 3,
    leaflet_tip[2]-10
    ]
    # Calculate the hinge 
    hinge_point=[annulus_midpoint[0], annulus_midpoint[1], leaflet_tip[2]-30]

    arch_control_1=[(leaflet_tip[0]+commissure_1[0])/2, (leaflet_tip[1]+commissure_1[1])/2, (leaflet_tip[2]+commissure_1[2])/2-0.2]
    arch_control_2=[(leaflet_tip[0]+commissure_2[0])/2, (leaflet_tip[1]+commissure_2[1])/2, (leaflet_tip[2]+commissure_2[2])/2-0.2]

    # Define the control grid (3x3 control points)
    control_points = [
        [commissure_1, arch_control_1, leaflet_tip],
        [hinge_point, center, leaflet_tip],
        [commissure_2, arch_control_2, leaflet_tip]
    ]
    
    
    return control_points

def reconstruct_surface(control_points, degree_u=2, degree_v=2, knotvector_u=None, knotvector_v=None, delta=0.02):
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
        knotvector_u = [0, 0, 0, 1, 1, 1]
    if knotvector_v is None:
        knotvector_v = [0, 0, 0, 1, 1, 1]

    surf.knotvector_u = knotvector_u
    surf.knotvector_v = knotvector_v
    surf.delta = delta

    # Evaluate the surface
    surf.evaluate()
    return surf


def calc_wall_ctrlpts(commissure_1: list, commissure_2: list, leaflet_tip: list, center: list, degree_u=2, degree_v=2, knotvector_u=None, knotvector_v=None, delta=0.02):
    """
    Constructs and evaluates a B-Spline surface bound by the commissures and the hinge point.
    
    Parameters:
        commissure_1 (list): 3D coordinates of the first commissure point.
        commissure_2 (list): 3D coordinates of the second commissure point.
        annulus midpoint (list): 3D coordinates of the point between commissures on the annulus
        hinge_point (list): 3D coordinates of the hinge point.
        degree_u (int): Degree in the U direction (default: 2).
        degree_v (int): Degree in the V direction (default: 2).
        knotvector_u (list, optional): Knot vector for U direction.
        knotvector_v (list, optional): Knot vector for V direction.
        delta (float): Surface evaluation resolution (default: 0.02).
    
    Returns:
        BSpline.Surface: The reconstructed and evaluated B-Spline surface.
    """
    annulus_midpoint = midpoint_on_annulus(center, commissure_1, commissure_2)
    
    # Calculate the hinge 
    hinge_point=[annulus_midpoint[0], round(annulus_midpoint[1],2), 0]

    # Define the center of the wall 
    center_wall = [
    (annulus_midpoint[0]), 
    (annulus_midpoint[1]),
    (annulus_midpoint[2]+hinge_point[2])/2
    ]
    
    # Define a grid for the control points representing the wall
    control_points = [
        [commissure_1, annulus_midpoint, commissure_2],
        [commissure_1, center_wall, commissure_2],
        [commissure_1, hinge_point, commissure_2]
    ]
    
    control_points = [
        [[0, 0, 1], annulus_midpoint, [1, 2, 1]], 
        [[0, 0, 1], [0.3423658394074329, 1.556026136348903, 0.5], [1, 2, 1]], 
        [[0, 0, 1], [0.3423658394074329, 1.56, 0], [1, 2, 1]]
    ]

        
    return control_points


def midpoint_on_annulus(commissure_1, commissure_2, center):
    """
    Calculate the midpoint on a circular annulus between two commissures.

    Parameters:
        commissure_1 (list): [x, y, z] coordinates of first commissure.
        commissure_2 (list): [x, y, z] coordinates of second commissure.
        center (list): [x, y, z] coordinates of the annulus center.

    Returns:
        list: [x, y, z] coordinates of the midpoint.
    """
    # Convert to polar coordinates (relative to center)
    vec1 = np.array(commissure_1) - np.array(center)
    vec2 = np.array(commissure_2) - np.array(center)
    
    # Compute angles of each commissure relative to the center
    theta_1 = np.arctan2(vec1[1], vec1[0])
    theta_2 = np.arctan2(vec2[1], vec2[0])

    # Ensure that the angle difference is the smaller of the two possible angles
    theta_diff = np.abs(theta_2 - theta_1)
    if theta_diff > np.pi:
        # If the angle difference is larger than 180 degrees, use the other direction
        if theta_2 > theta_1:
            theta_1 += 2 * np.pi
        else:
            theta_2 += 2 * np.pi
    
    # Find the midpoint angle (average of the two angles)
    theta_mid = (theta_1 + theta_2) / 2

    # Compute radius (assumed to be the same for both commissures)
    R = np.linalg.norm(vec1)

    # Compute the midpoint coordinates (preserve the Z-coordinate)
    midpoint = [
        center[0] + R * np.cos(theta_mid),  # X-coordinate
        center[1] + R * np.sin(theta_mid),  # Y-coordinate
        commissure_1[2]  # Z-coordinate from commissure_1 (same Z as annulus)
    ]
    
    return midpoint

#%% Circle
# Define the residual function for the ellipsoid equation
def ellipsoid_residual(params, points):
    # Unpack parameters (center and radii)
    xc, yc, zc, a, b, c = params
    
    # Compute the residuals (differences from the ellipsoid equation)
    residuals = ((points[:, 0] - xc) ** 2) / a**2 + ((points[:, 1] - yc) ** 2) / b**2 + ((points[:, 2] - zc) ** 2) / c**2 - 1
    
    return residuals

def fit_ellipsoid(points):
    # Initial guess for the parameters (center and semi-axes)
    # Center at the centroid and semi-axes as the average distances from the center
    centroid = np.mean(points, axis=0)
    initial_guess = np.concatenate([centroid, np.ones(3)])

    # Use least-squares optimization to minimize the residuals
    result = least_squares(ellipsoid_residual, initial_guess, args=(points,))
    
    # Return the optimized parameters (center and semi-axes)
    return result.x

# Function to generate ellipsoid points
def generate_ellipsoid(center, a, b, c, num_points=100):
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = center[0] + a * np.outer(np.cos(u), np.sin(v))
    y = center[1] + b * np.outer(np.sin(u), np.sin(v))
    z = center[2] + c * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

# Function to export circle to VTK
def export_to_vtk(circle_points, filename="circle.vtp"):
    # Convert numpy array to vtk
    vtk_points = vtk.vtkPoints()
    for point in circle_points:
        vtk_points.InsertNextPoint(point)
    
    # Create a polyline to connect the points
    polyline = vtk.vtkCellArray()
    for i in range(len(circle_points) - 1):
        polyline.InsertNextCell(2)
        polyline.InsertCellPoint(i)
        polyline.InsertCellPoint(i + 1)
    
    # Create a PolyData object to hold the circle data
    circle_polydata = vtk.vtkPolyData()
    circle_polydata.SetPoints(vtk_points)
    circle_polydata.SetLines(polyline)
    
    # Write the PolyData to a VTK XML file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(circle_polydata)
    writer.Write()
    print(f"VTK file saved as: {filename}")


