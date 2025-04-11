from geomdl import BSpline
from geomdl.visualization import VisVTK
import numpy as np
import vtk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from geomdl import fitting
import pyvista as pv


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
    
    # Visualize the results using PyVista
    # Convert the evaluated points into a format PyVista can visualize (array of points)
    eval_pts = np.array(eval_pts)
    
    # Create a PyVista point cloud from the evaluated points
    point_cloud = pv.PolyData(eval_pts)
    
    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add the B-spline curve points to the plotter
    plotter.add_mesh(point_cloud, color='blue', point_size=10, render_points_as_spheres=True)

    # Add the input points (to show them in the visualization as well)
    input_points = np.array([point_1, point_2, arch_control])
    input_points_mesh = pv.PolyData(input_points)
    plotter.add_mesh(input_points_mesh, color='red', point_size=15, render_points_as_spheres=True)

    # Add arch_1 and arch_2 points
    arch_points = np.array([arch_1, arch_2])
    arch_points_mesh = pv.PolyData(arch_points)
    plotter.add_mesh(arch_points_mesh, color='green', point_size=15, render_points_as_spheres=True)

    # Optionally, add labels to the input points
    plotter.add_point_labels(input_points, ['Point 1', 'Point 2', 'Arch Control'], font_size=12)
    plotter.add_point_labels(arch_points, ['Arch 1', 'Arch 2'], font_size=12)

    # Show the plot
    #plotter.show()

    return arch_1, arch_2

def calc_surface_ctrlpts_hinge(cusp_landmarks, leaflet_tip: list):
    """
    Calculate the control points of the boundaries of a leaflet tip based on landmarks
    
    Parameters:
        Cusp_landmarks comprises the commissures and the hinge point of a specific cusp
        leaflet_tip, which is the tip of the leaflet
    Note that the function now only works for a single leaflet. The way that the
    curvature of the arch is now calculate is based on hard-coding
    
    Returns:
        grid of control points representing the control points reconstructing the 
        boundaries of the leaflet.
    """
    
    commissure_1 = cusp_landmarks[0]
    commissure_2 = cusp_landmarks[1]
    hinge = cusp_landmarks[2]
    
    center = [
        (hinge[0] + leaflet_tip[0]) / 2,
        (hinge[1]+ leaflet_tip[1]) / 2,
        hinge[2]  # Keep the z-coordinate unchanged
    ]

    arch_control_1=[(leaflet_tip[0]+commissure_1[0])/2, (leaflet_tip[1]+commissure_1[1])/2, (leaflet_tip[2]+commissure_1[2])/2.05]
    arch_control_2=[(leaflet_tip[0]+commissure_2[0])/2, (leaflet_tip[1]+commissure_2[1])/2, (leaflet_tip[2]+commissure_2[2])/2.05]

    leaflet_tip_1 = [leaflet_tip[0]+0.0001, leaflet_tip[1]+0.00001, leaflet_tip[2]+0.00001]
    leaflet_tip_2 = [leaflet_tip[0]-0.0001, leaflet_tip[1]-0.00001, leaflet_tip[2]-0.00001]

    # Define the control grid (3x3 control points)
    control_points = [
        [commissure_1, arch_control_1, leaflet_tip_1],
        [hinge, center, leaflet_tip],
        [commissure_2, arch_control_2, leaflet_tip_2]
    ]
    
    
    return control_points

def calc_additional_ctrlpoints(cusp_landmarks, leaflet_tip):
        """
    Function to calculate additional control points based on the evaluation points of a B-spline surface.
    
    Parameters:
        surf: A B-spline surface object generated by a 3x3 grid of control points.
        
    Returns:
        A 5x3 matrix representing additional control points based on the evaluation points.
    """
        # Extract the relevant annotations for one cusp
        commissure_1 = cusp_landmarks[0]
        commissure_2 = cusp_landmarks[1]
        hinge = cusp_landmarks[2]
        
        # Determining the center of the cusp. Assumed that it is just as high (Z-coordinate) as the hinge
        center = [
            (hinge[0] + leaflet_tip[0]) / 2,
            (hinge[1]+ leaflet_tip[1]) / 2,
            hinge[2]  # Keep the z-coordinate unchanged
        ]
    
        # Determine the arch between the commissures and the leaflet tip
        arch_control_1=[(leaflet_tip[0]+commissure_1[0])/2, (leaflet_tip[1]+commissure_1[1])/2, (leaflet_tip[2]+commissure_1[2])/2.05]
        arch_control_2=[(leaflet_tip[0]+commissure_2[0])/2, (leaflet_tip[1]+commissure_2[1])/2, (leaflet_tip[2]+commissure_2[2])/2.05]
    
        # Create 5 leaflet tip points in order to solve issue with 5 points being the same (reconstruct function cannot handle it)
        leaflet_tip_1 = [leaflet_tip[0]+0.00001, leaflet_tip[1]+0.000001, leaflet_tip[2]]
        leaflet_tip_2 = [leaflet_tip[0]-0.00001, leaflet_tip[1]-0.000001, leaflet_tip[2]]
        leaflet_tip_3 = [leaflet_tip[0]+0.00002, leaflet_tip[1]+0.000001, leaflet_tip[2]]
        leaflet_tip_4 = [leaflet_tip[0]-0.00002, leaflet_tip[1]-0.000001, leaflet_tip[2]]
        
        # center_1 = [center[0]+0.0001, center[1]+0.00001, center[2]+0.00001]
        # center_2 = [center[0]-0.0001, center[1]-0.00001, center[2]-0.00001]
    
        # Determine, by means of Bsplines, other relevant landmarks based on annotations 
        hinge_arch_1, hinge_arch_2 = fit_spline_pts(commissure_1, commissure_2, hinge)
        arch_left_1, arch_left_2 = fit_spline_pts(commissure_1, leaflet_tip, arch_control_1)
        arch_right_1, arch_right_2 = fit_spline_pts(commissure_2, leaflet_tip, arch_control_2)
        center_left, center_right= fit_spline_pts(arch_control_1, arch_control_2, center)

        # Define the control grid (5x5 control pointsm, DOES NOT YET WORK!)
        # control_points = [
        #     [commissure_1, arch_left_1, arch_control_1, arch_left_2, leaflet_tip],
        #     [hinge_arch_1, center_1, center_left, center_2, leaflet_tip_1],
        #     [hinge, center_1, center, center_2, leaflet_tip_2],
        #     [hinge_arch_2, center_1, center_right, center_2, leaflet_tip_3],
        #     [commissure_2, arch_right_1, arch_control_2, arch_right_2, leaflet_tip_4]
        # ]
        
        control_points = [
            [commissure_1, arch_control_1, leaflet_tip],
            [hinge_arch_1, center_left, leaflet_tip_1],
            [hinge, center, leaflet_tip_2],
            [hinge_arch_2, center_right,leaflet_tip_3],
            [commissure_2, arch_control_2, leaflet_tip_4]
        ]
        
        return control_points

def reconstruct_surface(control_points, degree_u=2, degree_v=2, knotvector_u=None, knotvector_v=None, delta=0.005):
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
    Constructs a NURBS surface that exactly passes through the hinge point.
    
    Parameters:
        interp_points: List of points (some might be NumPy arrays or Python lists).
    
    Returns:
        Interpolated NURBS surface object.
    """
    # Convert all points to lists if they are NumPy arrays
    interp_points = [pt for row in interp_points for pt in row]


    # Define the number of points in each direction (3x3 grid)
    size_u, size_v = 5, 3

    # Construct interpolated surface
    surf = fitting.interpolate_surface(interp_points, size_u, size_v, degree_u=2, degree_v=2)

    # Evaluate surface
    surf.evaluate()

    return surf


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


