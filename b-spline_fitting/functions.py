from geomdl import BSpline
from geomdl.visualization import VisMPL
import numpy as np
import vtk

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

# Save the surface as a VTK file
#save_surface_as_vtk(surf, "H:/DATA/Afstuderen/2. Code/Stenosis-Severity/b-spline_fitting/nurbs_surface.vtk")

def calc_boundary_ctrlpts(commissure_1: list, commissure_2: list, leaflet_tip: list, hinge_point: list):
    """
    Calculate the control points of the boundaries of a leaflet tip based on landmarks
    
    Arguments:
        - Commisure_1, representing one commisure of the leaflet
        - Commisure_2, representing the other commisure of the leaflet
        - leaflet_tip, which is the tip of the leaflet
        - hinge_point, this is the hinge points, the lowest point of the leaflet between the commissures
    Note that the function now only works for a single leaflet. The way that the
    curvature of the arch is now calculate is based on hard-coding
    
    The function returns a grid of control points representing the control points reconstructing the 
    boundaries of the leaflet.
    """
    arch_control_1 = [(leaflet_tip[0] + commissure_1[0]) / 2, (leaflet_tip[1] + hinge_point[1]) / 2, (leaflet_tip[2] + hinge_point[2]) / 1.5]
    arch_control_2 = [(leaflet_tip[0] + commissure_2[0]) / 2, (leaflet_tip[1] + hinge_point[1]) / 2, (leaflet_tip[2] + hinge_point[2]) / 1.5]

    # Define the control grid (3x3 control points)
    control_points = [
        [commissure_1, arch_control_1, leaflet_tip],
        [hinge_point, hinge_point, hinge_point],
        [commissure_2, arch_control_2, leaflet_tip]
    ]
    
    return control_points




#def create_boundaries(commissure_1: list, commissure_2: list, leaflet_tip: list):
    