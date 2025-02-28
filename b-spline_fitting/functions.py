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
