import os 

os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/b-spline_fitting")

import numpy as np
import pyvista as pv
from geomdl import fitting
import functions


# Updated control points
control_points = [
    [
        np.array([4.287072158296684, 97.62467910624702, 119.94100278880288]),
        np.array([7.726541096702929, 91.0607264191898, 117.17947416182642]),
        np.array([11.16601004, 84.49677373, 120.27691924])
    ],
    [
        np.array([1.5113105074725364, 94.17310714319898, 110.3504355635508]),
        np.array([6.428266831040636, 89.60127899854871, 109.84676752972352]),
        np.array([11.166020035109174, 84.4967747321326, 120.27691924294125])
    ],
    [
        np.array([1.02968715e-01, 9.10302544e+01, 1.07538349e+02]),
        np.array([5.63448937498317, 87.76351404236993, 107.53834944978088]),
        np.array([11.166000035109175, 84.4967727321326, 120.27691924294125])
    ],
    [
        np.array([-1.6208055303567928, 82.8139212497673, 110.54544633207189]),
        np.array([4.859787013213816, 83.91448771234957, 110.01363476513342]),
        np.array([11.166030035109173, 84.4967747321326, 120.27691924294125])
    ],
    [
        np.array([-1.978389948370209, 75.25113054518995, 119.8155938766923]),
        np.array([4.593810043369483, 79.87395213866128, 117.1182990827481]),
        np.array([11.165990035109175, 84.4967727321326, 120.27691924294125])
    ]
]



size_u = 5
size_v = 3
degree_u = 2
degree_v = 2

# Flatten control points correctly
control_points_flat = [pt.tolist() for row in control_points for pt in row]

# Fit the NURBS surface
surface = fitting.interpolate_surface(control_points_flat, size_u, size_v, degree_u, degree_v)

# Export and visualize
vtk_path = r"H:\DATA\Afstuderen\2.Code\temporary\temporary_surface.vtk"
functions.export_vtk(surface, vtk_path)
mesh = pv.read(vtk_path)

plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightgray", opacity=0.7)

# Add the individual control points
control_points_flat_np = np.array(control_points_flat)
plotter.add_points(control_points_flat_np, color="red", point_size=10)

plotter.show()

