import os 

os.chdir("H:/DATA/Afstuderen/2.Code/Stenosis-Severity/b-spline_fitting")

import numpy as np
import pyvista as pv
from geomdl import fitting
import functions

# Updated control points
control_points = [
    [
        np.array([-1.978389948370209, 75.25113054518995, 119.8155938766923]),
        np.array([4.593810043369483, 79.87395213866128, 120.04625655981678]),
        np.array([11.16601004, 84.49677373, 120.27691924])
    ],
    [
        np.array([7.992229516757652, 75.08135792354658, 116.63201085467692]),
        np.array([9.73874868052191, 79.85866764485552, 117.43725926651581]),
        np.array([11.166110035109174, 84.4967837321326, 120.27692924294125])
    ],
    [
        np.array([13.18538679, 76.23405939, 117.38310994]),
        np.array([12.17569841037245, 80.36541656018179, 117.38310994427064]),
        np.array([11.165910035109174, 84.49676373213259, 120.27690924294124])
    ],
    [
        np.array([15.451703939133594, 79.7098135712322, 123.48147016800833]),
        np.array([13.464171596991275, 82.17101433850873, 120.8894809713516]),
        np.array([11.166210035109174, 84.4967937321326, 120.27693924294125])
    ],
    [
        np.array([13.120887990942187, 84.28431750408579, 133.02860440765406]),
        np.array([12.14344901302568, 84.3905456181092, 126.65276182529766]),
        np.array([11.165810035109175, 84.49675373213259, 120.27689924294124])
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
plotter.show()

