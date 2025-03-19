import sys
import numpy as np
import pyvista as pv
from geomdl import BSpline
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout


class BSplineSurfaceEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("B-Spline Surface Editor")
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Create B-spline surface
        self.surf = BSpline.Surface()
        self.surf.degree_u = 2
        self.surf.degree_v = 2

        # Define control points as a 2D list (3x3 grid)
        self.control_points = np.array([
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [2, 1, 0]],
            [[0, 2, 0], [1, 2, 0], [2, 2, 0]]
        ])
        
        # Convert to a flat list and set control points
        num_u, num_v = self.control_points.shape[:2]
        self.surf.set_ctrlpts(self.control_points.reshape(-1, 3).tolist(), num_u, num_v)
        
        # Knot vectors
        self.surf.knotvector_u = [0, 0, 0, 1, 1, 1]
        self.surf.knotvector_v = [0, 0, 0, 1, 1, 1]
        
        # Evaluate the surface
        self.surf.evaluate()

        # PyVista visualization setup
        self.plotter = pv.Plotter()
        
        # Add the B-spline surface
        self.surface_mesh = self.get_surface_mesh()
        self.plotter.add_mesh(self.surface_mesh, color="lightblue", show_edges=True)
        
        # Add control points as interactive spheres
        self.control_points_mesh = pv.PolyData(self.control_points.reshape(-1, 3))
        self.plotter.add_mesh(self.control_points_mesh, render_points_as_spheres=True, point_size=15, color="red", pickable=True)
        
        # Enable point picking and set the callback
        self.plotter.enable_point_picking(callback=self.on_point_picked, use_mesh=True, show_message=True)
        
        # Display the window
        self.setGeometry(100, 100, 800, 600)
        self.show()
        self.plotter.show()

    def get_surface_mesh(self):
        """Function to get the surface mesh for visualization."""
        eval_points = np.array(self.surf.evalpts)
        n_u, n_v = 20, 20  # Adjust based on sample size
        x = eval_points[:, 0].reshape((n_u, n_v))
        y = eval_points[:, 1].reshape((n_u, n_v))
        z = eval_points[:, 2].reshape((n_u, n_v))
        return pv.StructuredGrid(x, y, z)

    def on_point_picked(self, point):
        """Callback function for point picking."""
        # Find the nearest control point
        distances = np.linalg.norm(self.control_points.reshape(-1, 3) - point, axis=1)
        idx = np.argmin(distances)

        # Update the control point position
        self.control_points.reshape(-1, 3)[idx] = point
        
        # Update the B-spline surface
        self.surf.set_ctrlpts(self.control_points.reshape(-1, 3).tolist(), *self.control_points.shape[:2])
        self.surf.evaluate()

        # Update the surface and control points in PyVista
        self.plotter.clear()
        self.plotter.add_mesh(self.get_surface_mesh(), color="lightblue", show_edges=True)
        self.control_points_mesh = pv.PolyData(self.control_points.reshape(-1, 3))
        self.plotter.add_mesh(self.control_points_mesh, render_points_as_spheres=True, point_size=15, color="red", pickable=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = BSplineSurfaceEditor()
    sys.exit(app.exec_())
