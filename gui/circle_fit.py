import numpy as np
import pyvista as pv

# Define commissure points
commissure_1 = np.array([173, 156, 126])
commissure_2 = np.array([234, 171, 126])
commissure_3 = np.array([194, 218, 126])

# Function to compute the circumcenter and radius of a circle passing through three points
def fit_circle_through_points(p1, p2, p3):
    # Compute midpoints
    mid_ab = (p1 + p2) / 2
    mid_bc = (p2 + p3) / 2
    
    # Compute perpendicular vectors
    perp_ab = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])  # Perpendicular to AB
    perp_bc = np.array([-(p3[1] - p2[1]), p3[0] - p2[0]])  # Perpendicular to BC

    # Solve for intersection of perpendicular bisectors (center of circumcircle)
    A = np.array([perp_ab, perp_bc]).T  # Coefficients matrix
    b = np.array([np.dot(perp_ab, mid_ab), np.dot(perp_bc, mid_bc)])  # Right-hand side

    center_2d = np.linalg.solve(A, b)  # Solve for (h, k)
    
    # Compute radius
    radius = np.linalg.norm(center_2d - p1[:2])
    
    return center_2d, radius

# Fit the circle
center_2d, radius = fit_circle_through_points(commissure_1, commissure_2, commissure_3)

# Create a parametric representation of the circle
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = center_2d[0] + radius * np.cos(theta)
y_circle = center_2d[1] + radius * np.sin(theta)

# Use the average z-coordinate for 3D representation
z_circle = np.full_like(x_circle, np.mean([commissure_1[2], commissure_2[2], commissure_3[2]]))

# Create 3D points for the circle
circle_points = np.column_stack((x_circle, y_circle, z_circle))

# Visualization using PyVista
plotter = pv.Plotter()

# Add the circle to the plot
circle_mesh = pv.PolyData(circle_points)
plotter.add_mesh(circle_mesh, color='cyan', line_width=3)

# Add the commissure points
commissure_points = np.array([commissure_1, commissure_2, commissure_3])
commissure_mesh = pv.PolyData(commissure_points)
plotter.add_points(commissure_mesh, color='red', point_size=10)

# Add the computed center
center_3d = np.array([center_2d[0], center_2d[1], np.mean([commissure_1[2], commissure_2[2], commissure_3[2]])])
center_mesh = pv.PolyData(center_3d.reshape(1, -1))
plotter.add_points(center_mesh, color='yellow', point_size=10)

# Show the plot
plotter.show()
