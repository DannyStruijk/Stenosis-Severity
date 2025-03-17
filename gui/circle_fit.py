import numpy as np
import pyvista as pv
from scipy.optimize import minimize
from numpy.linalg import svd

# Define control points based on the user's input (3D)
commissure_1 = np.array([173, 156, 126])
commissure_2 = np.array([234, 171, 126])
commissure_3 = np.array([194, 218, 126])

# Step 1: Fit a plane to the three 3D points
# Compute vectors from point 1 to points 2 and 3
v1 = commissure_2 - commissure_1
v2 = commissure_3 - commissure_1

# Compute the normal vector to the plane (cross product)
normal_vector = np.cross(v1, v2)
normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the normal vector

# Step 2: Project points onto the plane
# Create a rotation matrix to align the normal vector with the Z-axis
rotation_matrix = np.array([normal_vector, np.cross([0, 0, 1], normal_vector), [0, 0, 1]]).T

# Apply the rotation to the points
rotated_commissures = np.dot(rotation_matrix, (np.array([commissure_1, commissure_2, commissure_3]).T - commissure_1).T).T

# Now we fit an ellipse in 2D on the rotated points (ignoring the new z-coordinate)
points_2d = rotated_commissures[:, :2]

# Function to fit an ellipse to 2D points
def fit_ellipse(points):
    def ellipse_residual(params, points):
        a, b, h, k, theta = params
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        residual = 0
        for x, y in points:
            x_rot = cos_theta * (x - h) + sin_theta * (y - k)
            y_rot = -sin_theta * (x - h) + cos_theta * (y - k)
            residual += ((x_rot**2) / a**2 + (y_rot**2) / b**2 - 1)**2
        return residual
    
    a_init = 1
    b_init = 1
    h_init = np.mean(points[:, 0])
    k_init = np.mean(points[:, 1])
    theta_init = 0

    initial_params = [a_init, b_init, h_init, k_init, theta_init]
    
    result = minimize(ellipse_residual, initial_params, args=(points,))
    return result.x

# Fit ellipse to the 2D points (after rotation)
params = fit_ellipse(points_2d)

a, b, h, k, theta = params

# Create the 2D ellipse (in the rotated plane)
t = np.linspace(0, 2 * np.pi, 100)
x_ellipse = h + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
y_ellipse = k + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)

# Step 3: Reconstruct the ellipse in 3D space
ellipse_points = np.column_stack((x_ellipse, y_ellipse, np.full_like(x_ellipse, 0)))

# Rotate the ellipse points back to 3D space
ellipse_points_3d = np.dot(rotation_matrix.T, ellipse_points.T).T + commissure_1

# Visualize the ellipse and the commissures in PyVista
plotter = pv.Plotter()

# Add the ellipse to the plot
ellipse_mesh = pv.PolyData(ellipse_points_3d)
plotter.add_mesh(ellipse_mesh, color='cyan', line_width=3)

# Add the commissure points to the plot
commissure_points = np.array([commissure_1, commissure_2, commissure_3])
commissure_mesh = pv.PolyData(commissure_points)
plotter.add_points(commissure_mesh, color='red', point_size=10)

# Show the plot
plotter.show()
