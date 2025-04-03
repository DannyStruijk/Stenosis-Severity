import numpy as np
import matplotlib.pyplot as plt

# Load hinge points
hinge_file = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\gui\hinge_points.txt"
hinge_points = np.loadtxt(hinge_file)

# Select three non-collinear points
p1, p2, p3 = hinge_points[:3]

# Compute two vectors on the plane
v1 = p2 - p1
v2 = p3 - p1

# Compute the normal vector
normal_vector = np.cross(v1, v2)
normal_vector /= np.linalg.norm(normal_vector)  # Normalize

# Ensure the normal vector points downward (negative Z direction)
if normal_vector[2] > 0:
    normal_vector = normal_vector

# Compute the centroid of the hinge points
centroid = np.mean(hinge_points, axis=0)

# Define a grid to visualize the plane
x_vals = np.linspace(min(hinge_points[:,0]), max(hinge_points[:,0]), 10)
y_vals = np.linspace(min(hinge_points[:,1]), max(hinge_points[:,1]), 10)
X, Y = np.meshgrid(x_vals, y_vals)

# Solve for Z using the plane equation: ax + by + cz = d
a, b, c = normal_vector
d = np.dot(normal_vector, centroid)  # Plane equation constant
Z = (d - a * X - b * Y) / c

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot hinge points
ax.scatter(hinge_points[:,0], hinge_points[:,1], hinge_points[:,2], color='red', label="Hinge Points")

# Plot the normal vector
ax.quiver(centroid[0], centroid[1], centroid[2], 
          normal_vector[0], normal_vector[1], normal_vector[2], 
          color='blue', length=10, normalize=True, label="Normal Vector (Corrected)")

print(f"Normal vector is equal to {normal_vector}")

# Plot the annular plane
ax.plot_surface(X, Y, Z, alpha=0.5, color='cyan')

# Labels and legend
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Annular Plane & Corrected Normal Vector")
ax.legend()
plt.show()
