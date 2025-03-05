# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:29:06 2025

@author: u840707
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define the control points
# control_points = [[[2, 0, 1], [2.342365839407433, 1.556026136348903, 1], [1, 2, 1]], 
#  [[2, 0, 1], [2.342365839407433, 1.556026136348903, 0.5], [1, 2, 1]], 
#  [[2, 0, 1], [2.342365839407433, 1.56, 0], [1, 2, 1]]]

# Convert to numpy array for easier manipulation
control_points = np.array(control_points)

# Plot the control points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, z coordinates
x = control_points[:,:,0]
y = control_points[:,:,1]
z = control_points[:,:,2]

# Plot the points
ax.scatter(x, y, z, color='r', label='Control Points')

# Connect the control points with lines
for i in range(len(control_points)-1):
    ax.plot([x[i][0], x[i+1][0]], [y[i][0], y[i+1][0]], [z[i][0], z[i+1][0]], color='b')
    ax.plot([x[i][1], x[i+1][1]], [y[i][1], y[i+1][1]], [z[i][1], z[i+1][1]], color='b')
    ax.plot([x[i][2], x[i+1][2]], [y[i][2], y[i+1][2]], [z[i][2], z[i+1][2]], color='b')

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Control Points in 3D Space')

# Show the plot
plt.show()
