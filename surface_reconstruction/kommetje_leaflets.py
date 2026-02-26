import vtk
import numpy as np
import functions

# het doeleinde van dit scripts is om de onderste delen van de aortic wall en de leaflets boundaries
# met elkaar te verbinden. Dit vormt de 'kommetjes' van de aortic leaflets. 

# %%  LOADING THE DATA

# For testing purposes, the LCC cusp is first calculated.
lcc_landmarks = np.load(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity-backup\temp\lcc_landmarks.npy")
lcc_wall = np.load(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity-backup\temp\lcc_wall.npy")

rcc_wall = np.load(r"H:\DATA\Afstuderen\3.Data\output_valve_segmentation\rcc_wall.npy")
rcc_landmarks = np.load(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity-backup\temp\rcc_landmarks.npy")

rcc_lcc_slice = np.load(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity-backup\temp\rcc_lcc_boundary_slice.npy")
lcc_ncc_slice = np.load(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity-backup\temp\lcc_ncc_boundary_slice.npy")
ncc_rcc_slice = np.load(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity-backup\temp\ncc_rcc_boundary_slice.npy")


lcc_com1 = lcc_landmarks[0]
lcc_com2 = lcc_landmarks[1]
lcc_hinge = lcc_landmarks[2]

rcc_com1 = rcc_landmarks[0]
rcc_com2 = rcc_landmarks[1]
rcc_hinge = rcc_landmarks[2]

# Now we are going to use the center height hard coded. Should be extracted from the main script.
center_height = 120
lcc_com1[0] = center_height
lcc_com2[0] = center_height

rcc_com1[0] = center_height
rcc_com2[0] = center_height

# Convert both masks
lcc_ncc_points = functions.mask_to_pointcloud(lcc_ncc_slice, center_height)
rcc_lcc_points = functions.mask_to_pointcloud(rcc_lcc_slice, center_height)
ncc_rcc_points = functions.mask_to_pointcloud(ncc_rcc_slice, center_height)



# Fit a spline through the points so that is less jagged and more smooth object
lcc_ncc_points = functions.fit_spline(lcc_ncc_points, smoothing = 50)
rcc_lcc_points = functions.fit_spline(rcc_lcc_points, smoothing = 50)
ncc_rcc_points = functions.fit_spline(ncc_rcc_points, smoothing = 50)

rcc_lcc_points = rcc_lcc_points[::-1]


# %% EXTRACTING THE CURVE OF COMMISSURES TO HINGE, ATTACHMENT REGION OF LEAFLET TO THE AORTIC WALL

from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev

def get_leaflet_attachment_curve(aortic_wall_mask, commissure1, hinge, commissure2, n_points=20, snap_to_wall=True):
    """
    Returns a smooth curve for the aortic leaflet attachment.

    Parameters:
        aortic_wall_mask : 3D bool array
            The segmented aortic wall.
        commissure1, hinge, commissure2 : tuple of (z,y,x)
            Landmarks defining the leaflet attachment.
        n_points : int
            Number of points to sample along the smooth curve.
        snap_to_wall : bool
            Whether to snap the curve to the nearest wall voxels.

    Returns:
        smooth_curve : (n_points, 3) array
            Smooth curve passing through landmarks.
        curve_on_wall : (n_points, 3) array
            Smooth curve snapped to nearest wall voxels (same as smooth_curve if snap_to_wall=False)
    """

    # --- Step 1: extract wall surface voxels ---
    wall_surface = aortic_wall_mask & ~binary_erosion(aortic_wall_mask)
    wall_coords = np.array(np.where(wall_surface)).T  # shape (N_voxels, 3)

    # --- Step 2: snap landmarks to nearest wall voxel ---
    tree = cKDTree(wall_coords)
    snapped_landmarks = []
    for lm in [commissure1, hinge, commissure2]:
        _, idx = tree.query(lm)
        snapped_landmarks.append(tuple(wall_coords[idx]))
    comm1, hinge, comm2 = snapped_landmarks

    # --- Step 3: fit smooth spline through the three landmarks ---
    landmarks_array = np.array([comm1, hinge, comm2])
    coords = landmarks_array.T  # shape (3,3)
    tck, _ = splprep(coords, s=0, k =2)
    u_new = np.linspace(0, 1, n_points)
    smooth_curve = np.array(splev(u_new, tck)).T  # (n_points, 3)

    # --- Step 4: optionally snap curve back to wall ---
    if snap_to_wall:
        _, idx = tree.query(smooth_curve)
        curve_on_wall = wall_coords[idx]
    else:
        curve_on_wall = smooth_curve.copy()

    return smooth_curve, curve_on_wall


smooth_curve, curve_on_wall = get_leaflet_attachment_curve(
    lcc_wall,           # your 3D mask
    lcc_landmarks[0],   # commissure1
    lcc_landmarks[2],   # hinge
    lcc_landmarks[1],   # commissure2
    n_points=20,
    snap_to_wall=True
)

print("Smooth curve shape:", smooth_curve.shape)
print("Snapped curve shape:", curve_on_wall.shape)


# %% VISUALIZATION

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# smooth_curve: continuous spline (n_points, 3)
# curve_on_wall: snapped curve on wall voxels
# wall_coords: wall surface voxels (N,3)
# landmarks: commissure1, hinge, commissure2

wall_surface = lcc_wall & ~binary_erosion(lcc_wall)
wall_coords = np.array(np.where(wall_surface)).T  # shape (N_voxels, 3)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# 1️⃣ Wall surface voxels (light gray, semi-transparent)
ax.scatter(wall_coords[:,2], wall_coords[:,1], wall_coords[:,0],
           color='orange', alpha=0.4, s=1)  # x,y,z order for plotting

# 2️⃣ Landmarks (blue)
landmarks_array = np.array([lcc_landmarks[0], lcc_landmarks[2], lcc_landmarks[1]])  # comm1, hinge, comm2
ax.scatter(landmarks_array[:,2], landmarks_array[:,1], landmarks_array[:,0],
            color='blue', s=50, label='Landmarks')

# 3️⃣ Smooth attachment curve (red line)
ax.plot(smooth_curve[:,2], smooth_curve[:,1], smooth_curve[:,0],
        color='red', linewidth=2, label='Smooth curve')

# 4️⃣ Snapped curve (green dots)
ax.scatter(curve_on_wall[:,2], curve_on_wall[:,1], curve_on_wall[:,0],
           color='green', s=20, label='Curve on wall')

# # Also try to display the lower side of the boundaries
ax.scatter(lcc_ncc_points[:,2], lcc_ncc_points[:,1], lcc_ncc_points[:,0])
ax.scatter(rcc_lcc_points[:,2], rcc_lcc_points[:,1], rcc_lcc_points[:,0])
# ax.scatter(ncc_rcc_points[:,2], ncc_rcc_points[:,1], ncc_rcc_points[:,0])



# Labels and settings
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Leaflet Attachment Curve on Aortic Wall')
ax.legend()
# Invert Z axis
ax.invert_zaxis()

plt.show()
plt.show()

# %% --------------- CONNECT THE BOTTOM CAP COORDINATES -----------------------

from geomdl import fitting, BSpline

# Make NURBS surface fit through all of the found points


# Suppose your curves are in a list
curves = [lcc_ncc_points, rcc_lcc_points, curve_on_wall]
print(np.linalg.norm(lcc_ncc_points[-1] - rcc_lcc_points[0]))
print(np.linalg.norm(rcc_lcc_points[-1] - curve_on_wall[0]))
print(np.linalg.norm(curve_on_wall[-1] - lcc_ncc_points[0]))

# Create a 2D array of points (rows = curves, cols = points along each curve)
surf_points = np.array(curves)  # shape (num_curves, num_points_per_curve, 3)

# Flatten for interpolation
surf_flat = []
for u in range(len(curves)):
    for v in range(len(curves[0])):
        surf_flat.append(surf_points[u][v])
surf_flat_tuples = [tuple(pt) for pt in surf_flat]

# Interpolated surface
size_u=len(curves)
size_v=len(curves[0])
surf = fitting.interpolate_surface(
    surf_flat_tuples,
    size_u=size_u,
    size_v=size_v,
    degree_u=2,
    degree_v=3,
    ctrlpts_size_u=3  # explicitly matches number of curves
)


surf.delta = 0.01   # sampling resolution
eval_pts = np.array(surf.evalpts)  # sampled surface points

# Make sure your surface has been evaluated

surf.evaluate()

functions.save_surface_evalpts(surf, filename="lcc_leaflet_surface")

# %% MAKE STL VOLUME

from scipy.ndimage import gaussian_filter, binary_closing
from skimage.morphology import binary_dilation, cube

volume_shape = (340, 512, 512)
closing_structure = cube(3) 


mask_3d = functions.create_3d_mask_from_points(eval_pts, volume_shape, thickness_voxels=1)

num_true = np.sum(mask_3d)
print(num_true)
mask_3d = binary_closing(mask_3d, structure=closing_structure)
mask_3d = binary_dilation(mask_3d, closing_structure)
mask_3d_smooth = gaussian_filter(mask_3d.astype(np.float32), sigma = 1.5)

pixel_spacing = (0.4, 0.35, 0.35)

dicom_origin = (-61, -248, 1126.8)

# Step 3: save as STL in patient space
functions.save_volume_as_stl_patient_space(
    volume=mask_3d_smooth,
    output_path=r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity-backup\temp",
    patient_nr="savi_01",
    file_type="lcc_leaflet_surface",
    zoom_x=pixel_spacing[2],
    zoom_y=pixel_spacing[1],
    zoom_z=pixel_spacing[0],
    dicom_origin=dicom_origin
)

#%% CHECKING

import matplotlib.pyplot as plt
import numpy as np

def plot_curve_2d(curve, label, color):
    x = curve[:, 2]   # X
    y = curve[:, 1]   # Y

    plt.plot(x, y, color=color, linewidth=2, label=label)
    plt.scatter(x[0], y[0], color=color, marker='o', s=80)   # start
    plt.scatter(x[-1], y[-1], color=color, marker='x', s=80) # end

# --- Plot ---
plt.figure(figsize=(6,6))

plot_curve_2d(lcc_ncc_points, "lcc_ncc", "blue")
plot_curve_2d(rcc_lcc_points, "rcc_lcc", "red")
plot_curve_2d(curve_on_wall, "wall_curve", "green")

plt.gca().set_aspect('equal', adjustable='box')
plt.title("2D Projection of Curves (XY plane)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)

plt.show()
