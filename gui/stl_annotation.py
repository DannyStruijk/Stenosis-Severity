import numpy as np
import pyvista as pv
import os
import sys
import pandas as pd

# Add functions.py path
sys.path.append(r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\surface_reconstruction")
from functions import calc_leaflet_landmarks  # Optional

# Patient info
patient_id = 14

# File paths
stl_path = fr"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data\aos{patient_id}\ncc_reconstruction_{patient_id}.vtk"
save_txt = fr"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data\aos{patient_id}\landmarks_template_ncc_{patient_id}.txt"
save_excel = fr"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data\aos{patient_id}\landmarks_template_ncc_{patient_id}.xlsx"

# Load STL
mesh = pv.read(stl_path)

# Setup plotter
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightgray", opacity=1)

# Landmark setup
landmarks = []
labels = ["Commissure 1 (clockwise)", "Commissure 2 (clockwise)", "Center", "Hinge"]
text_actor = None  # Placeholder for text actor

# Add or update instruction text (bottom left)
def update_instruction_text(current_index):
    global text_actor
    if text_actor:
        plotter.remove_actor(text_actor)
    if current_index < len(labels):
        text = f"Click to annotate: {labels[current_index]}"
    else:
        text = "All landmarks selected."
    text_actor = plotter.add_text(text, position='lower_left', font_size=10, color='red')

# Initialize with first instruction
update_instruction_text(0)

# Point picking callback
def callback(point):
    if len(landmarks) < len(labels):
        print(f"{labels[len(landmarks)]}: {point}")
        landmarks.append(point)

        plotter.add_point_labels([point], [labels[len(landmarks)-1]],
                                 point_size=10, render_points_as_spheres=True,
                                 always_visible=True, text_color="red")
        
        update_instruction_text(len(landmarks))
        plotter.render()
    else:
        print("All landmarks selected.")

# Enable point picking and show
plotter.enable_surface_point_picking(callback, show_message=True)
plotter.show()

# Save output
if len(landmarks) == len(labels):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_txt), exist_ok=True)

    np.savetxt(save_txt, np.array(landmarks))
    print(f"Saved landmarks to: {save_txt}")

    df = pd.DataFrame(landmarks, columns=["x", "y", "z"])
    df.insert(0, "label", labels)
    df.to_excel(save_excel, index=False)
    print(f"Saved Excel file to: {save_excel}")
else:
    print("Not all landmarks were selected. Nothing saved.")
