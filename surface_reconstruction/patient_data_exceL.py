import os
import trimesh
import pandas as pd

base_folder = r"H:\DATA\Afstuderen\3.Data\output_valve_segmentation"
master_path = r"H:\DATA\Afstuderen\3.Data\output_valve_segmentation\patient_data.xlsx"

stl_names = [
    "LCC_calc",
    "RCC_calc",
    "NCC_calc",
    "calc_volume",
    "central_NCC_calc",
    "peripheral_NCC_calc",
    "central_LCC_calc",
    "peripheral_LCC_calc",
    "central_RCC_calc",
    "peripheral_RCC_calc"
]

# 🔥 Select patients here
patients_to_process = [f"CZE{i:03d}" for i in range(1, 31)]  # all patients again

# 🔥 Load master file
df_master = pd.read_excel(master_path)
df_master["PatientNr"] = df_master["PatientNr"].astype(str).str.strip()

for patient_id in patients_to_process:
    
    folder = os.path.join(
        base_folder,
        patient_id,
        "patient_space",
        "calc_volumes"
    )
    
    if not os.path.exists(folder):
        print(f"Skipping {patient_id} (folder not found)")
        continue

    print(f"Processing {patient_id}...")

    volumes = {}

    for name in stl_names:
        file_name = f"{patient_id}_{name}.stl"
        file_path = os.path.join(folder, file_name)

        if os.path.exists(file_path):
            mesh = trimesh.load(file_path)
            
            if not mesh.is_watertight:
                print(f"Warning: {file_name} not watertight")

            volumes[name] = mesh.volume
        else:
            print(f"Missing file: {file_name}")
            volumes[name] = None

    # 🔥 Assign values to the correct patient row
    for key, value in volumes.items():
        df_master.loc[df_master["PatientNr"] == patient_id, key] = value


# 🔥 Save updated master file
output_path = master_path.replace(".xlsx", "_with_volumes.xlsx")
df_master.to_excel(output_path, index=False)

print(f"Updated file saved to: {output_path}")