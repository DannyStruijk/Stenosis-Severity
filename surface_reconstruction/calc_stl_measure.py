import os
import trimesh
import pandas as pd

base_folder = r"H:\DATA\Afstuderen\3.Data\output_valve_segmentation"

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

for i in range(29,30):
    patient_id = f"CZE{i:03d}"
    
    folder = os.path.join(
        base_folder,
        patient_id,
        "patient_space",
        "calc_volumes"
    )
    
    # 🔥 Skip if patient folder does not exist
    if not os.path.exists(folder):
        print(f"Skipping {patient_id} (folder not found)")
        continue

    print(f"Processing {patient_id}...")

    results = []

    for name in stl_names:
        file_name = f"{patient_id}_{name}.stl"
        file_path = os.path.join(folder, file_name)
        
        if os.path.exists(file_path):
            mesh = trimesh.load(file_path)
            
            if not mesh.is_watertight:
                print(f"Warning: {file_name} is not watertight!")
            
            volume = mesh.volume
            
            results.append({
                "Structure": name,
                "Volume (mm^3)": volume
            })
        else:
            print(f"Missing file: {file_name}")
            results.append({
                "Structure": name,
                "Volume (mm^3)": None
            })

    df = pd.DataFrame(results)

    output_path = os.path.join(
        folder,
        f"{patient_id}_calcification_volumes_stls.xlsx"
    )
    
    df.to_excel(output_path, index=False)

    print(f"Saved to: {output_path}")