import os
import pandas as pd

base_folder = r"H:\DATA\Afstuderen\3.Data\output_valve_segmentation"
master_path = r"H:\DATA\Afstuderen\3.Data\patient_data\patient_data.xlsx"

patients_to_process = [f"CZE{i:03d}" for i in range(38, 39)]

# Load master file
df_master = pd.read_excel(master_path)
df_master["PatientNr"] = df_master["PatientNr"].astype(str).str.strip()

for patient_id in patients_to_process:
    print(f"\nProcessing {patient_id}...")

    patient_folder = os.path.join(
        base_folder,
        patient_id,
        "patient_space",
        "calc_volumes"
    )

    if not os.path.exists(patient_folder):
        print(f"Skipping {patient_id} (folder not found)")
        continue

    # Ensure patient exists in master
    if not (df_master["PatientNr"] == patient_id).any():
        print(f"{patient_id}: not found in master file")
        continue

    # ----------------------------
    # 1. Read calc summary volumes
    # ----------------------------
    summary_path = os.path.join(patient_folder, f"{patient_id}_calc_summary.xlsx")

    if os.path.exists(summary_path):
        df_summary = pd.read_excel(summary_path)
        df_summary.columns = df_summary.columns.str.strip()

        required_cols = ["Segment Name", "Volume (mm^3)"]
        if all(col in df_summary.columns for col in required_cols):
            for _, row in df_summary.iterrows():
                segment_name = str(row["Segment Name"]).strip()
                volume = row["Volume (mm^3)"]

                if isinstance(volume, str):
                    volume = float(volume.replace(",", "."))

                if segment_name not in df_master.columns:
                    df_master[segment_name] = None
                    print(f"Created volume column: {segment_name}")

                df_master.loc[df_master["PatientNr"] == patient_id, segment_name] = volume
        else:
            print(f"{patient_id}: calc summary missing required columns")
    else:
        print(f"{patient_id}: calc summary file not found")

    # ----------------------------
    # 2. Read region areas
    # ----------------------------
    region_areas_path = os.path.join(patient_folder, f"{patient_id}_region_areas.xlsx")

    if os.path.exists(region_areas_path):
        df_areas = pd.read_excel(region_areas_path)
        df_areas.columns = df_areas.columns.str.strip()

        if df_areas.empty:
            print(f"{patient_id}: region areas file is empty")
        else:
            area_row = df_areas.iloc[0]

            for col in df_areas.columns:
                if col.lower() == "patient":
                    continue

                value = area_row[col]

                if isinstance(value, str):
                    value = float(value.replace(",", "."))

                if col not in df_master.columns:
                    df_master[col] = None
                    print(f"Created area column: {col}")

                df_master.loc[df_master["PatientNr"] == patient_id, col] = value
    else:
        print(f"{patient_id}: region areas file not found")


# Save updated master file
output_path = master_path.replace(".xlsx", "_with_volumes_and_areas.xlsx")
df_master.to_excel(output_path, index=False)

print(f"\nUpdated file saved to: {output_path}")