import os
import numpy as np
import preprocessing_meshes as prep
import reconstruction_to_template as rec
import registration_pipeline as reg
import average_shape_to_pointcloud as asp


def main():
    # Define the target cusp (change this to 'ncc', 'lcc', etc. as needed)
    target_cusp = "lcc"  # ← Change this to switch quickly!

    print(f"Running pipeline for: {target_cusp.upper()}")

    # Define paths
    patient_ids = [13, 14, 15]
    base_path = r"H:\DATA\Afstuderen\3.Data\SSM"
    patient_folder = "aos14"

    stl_paths = [
        os.path.join(base_path, "patient_database", f"aos{pid}", "cusps", target_cusp, f"{target_cusp}_trimmed.stl")
        for pid in patient_ids
    ]
    # stl_paths[0] = stl_paths[0].replace("trimmed.stl", "trimmed_smoothed.stl")  # handle exception for aos13

    output_folder = os.path.join(base_path, target_cusp, "output_patients", patient_folder)
    output_folder_meshes = os.path.join(output_folder, "simplified_meshes")
    prep.preprocess_default_meshes(stl_paths, output_folder_meshes)


    # Step 2: Sample and thicken
    print("Step 2: Sampling and thickening points from VTK surface...")
    vtk_file = os.path.join(base_path, target_cusp, "input_patients", patient_folder, f"reconstructed_{target_cusp}.vtk")
    plane_normal = np.array([0, 0, 1])
    output_path = os.path.join(output_folder, f"{target_cusp}_thickened_points_run1.ply")

    rec.sample_and_thicken_points(
        vtk_file_path=vtk_file,
        plane_normal=plane_normal,
        distance=0.5,
        thickness=2.5,
        num_samples=5,
        save_path=output_path
    )


    # Step 3: Registration
    print("Step 3: Running registration pipeline...")
    reconstruction_landmarks_path = os.path.join(base_path,"patient_database", patient_folder, "landmarks", f"{target_cusp}_template_landmarks.txt")
    mean_shape_output_path = os.path.join(output_folder, f"{target_cusp}_mean_shape_reconstruction_14.ply")
    alpha_value = 100

    reg.run_registration_pipeline(
        patient_ids=patient_ids,
        base_path=base_path,
        reconstruction_path=output_path,
        reconstruction_landmarks_path=reconstruction_landmarks_path,
        output_path=mean_shape_output_path,
        alpha=alpha_value
    )


    # Step 4: Convert to NRRD
    print("Step 4: Voxelizing and saving to NRRD...")
    pointcloud_path = mean_shape_output_path
    output_nrrd_path = os.path.join(output_folder, f"{target_cusp}_mean_shape_voxelized_14.nrrd")
    dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"

    asp.convert_average_shape_to_nrrd(pointcloud_path, dicom_dir, output_nrrd_path)


if __name__ == "__main__":
    main()
