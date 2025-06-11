import numpy as np
import preprocessing_meshes as prep
import reconstruction_to_template as rec
import registration_pipeline as reg  # assuming your wrapper function is here
import average_shape_to_pointcloud as asp  # new import for voxelization and saving as NRRD


def main():
    # The meshes are first simplified, i.e. the amount of vertices is reduced to make registration possible
    print("Step 1: Preprocessing meshes...")

    # Define STL paths here (change as needed)
    stl_paths = [
        r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos13\cusps\ncc\ncc_trimmed_smoothed.stl",
        r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos14\cusps\ncc\ncc_trimmed.stl",
        r"H:\DATA\Afstuderen\3.Data\SSM\patient_database\aos15\cusps\ncc\ncc_trimmed.stl"
    ]

    # Define output folder here (change this for each run)
    output_folder = r"H:\DATA\Afstuderen\3.Data\SSM\non-coronary\output_patients\aos14"
    output_folder_meshes = output_folder + r"\simplified_meshes"
    prep.preprocess_default_meshes(stl_paths, output_folder_meshes)

    # Creating a pointcloud from the reconstruction
    # The reconstruction is based on the annotated aortic leaflets in 3Dslicer
    print("Step 2: Sampling and thickening points from VTK surface...")
    vtk_file = r"H:\DATA\Afstuderen\3.Data\SSM\non-coronary\input_patients\aos14\ncc_reconstruction_14.vtk"
    plane_normal = np.array([0, 0, 1])

    # Specify output path here (change this for each run)
    output_path = output_folder + r"\thickened_points_run1.ply"
    rec.sample_and_thicken_points(
        vtk_file_path=vtk_file,
        plane_normal=plane_normal,
        distance=0.5,
        thickness=2.5,
        num_samples=5,
        save_path=output_path
    )

    # Now run registration pipeline using the thickened points and other data
    print("Step 3: Running registration pipeline...")

    patient_ids = [13, 14, 15]
    base_path = r"H:\DATA\Afstuderen\3.Data\SSM"
    reconstruction_landmarks_path = r"H:\DATA\Afstuderen\3.Data\SSM\non-coronary\input_patients\aos14\landmarks_template_ncc_14.txt"
    mean_shape_output_path = output_folder + r"\mean_shape_reconstruction_14.ply"

    # Set CPD smoothness parameter alpha (adjust this to control deformation smoothness)
    alpha_value = 100

    reg.run_registration_pipeline(
        patient_ids=patient_ids,
        base_path=base_path,
        reconstruction_path=output_path,
        reconstruction_landmarks_path=reconstruction_landmarks_path,
        output_path=mean_shape_output_path,
        alpha=alpha_value
    )

    # Step 4: Convert mean shape point cloud to voxel labelmap and save as NRRD
    print("Step 4: Voxelizing and saving to NRRD...")
    pointcloud_path = mean_shape_output_path
    output_nrrd_path = output_folder + r"\mean_shape_voxelized_14.nrrd"
    dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"

    asp.convert_average_shape_to_nrrd(pointcloud_path, dicom_dir, output_nrrd_path)


if __name__ == "__main__":
    main()
