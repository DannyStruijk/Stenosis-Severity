# Predicting the Severity of Aortic Valve Stenosis Based on Revised Calcification Score

**Master Thesis Title:** Automatic Reconstruction of the Aortic Valves based on Landmark Annotated CT Image and Predicting the Severity of Aortic Valve Stenosis Based on Revised Calcification Score  
  
**Author:** D.M. Struijk (1452479)  
**Supervisors:**  Marcel van 't Veer, Pim Tonino and Marcel Breeuwer

*Eindhoven University of Technology  
Catharina Ziekenhuis*  

## Research Aim
The aim of this research is to predict the stress aorta valve index (SAVI), which is a measure of aorta stenosis severity, based on CT images. This way, the assessment of aortic valve stenosis can move towards non-invasive measures. This research includes the implementation of a semi-automatic tool which allows the clinician to annotate landmarks of the aortic valves. Using these landmarks, a reconstruction of the aortic valves is automatically made. Using the reconstruction and the observed calcification of the aortic valve, a new calcium score is calculated. The calcium score is a revision of the Agatston score, where a certain weight is given to a calcium cluster based on the location of the cluster on the aortic valve.

The code in this repository is a step-for-step guide to create the semi-automatic implementation and showcases how the necessary code predicts the SAVI. 

*UPDATE 11-06-2025*: Currently, the step-for-step guide shows how to prepare the data for the statistical shape model (SSM).

## Statistical Shape Model

### Step 0: Input - Create database for the SSM
- Use the exisisting segmentations of the AoS Stress study and import these into 3DSlicer.
- Annotatate the landmarks and use these as guides in order to slice the segmentation into three different cusps.
- If needed, smooth the cusps when there are gaps/inconsistencies present with a gaussian filter of width 0.2 mm. 

### Step 1: Input - Create reconstruction based on CT image
- Use the "Stenosis Severity" module in order to annotate the CT images and to create a reconstruction of the three cusps.
- Example output path for patient 14, ncc: "H:\DATA\Afstuderen\3.Data\SSM\non-coronary\input_patients\aos14"
- In addition, the annotated landmarks should be saved in the same folder.

### Step 2: Run the pipeline
When all of the preprocessed data is ready, you are able to run the pipeline. The pipeline is, currently, only able to run for one cusp. The pipeline uses a main file which calls the following function:

preprocessing_meshes.py - Loads the meshes from the database (the trimmed leaflets) and converts them into a pointcloud. Hyperparameter target_vertices dictate the resolution which is used.
    - Simplified meshes where the amount of vertices is reduced. Output path: "H:\DATA\Afstuderen\3.Data\SSM\non-coronary\output_patients\{patient}\simplified_meshes"
  
- reconstruction_to_template.py - It converts the reconstructed leaflets into a pointcloud and gives a certain thickness (given by hyperparameter "thickness") to the reconstruction. Output is the template for the SSM.
    - Reconstruction converted into a pointcloud. Output path: "H:\DATA\Afstuderen\3.Data\SSM\non-coronary\output_patients\{patient}\thickened_points_run1.ply"
  
- registration_pipeline.py - Registration steps are contained in this code. First, the trimmed leaflets from the database are rigidly registered to the template based on landmark annotation. Then, non-rigid registration, based on coherent point drift (CPD) registers the template pointcloud to the trimmed leaflets, which result in an amount of template pointclouds similar to the amount of trimmed leaflets used in the SSM. An average pointcloud is created on the basis of these deformed template pointclouds.
    - Mean shape pointcloud. Output path: "H:\DATA\Afstuderen\3.Data\SSM\non-coronary\output_patients\{patient}\mean_shape_reconstruction_14.ply"
  
- average_shape_to_pointcloud.py - The output of the registration_pipeline.py, the average shape, is converted into a voxelized binary 3D volume, which is in the same space as the CT image of the corresponding patients.
    - NRRD segmentation. Output path: "H:\DATA\Afstuderen\3.Data\SSM\non-coronary\output_patients\{patient}\mean_shape_voxelized_14.nrrd"
  
