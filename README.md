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
- preprocessing_meshes.py - Loads the meshes from the database (the trimmed leaflets) and converts them into a pointcloud. Hyperparameter target_vertices dictate the resolution which is used. 
- reconstruction_to_template.py - It converts the reconstructed leaflets into a pointcloud and gives a certain thickness (given by hyperparameter "thickness") to the reconstruction.
- 
- 
