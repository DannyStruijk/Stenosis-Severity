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

### Step 1: Create reconstruction based on CT image
- Use the "Stenosis Severity" module in order to annotate the CT images and to create a reconstruction of the three cusps.
- This template 
