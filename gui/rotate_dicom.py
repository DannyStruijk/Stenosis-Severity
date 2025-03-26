# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:46:29 2025

@author: u840707
"""
import pydicom

# Path to the DICOM file (replace this with your file path)
#dicom_path = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity\dicoms\dicom_viewer_0002\0002.DCM"
dicom_path = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00000BBF\EE0AE34C"
# Read the DICOM file
dcm_data = pydicom.dcmread(dicom_path)

# Access the Image Orientation (Patient)
orientation = dcm_data.ImageOrientationPatient

# Print the orientation
print("Image Orientation (Patient):", orientation)