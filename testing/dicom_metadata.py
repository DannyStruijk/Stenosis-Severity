# 21-11-2025. Now moving from the AoS data to the SAVI-stress data. The CT data of the SAVI data is way more complex
# which is the motvation for this script. This script to analyze for all the patients what fodleri s needed for my analysis
# namely the shot of the cardiac cycle of the heartt where the aortic valve is closed.
import pydicom

def read_dicomdir(dicomdir_path):
    try:
        # Read the DICOMDIR file (DICOM index file)
        dicomdir = pydicom.dcmread(dicomdir_path)

        # Extract patient, study, series, and image details from DICOMDIR
        print(f"Patient ID: {dicomdir.PatientID}")
        print(f"Patient Name: {dicomdir.PatientName}")
        print(f"Number of Studies: {len(dicomdir.root)}")

        # Loop through the root directory (each study)
        for study in dicomdir.root:
            print(f"\nStudy ID: {study.StudyID}")
            print(f"Study Description: {study.StudyDescription}")
            print(f"Study Date: {study.StudyDate}")
            
            # Loop through the series for each study
            for series in study:
                print(f"  Series ID: {series.SeriesID}")
                print(f"  Series Description: {series.SeriesDescription}")
                print(f"  Modality: {series.Modality}")
                
                # Loop through the images for each series
                for image in series:
                    print(f"    Image ID: {image.SOPInstanceUID}")
                    print(f"    Image Filename: {image.FileReference}")

    except Exception as e:
        print(f"Error reading DICOMDIR file {dicomdir_path}: {e}")

if __name__ == "__main__":
    # Path to the DICOMDIR file
    dicomdir_path = r"T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE001/DICOMDIR"

    # Read and parse the DICOMDIR file
    read_dicomdir(dicomdir_path)