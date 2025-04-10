import pydicom

def read_pixel_spacing(dicom_file_path):
    try:
        # Load the DICOM file
        dicom_data = pydicom.dcmread(dicom_file_path)
        
        # Check if PixelSpacing attribute exists
        if 'PixelSpacing' in dicom_data:
            pixel_spacing = dicom_data.PixelSpacing
            print(f"Pixel Spacing: {pixel_spacing}")
            return pixel_spacing
        else:
            print("Pixel Spacing attribute not found in the DICOM file.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
dicom_file_path = "path/to/your/dicom/file.dcm"
dicom_dir = r"T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9"
read_pixel_spacing(dicom_dir)  # testing