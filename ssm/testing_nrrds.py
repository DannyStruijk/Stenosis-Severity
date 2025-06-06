import nrrd
import numpy as np 

# Define the path to the NRRD file
nrrd_path = r"H:\DATA\Downloads\Segmentation.nrrd"
nrrd_path_new = r"H:\DATA\Afstuderen\3.Data\SSM\ssm_saved_data\aos14\mean_shape_voxelized_14.nrrd"

# Read the NRRD file
data, header = nrrd.read(nrrd_path_new)

# Print basic information
print("Data shape:", data.shape)
print("Header information:")
for key, value in header.items():
    print(f"{key}: {value}")

# Count how many pixels are 0 and how many are 1
unique, counts = np.unique(data, return_counts=True)
count_dict = dict(zip(unique, counts))

print("\nPixel value counts:")
for value in [0, 1]:
    print(f"Value {value}: {count_dict.get(value, 0)} pixels")