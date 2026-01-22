import functions
import functions
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import nrrd
from skimage import exposure
from scipy.ndimage import binary_erosion

# Goal of the script is to use the output of valve_segmentation_19_01_orig_resol.py and to create one smooth object
# this object should be smooth for visualization and also it should not contain any of the wholes


# %% Import the existing STLs. The code is first tried on existing outputs 

input_path = r"H:\DATA\Afstuderen\3.Data\output_valve_segmentation\savi_01\patient_space"

# For now the code is tried for only the aortic walls 
lcc_wall = rf"{input_path}\savi_01_LCC_com_to_com.stl"
ncc_wall = rf"{input_path}\savi_01_NCC_com_to_com.stl"
rcc_wall = rf"{input_path}\savi_01_RCC_com_to_com.stl"

import trimesh

mesh_lcc = trimesh.load(lcc_wall)
mesh_ncc = trimesh.load(ncc_wall)
mesh_rcc = trimesh.load(rcc_wall)

mesh_lcc.show()  # opens interactive viewer
