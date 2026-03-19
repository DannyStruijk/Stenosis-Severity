#%%%%%%%% IMPORTING MODULES AND INITIALIZING METADATA

import os
from typing import Optional
import vtk
import subprocess
import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper
from DICOMLib import DICOMUtils
import numpy as np
import sys

functions_path = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity-backup\surface_reconstruction"

if functions_path not in sys.path:
    sys.path.insert(0, functions_path)

import functions


class Interface(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Stenosis Severity")
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []
        self.parent.contributors = ["Your Name (Your Organization)"]
        self.parent.helpText = _("""This module does something amazing.""")
        self.parent.acknowledgementText = _("""Thanks to contributors and funding sources.""")

@parameterNodeWrapper
class InterfaceParameterNode:
    inputVolume: slicer.vtkMRMLScalarVolumeNode
    pass  # Add your parameters here as needed

#%%%%%%%%%%%%% INTIALIZING THE INTERFACE

class InterfaceWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.IsAnnotating = False
        self.landmarkIndex = 0
        self.annotationObserverTag = None
        self.annotationNode = None
        self.annotationStage = "commissures"
        self.numPoints = 0
        self.segmentation_node = None
    
    def setup(self):
        # Mandatory initiation for 3DSlicer
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Interface.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Required observers and logic integration to make the interface working. 
        self.logic = InterfaceLogic()
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        
        # Connect the annotation enable/disable buttons
        self.ui.enable_annotation_button.connect("clicked()", self.enable_annotation)
        self.ui.disable_annotation_button.connect("clicked()", self.disable_annotation)

        
        # Buttons which allow the user to select DICOM and load in volume
        self.populate_dicoms("T:\Research_01\CZE-2020.67 - SAVI-AoS\AoS stress\CT\Aosstress14\DICOM\000037EC\AA4EC564\AA3B0DE6\00007EA9")
        self.ui.load_dicom_button.connect("clicked()", self.load_dicom_button)

        # Connectnig the buttons which allow for reconstruction
        self.ui.deleteButton.connect("clicked()", self.clear_annotation_points)
        self.ui.reconstructButton.connect("clicked()", self.reconstruct)
        self.ui.importButton.connect("clicked()", self.import_reconstruction)
               
        # Initializing
        self.initializeParameterNode()
        
    def populate_dicoms(self, dicom_folder):
        # Clear existing items in the dropdown
        self.ui.dicom_dropdown.clear()
        
        # Add SAVI-AoS patients
        self.ui.dicom_dropdown.addItem("CZE001", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE001/DICOM/00003852/AA44D04F/AA7BB8C5/000050B5")
        self.ui.dicom_dropdown.addItem("CZE002", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE002/DICOM/0000AFC5/AAAA2796/AAFF16B0/0000ADA6")
        self.ui.dicom_dropdown.addItem("CZE003", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE003/DICOM/0000AF6A/AA4272CE/AA72A45E/000050F2")
        self.ui.dicom_dropdown.addItem("CZE004", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE004/DICOM/00002F76/AA1F4542/AAB1E4E9/0000CAC5")
        self.ui.dicom_dropdown.addItem("CZE005", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE005/DICOM/0000AF52/AA590C3F/AAC428CF/0000FEE0")
        self.ui.dicom_dropdown.addItem("CZE006", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE006/DICOM/00000EED/AA87381C/AAAEC035/00002581")
        self.ui.dicom_dropdown.addItem("CZE007", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE007/DICOM/000065F6/AAB95BAE/AA7A7E4C/00005896")
        self.ui.dicom_dropdown.addItem("CZE008", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE008/DICOM/000053DF/AA102722/AA7E9491/000073C6")
        self.ui.dicom_dropdown.addItem("CZE010", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE010/DICOM/00003911/AA8B6291/AA8D4457/0000EDE8")
        self.ui.dicom_dropdown.addItem("CZE011", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE011/DICOM/00005CB5/AAAA99F9/AA6670D7/0000949B")
        self.ui.dicom_dropdown.addItem("CZE012", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE012/DICOM/00003CC2/AA022842/AA289D50/00007754")
        self.ui.dicom_dropdown.addItem("CZE013", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE013/DICOM/0000EA91/AACEC60C/AA7E3964/0000E5BC")
        self.ui.dicom_dropdown.addItem("CZE014", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE014/DICOM/00005E34/AA79D3B8/AA0DFFCE/0000C950")
        self.ui.dicom_dropdown.addItem("CZE015", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE015/2de versie/DICOM/00003971/AA8D3DE7/AA73252D/0000D7A8")
        self.ui.dicom_dropdown.addItem("CZE016", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE016/DICOM/00002765/AAECDDFB/AAE12657/00001053")
        self.ui.dicom_dropdown.addItem("CZE017", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE017/DICOM/00009314/AA9EE817/AAD307C6/000000A9")
        self.ui.dicom_dropdown.addItem("CZE018", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE018/DICOM/0000767B/AAC5650C/AADB5D54/0000CAEC")
        self.ui.dicom_dropdown.addItem("CZE019", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE019/DICOM/00008A57/AA9CD7F4/AAA7440B/00006E56")
        self.ui.dicom_dropdown.addItem("CZE020", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE020/DICOM/000057A3/AA56DBD9/AA4B0694/00005E78")
        self.ui.dicom_dropdown.addItem("CZE022", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE022/DICOM/00002E36/AA0DF707/AAE959BA/0000B1D3")
        self.ui.dicom_dropdown.addItem("CZE023", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE023/DICOM/0000B09F/AA9FDA4D/AA8D0F36/0000F21F")
        self.ui.dicom_dropdown.addItem("CZE024", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE024/DICOM/0000431B/AA1DAD4A/AAAA0072/0000E6A7")
        self.ui.dicom_dropdown.addItem("CZE025", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE025/DICOM/0000A32D/AAF725DA/AA6A556D/000058A9")
        self.ui.dicom_dropdown.addItem("CZE026", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE026/DICOM/00007858/AACE1771/AA3C074D/000033A6")
        self.ui.dicom_dropdown.addItem("CZE027", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE027/DICOM/000046F3/AA0B28CE/AA933D3E/00007339")
        self.ui.dicom_dropdown.addItem("CZE029", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE029/DICOM/00001B91/AADD797F/AA3A0E6E/00002DA1")
        self.ui.dicom_dropdown.addItem("CZE030", "T:/Research_01/CZE-2020.67 - SAVI-AoS/SAVI-AoS/CZE030/DICOM/000032F8/AAD2875F/AA7CD947/00002E0D")
        
        
    def load_dicom_button(self):
        # Get the patient name and DICOM path from dropdown
        patientName = self.ui.dicom_dropdown.currentText
        dicomFolderPath = self.ui.dicom_dropdown.currentData  # <-- need () here, it's a method
    
        print(f"Loading DICOM data for {patientName} from {dicomFolderPath}")
    
        loadedNodeIDs = []
        # Import DICOM folder and load the first series
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(dicomFolderPath, db)
            patientUIDs = db.patients()
            if patientUIDs:
                for patientUID in patientUIDs:
                    # Load the first series (you can modify this to load a specific series if needed)
                    loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))   
                print(f"Successfully loaded DICOM series for {patientName}")
            else:
                print(f"The DICOM folder for {patientName} did not contain any data. Path:  {dicomFolderPath}")
            
    #%%%%%%%%%% STANDARD 3DSLICER FUNCTIONS
    def cleanup(self):
        self.removeObservers()

    def enter(self):
        self.initializeParameterNode()

    def exit(self):
        self.clear_annotation_points()
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event):
        self.clear_annotation_points()
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        self.clear_annotation_points()
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[InterfaceParameterNode]):
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None):
        # Optional: Enable/disable buttons depending on node states
        pass

    def onApplyButton(self):
        with slicer.util.tryWithErrorDisplay(_("Processing failed."), waitCursor=True):
            # self.logic.process(...)
            pass
        
    #%%%%% FUNCTIONS WRITTEN BY DANNY
    def enable_annotation(self):
        self.landmarkIndex = 0
        self.create_annotation_node()
        

        if self.numPoints < 4:
            self.ui.annotation_label.setText("Annotate the commissures")
        elif self.numPoints == 4:
            self.ui.annotation_label.setText("Now annotate the center")
        elif self.numPoints < 8:
            self.ui.annotation_label.setText("Now annotate the hinge points")
        elif self.numPoints == 8:
            self.ui.annotation_label.setText("Done annotating. Reconstruct.")
        self.ui.annotation_label.repaint()
        self.activate_placement_widget()
 

    def disable_annotation(self):
        self.isAnnotating = False
        self.ui.annotation_label.setText("Annotation: Disabled")
        self.ui.MarkupsPlaceWidget.setMRMLScene(None)  # Disconnect from MRML scene
        self.ui.MarkupsPlaceWidget.setPlaceModeEnabled(False)  # Ensure place mode is disabled
        
    def activate_placement_widget(self):
        self.ui.MarkupsPlaceWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.MarkupsPlaceWidget.setCurrentNode(self.annotationNode)

    def create_annotation_node(self):
        if not self.annotationNode:
            self.annotationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Commissure")
            self.annotationObserverTag = self.annotationNode.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointAddedEvent,
                self.onAnnotationPointAdded
            )
      
    def onClick(self, caller, event):
        return
        
    def onAnnotationPointAdded(self, caller, event):
        print(self.annotationNode.GetNumberOfControlPoints())
        self.numPoints = self.annotationNode.GetNumberOfControlPoints()
    
        # Assign names based on the number of points
        if self.numPoints < 4:
            # The first four points are "Commissure"
            self.annotationNode.SetNthControlPointLabel(self.numPoints - 1, f"Commissure {self.numPoints}")
        elif self.numPoints == 4:
            # The fifth point is "Center"
            self.annotationNode.SetNthControlPointLabel(self.numPoints - 1, "Center")
        elif self.numPoints < 8:
            # Points 6 and 7 are "Hinge"
            self.annotationNode.SetNthControlPointLabel(self.numPoints - 1, f"Hinge Point {self.numPoints - 4}")
        elif self.numPoints == 8:
            # Done annotating
            self.ui.annotation_label.setText("Done annotating. Reconstruct.")
            self.disable_annotation()
            
    def clear_annotation_points(self):
        if self.annotationNode:
            numPoints = self.annotationNode.GetNumberOfControlPoints()
            for i in range(numPoints):
                self.annotationNode.RemoveNthControlPoint(0)  # Remove first control point iteratively
        self.disable_annotation()
        
    def reconstruct(self):

        # Get selected patient
        patient_id = self.ui.dicom_dropdown.currentText
        
        # Set annotations folder and patient-specific files
        annotations_folder = r"H:\DATA\Afstuderen\2.Code\Stenosis-Severity-backup\annotations"
        os.makedirs(annotations_folder, exist_ok=True)
        ras_file = os.path.join(annotations_folder, f"{patient_id}_ras_coordinates.txt")
    
        # Check if annotation node exists
        if not self.annotationNode or self.numPoints == 0:
            print("No points annotated yet!")
            return
    
        # ---- SAVE ONLY VALID POINTS (ignore unplaced placeholders) ----
        landmarks_to_save = []
        num_control_points = self.annotationNode.GetNumberOfControlPoints()  # <-- get it directly
        
        for index in range(num_control_points):
            point_Ras = [0, 0, 0]
            self.annotationNode.GetNthControlPointPositionWorld(index, point_Ras)
            if point_Ras == [0, 0, 0]:
                continue  # skip unplaced placeholder points
            point_lps = self.transform_to_lps(point_Ras)
            landmarks_to_save.append(point_lps)
    
        if len(landmarks_to_save) != 7:
            print(f"Error: exactly 7 points required, got {len(landmarks_to_save)}")
            return
    
        # Save the RAS/LPS landmarks
        with open(ras_file, 'w') as f:
            for p in landmarks_to_save:
                f.write(f"{p[0]}\t{p[1]}\t{p[2]}\n")
    
        print(f"RAS/LPS landmarks saved to {ras_file}")
    
        # ---- PROCESS LANDMARKS ----
        landmarks = np.array(landmarks_to_save)
        commissure_1, commissure_2, commissure_3, center, hinge_1, hinge_2, hinge_3 = landmarks
    
        # Calculate cusp landmarks
        cusp_landmarks = functions.calc_leaflet_landmarks(
            commissure_1,
            commissure_2,
            commissure_3,
            hinge_1,
            hinge_2,
            hinge_3
        )
        print("CUSP LANDMARKS", cusp_landmarks)
    
        # ---- SAVE PROCESSED LANDMARKS ----
        # 1️⃣ Patient database
        db_output_path = os.path.join(r"H:\DATA\Afstuderen\3.Data\SSM\patient_database", patient_id, "landmarks")
        os.makedirs(db_output_path, exist_ok=True)
        functions.save_ordered_landmarks(cusp_landmarks, center, db_output_path)
        print(f"Ordered landmarks saved in patient database: {db_output_path}")
    
        # 2️⃣ Also save a copy in the annotations folder
        annotations_output_file = os.path.join(annotations_folder, f"{patient_id}_ordered_landmarks.txt")
        functions.save_ordered_landmarks(cusp_landmarks, center, annotations_output_file)
        print(f"Ordered landmarks also saved in annotations folder: {annotations_output_file}")
    
    def transform_to_lps(self, point_Ras):
        """
        Convert RAS coordinates to LPS by inverting the x and y axes.
        
        Parameters:
            point_Ras: A list of RAS coordinates [x, y, z]
        
        Returns:
            point_Lps: A list of LPS coordinates [x, y, z]
        """
        # Invert the x and y coordinates for RAS to LPS conversion
        point_lps = [-point_Ras[0], -point_Ras[1], point_Ras[2]]
        return point_lps

    def import_reconstruction(self):
        
        # CHANGE THIS PIECE OF CODE FOR WHAT TO IMPORT
        vtk_files = [
            r"H:\DATA\Afstuderen\3.Data\SSM\lcc\input_patients\aos14\reconstructed_lcc.vtk",
            r"H:\DATA\Afstuderen\3.Data\SSM\ncc\input_patients\aos14\reconstructed_ncc.vtk",
            r"H:\DATA\Afstuderen\3.Data\SSM\rcc\input_patients\aos14\reconstructed_rcc.vtk"
        ]
    
        # Create one segmentation node
        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
    
        # Import each model into the segmentation node as a new segment
        for i, vtk_file in enumerate(vtk_files, 1):
            model_node = slicer.modules.models.logic().AddModel(vtk_file)
            model_node.SetName(f"ReconstructedLeaflet_{i}")
            slicer.modules.segmentations.logic().ImportModelToSegmentationNode(model_node, self.segmentation_node)
    
        # Set segmentation node in the table view UI
        self.ui.SegmentsTableView.setSegmentationNode(self.segmentation_node)
        self.ui.SegmentationDisplayNodeWidget.setSegmentationNode(self.segmentation_node)
        # After changing the display properties, ensure the segmentation node is updated
        self.segmentation_node.GetDisplayNode().Modified()
        return





            
#%%%%% INTERFACE LOGIC
class InterfaceLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return InterfaceParameterNode(super().getParameterNode())

    def process(self):
        # Add core processing logic here
        pass

#%%%% INTERFACE TEST
class InterfaceTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_Interface1()

    def test_Interface1(self):
        self.delayDisplay("Starting the test")
        # Implement basic testing of logic here
        self.delayDisplay("Test passed")
