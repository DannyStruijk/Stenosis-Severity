import os
import logging
from typing import Optional
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper
from DICOMLib import DICOMUtils
from slicer import vtkMRMLMarkupsFiducialNode



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

        self.ui.reconstructButton.connect("clicked()", self.disable_annotation)

        self.initializeParameterNode()
        
        
        
    def populate_dicoms(self, dicom_folder):
        # Clear existing items in the dropdown
        self.ui.dicom_dropdown.clear()
        self.ui.dicom_dropdown.addItem("Aosstress14", "T:/Research_01/CZE-2020.67 - SAVI-AoS/AoS stress/CT/Aosstress14/DICOM/000037EC/AA4EC564/AA3B0DE6/00007EA9")
 
    
    def load_dicom_button(self):
        dicomFolderPath = self.ui.dicom_dropdown.currentData  # Get the full path stored as data
        # print(dicomFolderPath)  # Debugging to confirm the path
        loadedNodeIDs = []
        # Import DICOM folder and load the first series
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(dicomFolderPath, db)
            patientUIDs = db.patients()
            if patientUIDs:
                    for patientUID in patientUIDs:
                        # Load the first series (you can modify this to load a specific series if needed)
                        loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))   
                        print(f"Successfully loaded DICOM series for patient: {patientUIDs[0]}")                  
            else:
                print("The DICOM folder did not contain any data")
            

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
            self.ui.MarkupsPlaceWidget.setPlaceModeEnabled(False)
            self.annotationNode.RemoveObserver(self.annotationObserverTag)
            self.ui.MarkupsPlaceWidget.setEnabled(False)



            
    def clear_annotation_points(self):
        if self.annotationNode:
            numPoints = self.annotationNode.GetNumberOfControlPoints()
            for i in range(numPoints):
                self.annotationNode.RemoveNthControlPoint(0)  # Remove first control point iteratively
                # Re-enable annotation after clearing points
        self.enable_annotation()


            
    

class InterfaceLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return InterfaceParameterNode(super().getParameterNode())

    def process(self):
        # Add core processing logic here
        pass


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
