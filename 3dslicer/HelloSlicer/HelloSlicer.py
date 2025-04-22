# Practice code to make a moduel in 3DSlicer

import os
import slicer
import qt
import ctk
import slicer.ScriptedLoadableModule

class HelloSlicer:
    def __init__(self, parent):
        parent.title = "HelloSlicer"
        parent.categories = ["Examples"]
        parent.dependencies = []
        parent.contributors = ["You"]
        parent.helpText = "Say Hello from Slicer"
        parent.acknowledgementText = "Thanks to you!"
        self.parent = parent

class HelloSlicerWidget:
    def __init__(self, parent=None):
        self.parent = parent
        self.layout = parent.layout() if parent else None
        self.number = 2
        self.setup()

    def setup(self):
        # Add a button
        self.helloButton = qt.QPushButton("Say Hello")
        self.layout.addWidget(self.helloButton)
        
        # 
        self.testbutton = qt.QPushButton("Say testing again")
        self.layout.addWidget(self.testbutton)

        # Connect to action
        self.helloButton.connect('clicked(bool)', self.sayHello)
        self.testbutton.connect('clicked(bool)', self.calculate)

    def sayHello(self):
        print("👋 Hello from the Python console in Slicer!")
        
    def calculate(self):
        print("The square of {self.number} is: ", self.number*self.number)
