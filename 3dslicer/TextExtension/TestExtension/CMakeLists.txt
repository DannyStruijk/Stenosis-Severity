cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(TestExtension)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/TestExtension")
set(EXTENSION_CONTRIBUTORS "Danny (R&D Cardiology Catharian Ziekenhuis)")
set(EXTENSION_DESCRIPTION "This is a test script")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/TestExtension.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/TestExtension/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Interface)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
