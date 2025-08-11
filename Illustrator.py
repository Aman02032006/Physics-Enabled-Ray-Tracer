import win32com.client
from win32com.client import gencache
import pythoncom
import numpy as np
import math
import os
import uuid

# Assuming these are your custom modules and classes
from Utils import *
from Tracer import Tracer
from Sources import *

def initialize_inventor():
    
    # Establishes a robust, early-bound connection to the Inventor application and creates a new assembly document.
    
    inv_app = None

    try:
        # Initialize the COM library for the current thread
        pythoncom.CoInitialize()

        # Use gencache for early binding, enabling access to Inventor's constants
        inv_prog_id = 'Inventor.Application'
        inv_guid = '{D98A091D-3A0F-4C3E-B36E-61F62068D488}'
        inv_module = gencache.EnsureModule(inv_guid, 0, 1, 0)
        
        # Connect to Inventor, launching it if not running
        inv_app = inv_module.Application(win32com.client.Dispatch(inv_prog_id))
        inv_app.Visible = True
        inv_app.SilentOperation = True

        constants = inv_module.constants
        tg = inv_app.TransientGeometry

        # Create a new assembly document
        assembly_template = inv_app.FileManager.GetTemplateFile(constants.kAssemblyDocumentObject)
        asm_doc_generic = inv_app.Documents.Add(constants.kAssemblyDocumentObject, assembly_template, True)
        
        # Cast the generic Document to a specific AssemblyDocument to access its properties
        asm_doc = win32com.client.CastTo(asm_doc_generic, 'AssemblyDocument')
        asm_def = asm_doc.ComponentDefinition
        print("New assembly created.")

        return inv_app, asm_doc, asm_def, tg, constants

    except Exception as e:
        print(f"Failed to initialize Inventor and create assembly: {e}")
        if inv_app:
            inv_app = None
        pythoncom.CoUninitialize()
        raise

def get_appearance(doc, appearance_name: str):
    # Finds an appearance asset in the document. If not found, it searches the Autodesk Appearance Library and copies it to the document.
    
    try:
        # Attempt to get the appearance from the document's local assets
        return doc.AppearanceAssets.Item(appearance_name)
    
    except:
        # If not found, search the main library
        try:
            oApp = doc.Parent
            lib = oApp.AssetLibraries.Item("Autodesk Appearance Library")
            lib_asset = lib.AppearanceAssets.Item(appearance_name)
            # Copy the asset from the library to the current document
            local_asset = lib_asset.CopyTo(doc)
            return local_asset
        except:
            print(f"Warning: Appearance '{appearance_name}' not found in document or library.")
            return None

def place_element(asm_def, tg, element):
    # Places a pre-existing component into the assembly at a specified position and orientation.
    print(f"Placing {element.name}")
    
    path = os.path.abspath(element.model_path)
    if not os.path.exists(path):
        print(f"Error: Part file not found at {path}")
        return

    position_np = np.array(element.position) * 200 # Retaining original scaling factor
    orientation_np = np.array(element.orientation)
    
    default_axis_np = np.array([0,0,1])

    # Normalize the target orientation vector
    norm_orientation = np.linalg.norm(orientation_np)
    if norm_orientation < 1e-6:
        print(f"Warning: Orientation vector for {element.model_path} has zero length. Using default.")
        orientation_np = default_axis_np
    else:
        orientation_np /= norm_orientation

    # Calculate rotation axis and angle
    rotation_axis_np = np.cross(default_axis_np, orientation_np)
    dot_product = np.dot(default_axis_np, orientation_np)
    angle = math.acos(np.clip(dot_product, -1.0, 1.0))

    # Create the Inventor transformation matrix
    base_matrix = tg.CreateMatrix() # Starts as an identity matrix [1, 2]

    # Set rotation
    if np.linalg.norm(rotation_axis_np) > 1e-6:
        axis_vector = tg.CreateVector(*rotation_axis_np)
        center_point = tg.CreatePoint(0, 0, 0)
        base_matrix.SetToRotation(angle, axis_vector, center_point)
    elif dot_product < 0: # Anti-parallel case (180-degree rotation)
        axis_vector = tg.CreateVector(0, 1, 0) # Arbitrary perpendicular axis
        center_point = tg.CreatePoint(0, 0, 0)
        base_matrix.SetToRotation(PI, axis_vector, center_point)

    # Set translation
    translation_vector = tg.CreateVector(*position_np)
    base_matrix.SetTranslation(translation_vector)

    try:
        # Add the component occurrence to the assembly [2, 4]
        asm_def.Occurrences.Add(path, base_matrix)
    except Exception as e:
        print(f"Error placing {path}: {e}")
        raise

def create_and_place_beam_segment(inv_app, asm_def, tg, constants, start_point, end_point, radius, part_name, appearance_name, temp_dir):
    
    # Creates a new cylinder part file, saves it, and places it between two points in the assembly.
    
    temp_part_path = ""
    try:
        # --- Part 1: Calculate Geometry ---
        p1_np = np.array(start_point)
        p2_np = np.array(end_point)
        axis_vector_np = p2_np - p1_np
        length = np.linalg.norm(axis_vector_np)
        if length < 1e-6:
            print("Beam segment too short—skipped.")
            return None, None

        # --- Part 2: Create and Model the Part ---
        part_template = inv_app.FileManager.GetTemplateFile(constants.kPartDocumentObject)
        part_doc_generic = inv_app.Documents.Add(constants.kPartDocumentObject, part_template, False)
        part_doc = win32com.client.CastTo(part_doc_generic, 'PartDocument')
        part_def = part_doc.ComponentDefinition

        sketch = part_def.Sketches.Add(part_def.WorkPlanes.Item(3))
        center_pt_2d = tg.CreatePoint2d(0, 0)
        sketch.SketchCircles.AddByCenterRadius(center_pt_2d, radius)
        
        profile = sketch.Profiles.AddForSolid()

        # --- FIX: Replace AddSimple with the standard Extrude Definition workflow ---
        # 1. Create an ExtrudeDefinition object
        extrude_def = part_def.Features.ExtrudeFeatures.CreateExtrudeDefinition(profile, constants.kJoinOperation)
        # 2. Set the distance and direction for the extrusion
        extrude_def.SetDistanceExtent(length, constants.kPositiveExtentDirection)
        # 3. Add the feature to the part using the definition
        part_def.Features.ExtrudeFeatures.Add(extrude_def)
        # --- End of Fix ---

        # --- Part 3: Apply Appearance and Save ---
        appearance_asset = get_appearance(part_doc, appearance_name)
        if appearance_asset:
            part_doc.ActiveAppearance = appearance_asset

        unique_id = uuid.uuid4().hex[:8]
        temp_part_filename = f"{part_name}_{unique_id}.ipt"
        temp_part_path = os.path.join(temp_dir, temp_part_filename)
        part_doc.SaveAs(temp_part_path, False)
        part_doc.Close()

        # --- Part 4: Calculate Placement Matrix ---
        default_axis_np = np.array([0,0,1])
        target_axis_np = axis_vector_np / length

        rotation_axis_np = np.cross(default_axis_np, target_axis_np)
        dot_product = np.dot(default_axis_np, target_axis_np)
        angle = math.acos(np.clip(dot_product, -1.0, 1.0))

        oMatrix = tg.CreateMatrix()

        if np.linalg.norm(rotation_axis_np) > 1e-6:
            axis_vector = tg.CreateVector(*rotation_axis_np)
            oMatrix.SetToRotation(angle, axis_vector, tg.CreatePoint(0,0,0))
        elif dot_product < 0:
            axis_vector = tg.CreateVector(1, 0, 0)
            oMatrix.SetToRotation(math.pi, axis_vector, tg.CreatePoint(0,0,0))

        translation_vector = tg.CreateVector(*start_point)
        oMatrix.SetTranslation(translation_vector)

        # --- Part 5: Place Component ---
        occurrence = asm_def.Occurrences.Add(temp_part_path, oMatrix)
        occurrence.Grounded = True
        
        # print(f"Successfully created and placed '{temp_part_filename}'.")
        return occurrence, temp_part_path

    except Exception as e:
        print(f"An error occurred in create_and_place_beam_segment: {e}")
        import traceback
        traceback.print_exc()
        if temp_part_path and os.path.exists(temp_part_path):
            try:
                if 'part_doc' in locals() and part_doc:
                    part_doc.Close(True)
                os.remove(temp_part_path)
                print(f"Cleaned up failed part file: {temp_part_path}")
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")
        return None, None

def visualize_beam(inv_app, asm_def, tg, constants, source, path, temp_dir):
    
    # Creates a visual representation of a beam path using cylinders.
    
    # Create the first segment from the source to the first interaction point
    create_and_place_beam_segment(
        inv_app, asm_def, tg, constants,
        start_point = source.position * 200,
        end_point = path[0].position * 200,
        radius = 0.5,
        part_name = "BeamSegment",
        appearance_name = "Smooth - Red",
        temp_dir = temp_dir
    )

    # Create segments for the rest of the path
    for i in range(1, len(path)):
        create_and_place_beam_segment(
            inv_app, asm_def, tg, constants,
            start_point=path[i - 1].position * 200,
            end_point=path[i].position * 200,
            radius = 0.5,
            part_name="BeamSegment",
            appearance_name="Smooth - Red",
            temp_dir=temp_dir
        )

def CreateSetupAssembly(Elements, source = None):
    # Main function to create the full assembly, place existing elements, and visualize the beam paths.

    temp_dir = os.path.join(os.path.expanduser("~"), "TempInventorParts")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        inv_app, asm_doc, asm_def, tg, constants = initialize_inventor()

        print("Placing Elements...")

        for element in Elements:
            place_element(asm_def, tg, element)
        
        if source is not None:
            # Assuming source might also be an element to be placed
            # placeElement(asm_def, tg, source)

            TraceElements = []
            for element in Elements:
                element_subparts = [*element]
                for element_subpart in  element_subparts:
                    TraceElements.append(element_subpart)
            
            tracer = Tracer(TraceElements)
            Paths = []

            # Your original tracing logic
            for _ in range(100):
                ray = Beamlet(position=source.position, direction=source.direction, amplitude=1, phase=0.0, polarization=[1.0, 0.0, 0.0], wavelength=633e-9)
                Element_Path = tracer.trace(ray)
                # print(Element_Path)
                if Element_Path not in Paths:
                    Paths.append(Element_Path)
            
            for path in Paths:
                visualize_beam(inv_app, asm_def, tg, constants, source, path, temp_dir)

        print(f"Assembly setup complete. Temporary parts are in {temp_dir}")

    except Exception as e:
        print(f"A critical error occurred in CreateSetupAssembly: {e}")
    finally:
        # Release the COM object and uninitialize the COM library
        if inv_app:
            inv_app = None
        pythoncom.CoUninitialize()
        print("Script finished and connection to Inventor closed.")
