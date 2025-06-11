import os
import re
import trimesh
import pyfqmr
from typing import List

def load_mesh(path: str) -> trimesh.Trimesh:
    """Load a mesh from a file path and print basic stats."""
    mesh = trimesh.load_mesh(path)
    print("Original Mesh:")
    print(f"  Number of vertices: {len(mesh.vertices)}")
    print(f"  Number of faces: {len(mesh.faces)}")
    print(f"  Bounding box extents: {mesh.bounds}\n")
    return mesh

def simplify_mesh(mesh: trimesh.Trimesh, target_vertices: int) -> trimesh.Trimesh:
    """Simplify a mesh to a target number of vertices using pyfqmr."""
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
    mesh_simplifier.simplify_mesh(
        target_count=target_vertices,
        aggressiveness=7,
        preserve_border=True,
        verbose=True
    )
    vertices, faces, _ = mesh_simplifier.getMesh()
    simplified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    print("Simplified Mesh:")
    print(f"  Number of vertices: {len(simplified_mesh.vertices)}")
    print(f"  Number of faces: {len(simplified_mesh.faces)}")
    print(f"  Bounding box extents: {simplified_mesh.bounds}\n")
    return simplified_mesh

def extract_subject_id(path: str) -> str:
    """Extract a subject ID like 'aos13' from the file path."""
    match = re.search(r'aos\d+', path.lower())
    return match.group(0) if match else 'unknown'

def process_and_save_mesh(path: str, target_vertices: int, output_dir: str, output_name: str) -> None:
    """Process a single mesh and save the simplified version in output_dir."""
    mesh = load_mesh(path)
    simplified = simplify_mesh(mesh, target_vertices)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    
    simplified.export(output_path)
    print(f"Simplified mesh saved to: {output_path}\n")

def process_all_meshes(stl_paths: List[str], target_vertices: int, output_dir: str) -> None:
    """Process a list of STL files using subject IDs and save in output_dir."""
    for path in stl_paths:
        subject_id = extract_subject_id(path)
        print(f"\nProcessing STL for {subject_id}...")
        output_filename = f"ncc_simplified_mesh_{subject_id}.stl"
        process_and_save_mesh(path, target_vertices, output_dir, output_filename)

def preprocess_default_meshes(stl_paths: List[str], output_dir: str):
    """Wrapper function to preprocess given STL paths and save to output_dir."""
    target_vertices = 2000
    process_all_meshes(stl_paths, target_vertices, output_dir)
