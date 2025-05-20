import os
import trimesh
import pyfqmr

#%%%%%%%%%%%%%%%%%% LOADING THE DATA

# List of your STL files
stl_paths = [
    r"H:\DATA\Afstuderen\3.Data\SSM\aos13\cusps\ncc_trimmed_smoothed.stl",
    r"H:\DATA\Afstuderen\3.Data\SSM\aos14\cusps\ncc_trimmed.stl",
    r"H:\DATA\Afstuderen\3.Data\SSM\aos15\cusps\ncc_trimmed.stl"
]

# Target number of vertices
target_vertices = 2000

#%%%%%%%%%%%%%%%%% PROCESSING AND SIMPLIFYING THE MESHES

for i, path in enumerate(stl_paths, start=13):  # 13, 14, 15 for aos13-15
    print(f"\nProcessing STL for aos{i}...")

    # Load mesh
    mesh = trimesh.load_mesh(path)

    print("Original Mesh:")
    print(f"  Number of vertices: {len(mesh.vertices)}")
    print(f"  Number of faces: {len(mesh.faces)}")
    print(f"  Bounding box extents: {mesh.bounds}")
    print()

    # Simplify using pyfqmr
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
    mesh_simplifier.simplify_mesh(
        target_count=target_vertices,
        aggressiveness=7,
        preserve_border=True,
        verbose=True
    )
    vertices, faces, normals = mesh_simplifier.getMesh()

    # Reconstruct the simplified mesh
    simplified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    #%%%%%%%%%%%%%%%%% INSPECT SIMPLIFIED MESH

    print("Simplified Mesh:")
    print(f"  Number of vertices: {len(simplified_mesh.vertices)}")
    print(f"  Number of faces: {len(simplified_mesh.faces)}")
    print(f"  Bounding box extents: {simplified_mesh.bounds}")
    print()

    #%%%%%%%%%%%%%%%%% SAVING THE SIMPLIFIED MESH

    output_dir = os.path.dirname(path)
    output_filename = f"ncc_simplified_mesh_{i}.stl"
    output_path = os.path.join(output_dir, output_filename)

    simplified_mesh.export(output_path)

    print(f"Simplified mesh saved to: {output_path}")
