# Backend
MESH_FOLDER = "/workspace/obj/"
SUPPORTED_FORMATS = [".stl"]

# Frontend
MATERIAL_SIZES = [[20, 40, 60]]
VALID_ARGVS = {
    "kerfs": int,
    "projection-axis": [int, int, int],
    "velocity": float,
    "material-sizes": [[int, int, int]],
    "autoselect-material-size": bool,
    "selected-material-size": [int, int, int],
    "align-part": bool,
    "scale-to-machine": bool,
    "scale-to-material": bool,
    "slice-to-fit": bool
}

