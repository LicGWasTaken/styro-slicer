# Backend
MESH_FOLDER = "/workspace/obj/"
JSON_FOLDER = "/workspace/"
SUPPORTED_FORMATS = [".stl"]

DEFAULT_SETTINGS = {
    "kerfs": 0.5,
    "projection-axis": [0, 1, 0],
    "velocity": 10,
    "material-sizes": [[1000, 1000, 2000]],
    "autoselect-material-size": False,
    "selected-material-size": None,
    "align-part": True,
    "scale-to-machine": False,
    "scale-to-material": False,
    "slice-to-fit": False,
    "as-convex-hull": True,
    "as-projection": False
}

# Frontend
VALID_ARGVS = {
    "save-as-prefs": bool,
    "kerfs": float,
    "projection-axis": [int, int, int],
    "velocity": float,
    "material-sizes": [[int, int, int]],
    "autoselect-material-size": bool,
    "selected-material-size": [int, int, int],
    "align-part": bool,
    "scale-to-machine": bool,
    "scale-to-material": bool,
    "slice-to-fit": bool,
    "as-convex-hull": bool,
    "as-projection": bool
}

