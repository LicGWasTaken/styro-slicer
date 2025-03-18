# Backend
MESH_FOLDER = "/workspace/obj/"
OUTPUT_FOLDER = "/workspace/output/"
JSON_FOLDER = "/workspace/"
SUPPORTED_FORMATS = [".stl"]
NUMPY_DECIMALS = 5
MOTOR_AXIS = [1, 0, 0]

DEFAULT_SETTINGS = {
    "kerf": 0.5,
    "projection-axis": [0, 1, 0],
    "velocity": 100,
    "material-sizes": [[1000, 1000, 2000]],
    "machine-size": [1000, 1000, 1000],
    "mesh-alignment": None,
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
    "kerf": float,
    "projection-axis": [int, int, int],
    "velocity": int,
    "material-sizes": [[int, int, int]],
    "machine-size": [int, int, int],
    "mesh-alignment": [[int, int, int], [int, int, int], [int, int, int]],
    "selected-material-size": [int, int, int],
    "align-part": bool,
    "scale-to-machine": bool,
    "scale-to-material": bool,
    "slice-to-fit": bool,
    "as-convex-hull": bool,
    "as-projection": bool
}

