import numpy as np
import step_utils as su
from step_utils import nth_char

class Entity:
    string = None
    index = None
    name = None

    def __init__(self, str_: str):
        self.string = str_
        self.index = int(str_[1 : str_.index("=")])
        self.name = str_[nth_char(str_, "'", 0) + 1 : nth_char(str_, "'", 1)]

class Advanced_face(Entity):
    bounds = None # TODO the docs say that this can be a list, find an example and handle that case
    face_geometry = None
    same_sense = None

    def __init__(self, str_: str):
        try:
            self.bounds = int(str_[nth_char(str_, "#", 1) + 1 : nth_char(str_, ")", 0)])
        except ValueError:
            print("Advanced_face: bounds is not an int")
        self.face_geometry = int(str_[nth_char(str_, "#", 2) + 1 : nth_char(str_, ",", 2)])
        self.same_sense = str_[nth_char(str_, ".", 0) + 1 : nth_char(str_, ".", 1)]
        if self.same_sense == "T":
            self.same_sense = True
        else:
            self.same_sense = False

class Axis2_placement_3d(Entity):
    location = None # Cartesian_point
    axis = None # Direction
    ref_direction = None # Direction

    def __init__(self, str_: str):
        Entity.__init__(self, str_)
        self.location = int(str_[nth_char(str_, "#", 1) + 1 : nth_char(str_, ",", 1)])
        self.axis = int(str_[nth_char(str_, "#", 2) + 1 : nth_char(str_, ",", 2)])
        self.ref_direction = int(str_[nth_char(str_, "#", 3) + 1 : nth_char(str_, ")", 0)])

class Bspline_curve_with_knots(Entity):
    pass

class Bspline_surface_with_knots(Entity):
    u_degree = None
    v_degree = None
    control_points_list = []
    surface_form = None
    u_closed = None
    v_closed = None
    self_intersect = None
    u_multiplicities = None
    v_multiplicities = None
    u_knots = None
    v_knots = None
    knot_spec = None

    def __init__(self, str_: str):
        Entity.__init__(self, str_)
        self.u_degree = int(str_[nth_char(str_, ",", 0) + 1 : nth_char(str_, ",", 1)])
        self.v_degree = int(str_[nth_char(str_, ",", 1) + 1 : nth_char(str_, ",", 2)])
        list_substr = str_[su.nth_substr(str_, "((", 0) + 1: su.nth_substr(str_, "))", 0) + 1]
        all_closed_brackets = [i for i, c in enumerate(list_substr) if c == ")"]
        for i in range(len(all_closed_brackets)):
            l = list_substr[nth_char(list_substr, "(", i) + 1 : nth_char(list_substr, ")", i)]
            l = l.replace("#", "")
            l = np.fromstring(l, dtype=int, sep=",")
            self.control_points_list.append(np.asarray(l))



class Cartesian_point(Entity):
    x = None
    y = None
    z = None
    xyz = None

    def __init__(self, str_: str):
        Entity.__init__(self, str_)
        self.x = float(str_[nth_char(str_, "(", 1) + 1 : nth_char(str_, ",", 1)])
        self.y = float(str_[nth_char(str_, ",", 1) + 1 : nth_char(str_, ",", 2)])
        self.z = float(str_[nth_char(str_, ",", 2) + 1 : nth_char(str_, ")", 0)])
        self.xyz = np.asarray([self.x, self.y, self.z])

class Conical_surface(Entity):
    pass

class Cylindrical_surface(Entity):
    pass

class Direction(Entity):
    x = None
    y = None
    z = None
    xyz = None

    def __init__(self, str_: str):
        Entity.__init__(self, str_)
        self.x = float(str_[nth_char(str_, "(", 1) + 1 : nth_char(str_, ",", 1)])
        self.y = float(str_[nth_char(str_, ",", 1) + 1 : nth_char(str_, ",", 2)])
        self.z = float(str_[nth_char(str_, ",", 2) + 1 : nth_char(str_, ")", 0)])
        self.xyz = np.asarray([self.x, self.y, self.z])

class Face_outer_bound(Entity):
    pass

class Plane(Entity):
    position = None # Axis2_placement_3d

    def __init__(self, str_: str):
        Entity.__init__(self, str_)
        self.position = int(str_[nth_char(str_, "#", 1) + 1 : nth_char(str_, ")", 0)])

    def slice(self):
        print(f"Slicing {self}")

class Spherical_surface(Entity):
    pass

class Surface_of_revolution(Entity):
    pass

class Toroidal_surface(Entity):
    pass