from steputils import p21
import trimesh
import numpy as np

import step_entities as se
import step_utils as su

if __name__ == '__main__':
    STEPNAME = "Part2.stp"
    stepfile = p21.readfile(STEPNAME)

    # Store all data in by index for quick lookup
    data = {}
    for datasection in stepfile.data:
        for instance in datasection.__iter__():
            str_ = instance.__str__()
            index = int(str_[1 : str_.index("=")])
            entity = str_[su.nth_char(str_, "=", 0) + 1 : su.nth_char(str_, "(", 0)].capitalize()
            try:
                class_ = getattr(se, entity)
                data[index] = class_(str_)               
            except AttributeError:
                data[index] = str_

    # Find all ADVANCED_FACE instances and their corresponding geometry
    for value in data.values():
        if type(value) == se.Advanced_face:
            surface = data[value.face_geometry]
            try:
                surface.slice()
            except:
                print(f"Attemping to slice invalid surface: {surface}")

    bss = se.Bspline_surface_with_knots("#50=B_SPLINE_SURFACE_WITH_KNOTS('',3,3,((#11931,#11932,#11933,#11934),(#11935, #11936,#11937,#11938),(#11939,#11940,#11941,#11942),(#11943,#11944,#11945, #11946)),.UNSPECIFIED.,.F.,.F.,.F.,(4,4),(4,4),(-0.00598824276041952,1.), (0.00528748132592779,0.998395381802503),.UNSPECIFIED.);")
    print(bss.control_points_list)
    # TODO Convert step to stl for collision detection
    # STLNAME = "Part2.stl"
    # mesh = trimesh.load(STLNAME)