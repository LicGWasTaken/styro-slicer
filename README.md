# Installation
## TODO
## Using github codespaces
* TODO

# Usage
    python main.py file_path --kwargs "{'argument-1': value, 'argument-2': value, etc.}"

* ***file_path*** - the path to the file containing the mesh, or the name of the file as long as it is within the *MESH_FOLDER* folder
* ***kwargs*** - keyword arguments as listed below. Put the name in **''** and make sure that the value fits the description. 

## Additional arguments (\*\*kwargs) X ==> TODO, V ==> Tested
* Arguments marked as *'req'* are mandatory.
* To save the given arguments as preferences, simply add ***'save-as-prefs': True*** as an additional argument.

***kerf*** - float X
offset in mm from the contour of the mesh. Varies with wire width and heat.&nbsp;

***projection-axis*** - array *((3), int)* X
plane normal of projection planes. e.g. *'[0, 1, 0]'*, invalid if not one the X, Y, or Z axis.&nbsp;

***velocity*** - float X
The cutting speed.&nbsp;

***material-sizes*** - array *((3,), int)*
Available material size(s). e.g. *'[100, 40, 60], [200, 400, 700]'*.&nbsp;

***machine-size*** - array *((3), int)* X
Machine size for scaling. e.g. *'[100, 40, 60]'*.&nbsp;

***mesh-alignment*** - array*((3, 3, 3), int)*
How to align the mesh. Expects an array defining 3 axes e.g. [[1, 0, 0], [0, 0, 1], [0, 1, 0]] that will be aligned with the x,y,z axes respectively. The above example aligns the y axis to the z and viceversa. Sets *align-part* to True

***selected-material-size*** - array *((3), int)*
The chosen material size, passed as an array. Otherwise selects the smallest possible size automatically.&nbsp;

***align-part*** - bool
Align part to vertical axis.&nbsp;

***scale-to-machine*** - bool X 
Scale the part down to fit within the machine boundaries.&nbsp;

***scale-to-material*** - bool X
Scale the part down to fit within the available material sizes.&nbsp;

***slice-to-fit*** - bool X
Weather to generate multiple files with sliced sections of the mesh that fit within the available material sizes.&nbsp;

### Slicing processes (exclusive)

***as-convex-hull*** - bool X
Convert the contour of the mesh to its convex hull. Leads to more accurate and time-efficient results.

***as-projection*** - bool X
Project the mesh to a plane. Faster computationally.

