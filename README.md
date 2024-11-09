# Installation
## TODO
## Using github codespaces
* TODO

# Usage
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus in tellus velit. Ut lobortis, turpis sed viverra pellentesque, eros orci imperdiet diam, a iaculis mauris sem vitae lacus. Vivamus varius bibendum purus, a lobortis nunc dapibus eu. Suspendisse est enim, molestie nec justo in, molestie fringilla felis. Curabitur ornare mauris a nibh cursus, vel elementum ligula imperdiet. Sed egestas lectus eros, a posuere sapien iaculis sit amet. Aliquam pharetra ipsum id lacinia dapibus. Quisque ac ante lorem. Suspendisse fermentum libero urna, at blandit odio mollis eleifend. Aenean sit amet tristique ipsum.

## Additional arguments (\*\*kwargs)
* Boolean command line arguments can just be inputted as *command-line-argument* instead of *command-line-argument=True*.   
* Arguments marked as *'req'* are mandatory.
* To save the given arguments as preferences, simply add ***save-as-prefs*** as an additional argument.  

***kerf*** - int
offset in mm from the contour of the mesh. Varies with wire width and heat.&nbsp;

***projection-axis*** - array *((3), int)*  
plane normal of projection planes. e.g. *'[0, 1, 0]'*, invalid if not one the X, Y, or Z axis.&nbsp;

***velocity*** - float  
The cutting speed.&nbsp;

***material-sizes*** - array *((3,), int)*
available material size(s). e.g. *'[100, 40, 60], [200, 400, 700]'*.&nbsp;

***autoselect-material-size*** - bool - def True  
Automatically selects the smallest possible material size.&nbsp;

***selected-material-size*** - array *((3), int)* or int
The chosen material size, passed as an array or as an index of *material-sizes*. Automatically sets autoselect-material-size to False.&nbsp;

***align-part*** - bool - def True  
Align part to vertical axis.&nbsp;

***scale-to-machine*** - bool - def False
Scale the part down to fit within the machine boundaries.&nbsp;

***scale-to-material*** - bool - def False
Scale the part down to fit within the available material sizes. Overrides *scale-to-material*.&nbsp;

***slice-to-fit*** - bool - def False  
Weather to generate multiple files with sliced sections of the mesh that fit within the available material sizes.&nbsp;

