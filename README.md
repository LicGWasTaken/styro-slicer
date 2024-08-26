# Installation
## TODO
## Using github codespaces
* go to *https://github.com/codespaces* and create a new codespace
* TODO

# Usage
If you haven't already, start a docker container:

    docker build -t image-name .  
    docker run -it --rm image-name

Whithin bash, run: 

    python styro-slicer.py MESH_PATH **kwargs

For files within the *'obj'* folder, the file name without the path suffices, e.g. *'cube.stl'* instad of *'/workspace/obj/cube.stl'*

## Additional arguments (\*\*kwargs)
***offset*** - offset in mm from the contour of the mesh. Varies with wire width and heat.&nbsp;

***mat-size*** - available material size(s). Expects *((3,), int)*, e.g. *'[100, 40, 60]'*. When passing multiple sizes, separate them with kommas, e.g. *'([100, 40, 60], [20, 30, 40])'*.&nbsp;