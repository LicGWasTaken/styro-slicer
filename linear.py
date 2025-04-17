# Compute convex mesh (?)
    # Not currently sure on weather this renders the remaining algorithm useless, so maybe try without it first
# Detect each face and corrisponding vertices and normal
    # Trimesh.faces
    # Trimesh.face_normals
    # Trimesh.vertices
# Merge adjacent with same normal
    # Trimesh.facets
    # Trimesh.facets_normal
# Find the longest vertical line on each face
    # Find facet origin
        # Trimesh.facet_origin
    # Plane intersection between facet and rotated vertical plane
    # make it start at the lowest and end at the highest vertex along the line on the facet
# Collision detection
    # raycast perpendicular to the points (either vertices or interpolated)
# If it hits try collision with vertical angle
    # Rotate the line by theta around normal vector (also do negative angles)
    # Make sure the current angle is achievable within machine space
    # For valid lines -> collision detection
    # Exit when no collision or angle reaches positive and negative boundary
# If no valid cut is found, iteratively repeat with planes further and further out from the center
    # See current raycasting implementation
    # Store the location of the plane to later color in the mesh for user feedback
    # Forseeable issues with angled facets never returning valid coordinates
# Check if the cut can be achieved with wire angle
    # Define 0deg and 90deg planes
    # Rotate the mesh/point accordingly
    # Put a 2d line (vertical rotation doesn't matter here) through the facet origin
    # Check for both 0 and 90 deg planes to find the smallest angle from the plane
    # Check if the motor coordinates are valid for that rotation
# For invalid coordinates calculate the minimum rotation offset necessary from 0 or 90 deg
    # Rotate line by offset around mesh origin until coordinates are valid
    # Probably doesn't need to be done iteratively
# Sort by rotation
    # if all are 0 or 90 just sort them and exit
    # Sort them in groups of 5deg/10deg
    # Recalculate coordinates for that rotation
    # Forseeable issues (maybe) with recalculations returning invalid coordinates
# Sort by motor coordinates 
    # Take each rotation group
    # define a starting point
    # Take offsets to p1 and p2
    # sort by the maximum value of those offsets
    # Take into account the ability to cut the other way oround as in from top to bottom istead of bottom to top
        # Compute the distance to each point indipendendly
        # Once one point is selected, automatically append the other end of the cut
# Forseeable issues with movement between points cutting into the mesh

# NEW
# Try each angle around the plane's normal with collision detection
# Store each valid cut
# Check which angles lie within the machine's boundaries with 0deg rotation
# If a valid angle is found, save it and continue
# Otherwise retry the process with rotational steps (like 5 to 10 deg)
# If no valid cut can be achieved, store that information too (and color the meshe's plane in)