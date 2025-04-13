import vtk

# Step 1: Load a Mesh File
reader = vtk.vtkSTLReader()
reader.SetFileName("cat.stl")
reader.Update()
mesh = reader.GetOutput()

# Step 2: Sample a Point Cloud from the Mesh
point_sampler = vtk.vtkPolyDataPointSampler()
point_sampler.SetInputData(mesh)
point_sampler.SetDistance(1)  # Adjust sampling density
point_sampler.Update()
point_cloud = point_sampler.GetOutput()
print(type(point_cloud))

# # Step 3: Reconstruct Mesh from Point Cloud
# surf_recon = vtk.vtkSurfaceReconstructionFilter()
# surf_recon.SetInputData(point_cloud)
# surf_recon.Update()

# # Step 4: Convert Implicit Surface to PolyData
# contour_filter = vtk.vtkContourFilter()
# contour_filter.SetInputConnection(surf_recon.GetOutputPort())
# contour_filter.SetValue(0, 0.0)  # Adjust for best results
# contour_filter.Update()
# reconstructed_mesh = contour_filter.GetOutput()

# # Step 5: Save the Reconstructed Mesh
# writer = vtk.vtkPLYWriter()
# writer.SetFileName("output_mesh.ply")
# writer.SetInputData(reconstructed_mesh)
# writer.Write()

# Convert the point cloud into visible points
vertex_filter = vtk.vtkVertexGlyphFilter()
vertex_filter.SetInputData(reader.GetOutput())
vertex_filter.Update()

# Step 6: Visualize the Results
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(vertex_filter.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.1, 0.1)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

render_window.Render()
interactor.Start()




def trimesh_to_vtk(tri_mesh):
    # Create a VTK polydata object
    vtk_mesh = vtk.vtkPolyData()

    # Convert vertices to VTK format
    points = vtk.vtkPoints()
    vertices = np.array(tri_mesh.vertices)
    for v in vertices:
        points.InsertNextPoint(v.tolist())
    vtk_mesh.SetPoints(points)

    # Convert faces to VTK format
    faces = np.array(tri_mesh.faces, dtype=np.int64)
    cells = vtk.vtkCellArray()
    for f in faces:
        cell = vtk.vtkTriangle()
        for i in range(3):
            cell.GetPointIds().SetId(i, f[i])
        cells.InsertNextCell(cell)
    vtk_mesh.SetPolys(cells)

    return vtk_mesh

def convex_mesh(mesh_: trimesh.Trimesh, num_sub_pcds: int):
    # Subdivide the mesh into submeshes along the z axis and compute their convex hulls
    convex_sub_meshes = [[] for _ in range(num_sub_pcds)]
    for i in range(num_sub_pcds):
        # Slice from below
        plane_origin = [0, 0, -extents_[2] / 2 + (extents_[2] / num_sub_pcds) * i * 0.999]
        tmp = trimesh.intersections.slice_mesh_plane(
            mesh_, plane_normal=[0, 0, 1], plane_origin=plane_origin
        )

        # Slice from above
        plane_origin = [0, 0, -extents_[2] / 2 + (extents_[2] / num_sub_pcds) * (i + 1) * 1.001]
        tmp = trimesh.intersections.slice_mesh_plane(
            tmp, plane_normal=[0, 0, -1], plane_origin=plane_origin
        )

        # Remove top and bottom of convex hull
        tmp = tmp.convex_hull
        if i > 0:
            plane_origin = [0, 0, -extents_[2] / 2 + (extents_[2] / num_sub_pcds) * i]
            tmp = trimesh.intersections.slice_mesh_plane(
                tmp, plane_normal=[0, 0, 1], plane_origin=plane_origin
            )
        if i < num_sub_pcds - 1:
            plane_origin = [0, 0, -extents_[2] / 2 + (extents_[2] / num_sub_pcds) * (i + 1)]
            tmp = trimesh.intersections.slice_mesh_plane(
                tmp, plane_normal=[0, 0, -1], plane_origin=plane_origin
            )

        convex_sub_meshes[i] = tmp

    # Sample and merge sub_pcds
    append_filter = vtk.vtkAppendPolyData()

    for sub_mesh in convex_sub_meshes:
        # The sampler needs to be reinitialized every time
        sampler = vtk.vtkPolyDataPointSampler() 
        vtk_mesh = trimesh_to_vtk(sub_mesh)
        sampler.SetInputData(vtk_mesh)
        sampler.SetDistance(5)  # Adjust sampling density
        sampler.Update()

        sub_pcd = sampler.GetOutput()
        append_filter.AddInputData(sub_pcd)
        append_filter.Update()

    append_filter.Update()
    pcd = append_filter.GetOutput()
    ps = pcd.GetPoints()
    out = np.array([ps.GetPoint(i) for i in range(ps.GetNumberOfPoints())])

    # quadric = vtk.vtkQuadricClustering()
    # quadric.SetInputData(pcd)
    # resolution = 100
    # quadric.SetNumberOfXDivisions(resolution)
    # quadric.SetNumberOfYDivisions(resolution)
    # quadric.SetNumberOfZDivisions(resolution)
    # quadric.Update()

    # reduced_pcd = quadric.GetOutput()
    # ps = reduced_pcd.GetPoints()

    ps = pcd.GetPoints()
    out = np.array([ps.GetPoint(i) for i in range(ps.GetNumberOfPoints())])

    # Remesh
    surf_recon = vtk.vtkSurfaceReconstructionFilter()
    # surf_recon.SetInputData(reduced_pcd)
    surf_recon.SetInputData(pcd)
    surf_recon.Update()

    contour_filter = vtk.vtkContourFilter()
    contour_filter.SetInputConnection(surf_recon.GetOutputPort())
    contour_filter.SetValue(0, 0.0) 
    contour_filter.Update()
    remeshed = contour_filter.GetOutput()

    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName("convex_mesh.stl")
    
    # Ensure the mesh is properly set
    stl_writer.SetInputData(remeshed)
    stl_writer.Write()

    return out