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
