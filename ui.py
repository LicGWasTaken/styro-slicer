# pyuic5 -o Gui.py Gui.ui
print('Programm startet!')

import locale
locale.setlocale(locale.LC_NUMERIC, '')

import trimesh
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget
import sys
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from qtui import Ui_MainWindow
import slicer
import utils as u

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()  
        
        # Set up the user interface from Designer.
        self.setupUi(self)
        self.InFillValue = 0.0

        # Set up VTK rendering
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1, 1, 1)
        self.w_vtk_viewer.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.w_vtk_viewer.GetRenderWindow().GetInteractor()
        self.iren.Initialize()
        
        # Connect functions to buttons
        self.b_open_file.clicked.connect(self.load_file)
        self.b_slice.clicked.connect(self.compute_coordinates)

    file_path = None
    file_name = None
    active_actors = {}

    def load_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*)')
        
        if not self.file_path:
            print('No file selected')
            self.t_selected_file.setText("None")
        else:
            print(f'Selected file: {self.file_path}')
            self.file_name = self.file_path.rsplit('/', 1)[1]
            self.t_selected_file.setText(self.file_name)
            self.render_stl(self.file_path)
            # TODO check for valid file extension

    def compute_coordinates(self):
        if self.file_path is None:
            print("No File Path")
            #TODO show this in UI, maybe highlight button
        else:
            out = slicer.main(self.file_path, self.file_name)
            if out.size != 0:
                # self.render_pcd(out)
                self.render_lcd(out)

    def render_actor(self, tag, actor: vtk.vtkActor):
        # Colored axes
        self.axes = vtk.vtkAxesActor()
        bounds = actor.GetBounds()
        self.axes.SetTotalLength(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        self.axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1, 0, 0)
        self.axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 1, 0)
        self.axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 1)
        if "axes" in self.active_actors:
            self.renderer.RemoveActor(self.active_actors["axes"])
        self.renderer.AddActor(self.axes)
        self.active_actors["axes"] = self.axes

        # Passed actor argument
        if tag in self.active_actors:
            self.renderer.RemoveActor(self.active_actors[tag])
        self.renderer.AddActor(actor)
        self.active_actors[tag] = actor

        # Render
        self.renderer.ResetCamera()
        self.w_vtk_viewer.GetRenderWindow().Render()

    def render_stl(self, file_):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_)

        # Translate the mesh to the world origin
        mesh = trimesh.load_mesh(file_)
        to_origin, extents = u.axis_oriented_extents(mesh)
        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, to_origin[i, j])

        transform = vtk.vtkTransform()
        transform.SetMatrix(vtk_matrix)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(reader.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        # Render
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(transform_filter.GetOutput())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.render_actor("mesh", actor)

    def render_pcd(self, arr):
        vtk_points = vtk.vtkPoints()
        for p in arr:
            vtk_points.InsertNextPoint(p)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)

        vertices = vtk.vtkCellArray()
        for i in range(len(arr)):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)
        polydata.SetVerts(vertices)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)
        actor.GetProperty().SetPointSize(5)

        self.render_actor("pcd", actor)
    
    def render_lcd(self, arr):
        vtk_points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        for line_points in arr:
            point1, point2 = line_points
            point_id1 = vtk_points.InsertNextPoint(point1)
            point_id2 = vtk_points.InsertNextPoint(point2)

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, point_id1)
            line.GetPointIds().SetId(1, point_id2)

            lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)
        actor.GetProperty().SetLineWidth(3)

        self.render_actor("lines", actor)

if __name__ == '__main__':
    print("running...")
    APP = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(APP.exec_())
