# pyuic5 -o Gui.py Gui.ui
print('Programm startet!')

import locale
locale.setlocale(locale.LC_NUMERIC, '')

import json
import numpy as np
import trimesh
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox
import sys
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from qtui import Ui_MainWindow
import gcode as gc
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
        self.renderer.SetBackground(255/255, 200/255, 255/255)
        self.w_vtk_viewer.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.w_vtk_viewer.GetRenderWindow().GetInteractor()
        self.iren.Initialize()
        
        # Connect functions to buttons
        self.b_open_file.clicked.connect(self.load_file)
        self.b_slice.clicked.connect(self.compute_coordinates)

        # Set default parameter values
        with open("preferences.json", "r") as file:
            prefs = json.load(file)

        spins = self.scrollArea.widget().findChildren(QSpinBox)
        dblspins = self.scrollArea.widget().findChildren(QDoubleSpinBox)
        combos = self.scrollArea.widget().findChildren(QComboBox)
        checks = self.scrollArea.widget().findChildren(QCheckBox)
        for key in prefs.keys():
            val = prefs[key]
            if isinstance(val, bool):
                for check in checks:
                    if key in check.objectName():
                        check.setChecked(val)
            elif isinstance(val, int):
                for spin in spins:
                    if key in spin.objectName():
                        spin.setValue(val)
                for dblspin in dblspins:
                    if key in dblspin.objectName():
                        dblspin.setValue(val)
            elif isinstance(val, float):
                for dblspin in dblspins:
                    if key in dblspin.objectName():
                        dblspin.setValue(val)
            elif isinstance(val, str):
                for combo in combos:
                    if key in combo.objectName():
                        combo.setCurrentText(val)
            else:
                u.msg("Invalid value in preferences.json", "error")
                return
    
    file_path = None
    file_name = None
    active_actors = {}

    mesh = None
    extents = None

    def load_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*)')
        
        if not self.file_path:
            u.msg("No file selected", "warning")
            self.t_selected_file.setText("None")
        else:
            u.msg(f"Selected file: {self.file_path}", "debug")
            self.file_name = self.file_path.rsplit("/", 1)[1]
            self.t_selected_file.setText(self.file_name)
            self.render_stl(self.file_path)
            # TODO check for valid file extension

    def compute_coordinates(self):
        if self.file_path is None:
            # TODO show this in UI, maybe highlight button
            u.msg("Missing file path", "warning")
            return

        self.mesh_pre_processing()
        
        if self.cb_algorithm_selector.currentText() == "Axisymmetric":
            self.slice_axisymmetric()
        elif self.cb_algorithm_selector.currentText() == "Linear":
            self.slice_linear()

    def check_argument_validity(self, widget):
        # TODO
        # Compare the argument to it's value in a reference dictionary and return 1 if invalid
        return 0
    
    def mesh_pre_processing(self):
        self.mesh = trimesh.load_mesh(self.file_path)
        to_origin, self.extents = u.axis_oriented_extents(self.mesh)
        self.mesh.apply_transform(to_origin)

        # rad = math.pi
        # matrix = trimesh.transformations.rotation_matrix(rad, [1, 0, 0], mesh_.centroid)
        # mesh_ = mesh_.apply_transform(matrix)

    def slice_axisymmetric(self):
        u.msg("Running axysimmetric", "debug")
        
        # Check required arguments
        err = 0
        err += self.check_argument_validity(self.int_num_cuts)
        err += self.check_argument_validity(self.int_points_per_cut)
        err += self.check_argument_validity(self.float_kerf)
        if err > 0:
            u.msg("Invalid argument", "error")
            return
        
        if self.cb_do_gcode.isChecked():
            err = 0
            err += self.check_argument_validity(self.int_material_size_x)
            err += self.check_argument_validity(self.int_material_size_y)
            err += self.check_argument_validity(self.int_material_size_z)
            err += self.check_argument_validity(self.float_rotary_axis_origin_x)
            err += self.check_argument_validity(self.float_rotary_axis_origin_x)
            err += self.check_argument_validity(self.float_rotary_axis_origin_x)
            err += self.check_argument_validity(self.int_feed)
            err += self.check_argument_validity(self.int_slew)
            if err > 0:
                u.msg("Invalid argument", "error")
                return

        num_points = self.int_num_cuts.value() * self.int_points_per_cut.value()
        out, coords = slicer.axysimmetric(self.mesh, self.int_num_cuts.value(), num_points, self.float_kerf.value())
        if out.size != 0:
            self.render_pcd(out)

        if self.cb_do_gcode.isChecked():
            material_size = np.asarray([self.int_material_size_x.value(), self.int_material_size_y.value(), self.int_material_size_z.value()])
            rotary_axis_origin = np.asarray([self.float_rotary_axis_origin_x.value(), self.float_rotary_axis_origin_y.value(), self.float_rotary_axis_origin_z.value()])
            mesh_origin = rotary_axis_origin + (self.extents[2] / 2)
            gc.to_gcode_axysimmetric(self.file_name, coords, mesh_origin, material_size, self.int_feed.value(), self.int_slew.value())

    def slice_linear(self):
        u.msg("Running linear", "debug")
        
        # Check required arguments
        err = 0
        err += self.check_argument_validity(self.cb_render_planes)
        err += self.check_argument_validity(self.float_plane_data_x)
        err += self.check_argument_validity(self.float_plane_data_pos_y)
        err += self.check_argument_validity(self.float_plane_data_neg_y)
        err += self.check_argument_validity(self.float_plane_data_pos_z)
        err += self.check_argument_validity(self.float_plane_data_neg_z)
        err += self.check_argument_validity(self.float_kerf)
        if err > 0:
            u.msg("Invalid argument", "error")
            return
        
        motor_plane_data = np.asarray([self.float_plane_data_x.value(),
                                       self.float_plane_data_pos_y.value(),
                                       -self.float_plane_data_neg_y.value(),
                                       self.float_plane_data_pos_z.value(),
                                       -self.float_plane_data_neg_z.value()])
        
        out = slicer.linear(self.mesh, motor_plane_data, self.float_kerf.value(), self.cb_render_planes.isChecked())
        if out.size != 0:
            self.render_pcd(out)

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
