# pyuic5 -o Gui.py Gui.ui
print('Programm startet!')

import locale
locale.setlocale(locale.LC_NUMERIC, '')

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget
import sys
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


from qtui import Ui_MainWindow

import slicer

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()  
        
        # Set up the user interface from Designer.
        self.setupUi(self)
        self.InFillValue = 0.0

        # Set up VTK rendering
        self.renderer = vtk.vtkRenderer()
        self.w_vtk_viewer.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.w_vtk_viewer.GetRenderWindow().GetInteractor()
        self.iren.Initialize()

        # Set up MatPlotLib
        self.mpl_layout = QVBoxLayout(self.w_mpl_plot)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.mpl_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas.draw()
    
        self.b_open_file.clicked.connect(self.load_file)
        self.b_slice.clicked.connect(self.compute_coordinates)

    file_path = None

    def load_file(self):
         # Open a file dialog and get the file path
        self.file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*)')
        
        if self.file_path:
            print(f'Selected file: {self.file_path}')
            file_name = self.file_path.rsplit('/', 1)[1]
            self.t_selected_file.setText(file_name)
            self.file_viewer(self.file_path)
            # TODO check for valid file extension
        else:
            print('No file selected')
            self.t_selected_file.setText("None")

    def file_viewer(self, file_):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()

        # Display colorcoded scaled axes above the mesh
        self.axes_actor = vtk.vtkAxesActor()
        self.axes_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1.0, 0.0, 0.0)  # Red X
        self.axes_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.0, 1.0, 0.0)  # Green Y
        self.axes_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.0, 0.0, 1.0)  # Blue Z
        bounds = actor.GetBounds()
        self.axes_actor.SetTotalLength(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        self.renderer.AddActor(self.axes_actor)

        self.w_vtk_viewer.GetRenderWindow().Render()

    def compute_coordinates(self):
        if self.file_path is None:
            print("No File Path")
            #TODO show this in UI, maybe highlight button
        else:
            out = slicer.main(self.file_path)
            if out.size != 0:
                self.plot(out)
    
    def plot(self, arr, color:str = "hotpink"):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.set_aspect('equal', adjustable='box')
        try:
            self.ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], color=color)
        except:
            self.ax.scatter(arr[:, 0], arr[:, 1], color=color)
        self.canvas.draw()

if __name__ == '__main__':
    print("running...")
    APP = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(APP.exec_())
