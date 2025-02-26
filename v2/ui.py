import locale
locale.setlocale(locale.LC_NUMERIC, '')

from PyQt5 import QtWidgets, QtCore
import sys

from qtui import Ui_MainWindow

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()  
        
        # Set up the user interface from Designer.
        self.setupUi(self)
        self.InFillValue = 0.0

if __name__ == '__main__':
    APP = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(APP.exec_())

