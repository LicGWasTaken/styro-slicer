# Installation

* Download and install git: https://git-scm.com/downloads/win
* Download and install python 3.13.2 and add it to PATH during the installation
* Open command prompt
* Navigate to your installation folder e.g. C:\Users\slicer
  - From your current folder, use ```cd <folder_name>``` to move to the installation folder (```C:\Users> cd slicer```)
  - Use ```cd ..``` to navigate back if needed (```C:\Users\slicer> cd ..```)
* Run ```git clone https://github.com/LicGWasTaken/styro-slicer```
* Create a virtual environment ```python -m venv venv```
* Activate the environment ```venv\Scripts\activate```
* Install dependences ```python -m pip install -r requirements.txt```
* Run ```python ui.py``` to start the program

## User Parameters
* ***kerf*** - offset in mm from the contour of the mesh. Dependent on wire diameter, resistance and current.
* ***feed*** - cutting speed (mm/min)
* ***slew*** - speed inbetween cuts (mm/min)
* ***vertical axis*** - axis to slice along
* ***slicing algorithm*** - linear or axisymmetric slicing:
  - axisymmetric is computationally quick, works for rotational parts and testing purposes.
  - linear is computationally intensive, has the highest precision and can manufacture most parts.

## Settings
*material size*
*motor plane data*
