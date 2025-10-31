# XPS-Schneider-README

Dieses Dokument soll alle Informationen zur Slicer-Seite des XPS-Schneiders kurz erläutern, damit die nächsten Diplomarbeitsgruppen einen Startpunkt haben und nicht dieselben Fehler machen wie wir.

Disclaimer: Ich bin KEIN Informatiker, ich programmiere nur gerne. Somit entschuldige ich mich im Voraus für die wahrscheinlich katastrophale Struktur und Implementierung dieser Software.

# Algorythmen
Im Laufe der ganzen Diplomarbeit wurde diese Software viel zu oft überarbeitet: teilweise aufgrund besserer Implementierungsmöglichkeiten, teilweise durch fehlende Libraries und unbrauchbare Laufzeiten. Dieser Abschnitt soll alle uns eingefallenen und ausprobierten **Algorithmen** beschreiben, mit deren Vor- und Nachteilen.

Egal welche Methode ausgewählt wird, empfiehlt sich erstmalig eine Implementierung in einer 3D-Software (Unity, Godot, etc.). Die Hälfte der "verlorenen" Zeit im Laufe des Projektes stammt aus sekundären Features, z. B. UI, 3D-Viewer, Settings usw. Die Implementierung in einer 3D-Software ist schneller, man muss keine sekundären Features implementieren und man kann deutlich leichter visuell debuggen.

Zusätzlich ist zu erwähnen, dass bei folgenden **Algorithmen** immer davon ausgegangen wurde, dass alle Schnittvorgänge direkt vom 3D-Modell in einem Durchgang generiert werden sollen. Es stellte sich vor kurzem heraus, dass rekursive Ansätze deutlich leichter sind und teilweise zu besseren Ergebnissen führen könnten. Remeshing kommt bei rekursiven Methoden allerdings wiederholt vor und hat trotz mehrerer Versuche bei uns nie wie gehofft funktioniert.

### 01. Projektionen
Betrachtet man ein 3D-Objekt orthogonal aus einer beliebigen Richtung, stellt die äußere Kontur des Objekts eine mögliche kollisionsfreie Schnittkontur für den Heißdraht dar. Diese Kontur kann durch eine 2D-Projektion entlang einer bestimmten Vektorrichtung ermittelt werden.

Dieser Ansatz ist vom Konzept her der einfachste und intuitivste, allerdings:

- Die generierten Schnittkonturen sind selten die effizientesten.
- Enge Ecken, die mit der vorgegebenen Schrägstellung geschnitten werden könnten, treten oft nur teilweise in den Projektionen auf.
- Die 2D-Projektionen können entweder im 3D-Space mit einer Kamera ermittelt werden (ohne extremen Zeitaufwand nur möglich in einer 3D-Software), oder es treten Probleme aufgrund der Konkavität des Bauteils auf.

Für einen rekursiven Ansatz wäre hier das Thema "Visual Hull" von Bedeutung, um sich das Remeshing zu ersparen. Voxels könnten für Approximierungen auch funktionieren.

### 02. Plane-Mesh Intersections
Wird eine Schnittebene vorgegeben, können die Schnittpunkte zu den 3D-Objekt-Kanten als Schnittkontur verwendet werden. Dieser Ansatz ist aus folgenden Gründen **nicht** empfehlenswert:

- Extrem ineffizient. _03. Raycasting_ macht praktisch dasselbe, nur schneller.
- Die ermittelten Koordinaten haben vor allem bei Low-Poly-Modellen keinen regelmäßigen Abstand zueinander. Somit müssen sie zusätzlich sortiert und in den richtigen Positionen regeneriert werden.
- Eine Ebene gibt keine Auskunft darüber, ob ein Schnitt entlang der gegebenen Koordinaten kollisionsfrei wäre. Dies muss für jeden Punkt einzeln überprüft werden (meistens mit rekursiven Raycasts), und idealerweise müssten noch sämtliche möglichen Schrägstellungen ausprobiert werden.
 
### 03. Raycasting
Die Konturkoordinaten können auch mit Raycasts um den Ursprung oder eine Ursprungsachse ermittelt werden. Vorteile:

- Keine Nachbearbeitungsschritte für die ermittelten Koordinaten notwendig (außer rekursiver Kollisionsüberprüfung)
- Genauigkeit einstellbar (= Anzahl der Koordinatenpunkte pro Schnitt)
- Horizontale Raycasts schließen automatisch nicht erreichbare Stellen aus und erfordern daher keine Mesh-Nachbearbeitung diesbezüglich.

Vom Nachteil allerdings:

- Ursprung und Richtung der Rays sind nicht klar definiert, die ideale Richtung zu finden ist deutlich komplexer, als nur horizontal zu casten.
- Der Abstand der Punkte zueinander ist immerhin variabel. Wenn z. B. alle Raycasts horizontal entlang der vertikalen Achse erfolgen, werden nach oben gerichtete Flächen und Rundungen ungenau oder gar nicht geschnitten.
- Die Raycasts neigen dazu, an schmalen Segmenten, die nicht direkt mittig platziert sind, vorbeizufliegen. Das führt hin und wieder zu falschen Schnitten (z. B. die nach oben gerichteten Ohren einer 3D-modellierten Katze werden abgeschnitten).

### 04. Single Tri-Cuts
Um die Schrägstellung auszunutzen und sicher zu sein, die Kontur möglichst genau zu schneiden, können für jede einzelne Mesh-Face Raycasts im Kreis zur Kollisionsüberprüfung ausgeführt werden. Theoretisch gesehen funktioniert das einwandfrei, praktisch ist dieser Ansatz:

1. VIEL ZU LANGSAM. Egal wie optimal die Implementierung, jede Fläche einzeln zu überprüfen, dauert EWIGKEITEN.
2. Suboptimal auch vom Ergebnis her. Angenommen, man findet einen guten Weg, die resultierenden Koordinaten effizient zu sortieren (ein schwieriges Problem an sich), würde dann jede einzelne Dreiecksfläche einzeln geschnitten werden. Die meisten sind bei hochqualitativeren Bauteilen nur wenige Millimeter breit. Approximierungen sind hierbei eindeutig die bessere Lösung.

### 05. STEP-Surface Cuts
Statt bei einer STL-Datei jedes Triangle zu überprüfen (siehe _04. Single Tri-Cuts_), könnte man denselben Ansatz auf einer STEP-Datei anwenden. STEP trennt Bauteile in Oberflächenelemente wie Zylinder und Halbkugeln. Wenn man für jede Oberflächenart eine Kollisionsüberprüfung und Koordinatenerzeugung einprogrammiert, werden (theoretisch) ideale Schnitte erzeugt.

Hier nochmal: vom Konzept her eine geniale Idee. Leider besteht ein Großteil der „komplexeren“ STEP-Dateien aus B-Spline-Surfaces. Eine B-Spline-Surface ist so variabel in ihrer Struktur, dass Koordinatenpositionen nicht so wie bei anderen Surfaces „einfach“ ermittelt werden können. Es wäre allerdings möglich, einen eher ineffizienten und suboptimalen **Algorithmus** für B-Spline-Surfaces zu implementieren und dafür beim Rest „perfekte“ Ergebnisse zu haben.

# Erklärung der Software
In diesem Absatz werde ich versuchen, die Struktur des Programms zu erklären. Details sind den einzelnen Dateien und den jeweiligen Kommentaren zu entnehmen.

### qtui.py
Beinhaltet die gesamte User Interface. Diese wurde mit QT-Designer hergestellt, lt. Empfehlung von Hr. Prof. Izaak. Möchte man sie modifizieren, kann im QT-Designer die Datei _ui.ui_ geöffnet und bearbeitet werden und schließlich im CMD _pyuic5 -o qtui.py ui.ui_ ausgeführt werden. Das regeneriert die Python-Datei mit den neuen Einstellungen.

### ui.py
Initialisiert Elemente der User Interface aus _qtui.py_ (z. B. den VTK-3D-Viewer) und verbindet alle UI-Elemente mit der restlichen Software. Hauptsächlich:

- Die Settings werden aktualisiert und weitergegeben.
- Der „Slice“-Knopf führt die jeweilige Funktion in _slicer.py_ aus.
- Die Ergebnisse aus dem Slice-Vorgang werden in Form einer Point Cloud (PCD) oder Line Cloud (LCD) gerendert.

### slicer.py
Gegliedert in zwei Hauptfunktionen:

- _Axysimmetric_ bezieht sich auf einen Schnittvorgang, bei dem die Schnittkoordinaten rund um die vertikale Bauteilachse generiert werden. Dabei wird die Schrägstellung nicht verwendet. Dieser ist implementiert als eine Reihe horizontaler Raycasts, versetzt entlang der vertikalen Achse (siehe _03. Raycasts_).
- _Linear_ versucht, die Schrägstellung auszunutzen, indem mögliche Schnittrichtungen entlang jeder Dreieckfläche ermittelt werden. Von der Idee her funktioniert dieser Prozess (vielleicht), allerdings fehlt die zweite Hälfte dieses Algorithmus’, die diese Schnitte möglichst effizient aussortieren soll.

### gcode.py
Wandelt die Schnittkoordinaten in G-Code um. Ein paar Kleinigkeiten sind hier suboptimal (z. B. sollte man mit M42 den Draht vor dem Eilgang ausschalten und nach dem Eilgang **einschalten**. Momentan wird allerdings der Draht nach dem Eilgang kurz aus- und wieder eingeschaltet). Diese Datei ist allerdings sehr kurz und einfach aufgebaut und sollte mithilfe der G-Code-Dokumentation leicht verständlich sein.

### utils.py
Hilfreiche Helper-Functions, die zwischendurch verwendet werden.

### step.py, step_utils.py, step_entities.py
Werden aktuell nicht verwendet. Kurz vor Diplomarbeitsende kam ich auf die Idee, für die _linear_-Implementierung STEP-Dateien zu verwenden, die, im Gegensatz zu einer STL, bei der jede Dreiecksfläche getrennt betrachtet werden muss, das 3D-Modell von Natur aus in Oberflächenelemente aufteilen. Rein aufgrund mangelnder Zeit bin ich damit nicht weitergekommen (Siehe *05. STEP-Surface Cuts*).
