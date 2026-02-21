# Truss Analysis
**The Definitive Educational Tool for 2D Truss Analysis.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)

Truss Analysis in Civil Bench is a structural analysis software built with Python. It utilizes the Direct Stiffness Method (Finite Element Method) to solve 2D structural frames. Designed for civil engineering students and educators, it not only provides the answers but reveals the underlying matrices and calculations.

  <img src="SS.png" alt="TrussAnalysis" width="100%">
  <img src="SS1.png" alt="TrussAnalysis" width="100%">
  <img src="SS2.png" alt="TrussAnalysis" width="100%">
  <img src="SS3.png" alt="TrussAnalysis" width="100%">

##  Features
* ** Interactive Editor:** Click to draw nodes and members on a dark-themed, snap-to-grid canvas.
* ** Quarter-View Dashboard:** Simultaneously view 4 independent plots:
  * Deflected Shape
  * Shear Force Diagram (SFD)
  * Bending Moment Diagram (BMD)
  * Axial Force Diagram (Color-coded: Red = Tension, Blue = Compression)
* ** Glass-Box Reporting:** View the underlying Global Stiffness Matrix $[K]$, nodal displacements, and automated stability checks directly in the app.
* ** Export Capabilities:** Copy any diagram directly to your clipboard as an SVG, or export the entire calculation report to CSV for Excel.
* ** Save/Load:** Save your structural models as lightweight JSON files.

##  Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/PyStructPro.git
   cd PyStructPro

   
