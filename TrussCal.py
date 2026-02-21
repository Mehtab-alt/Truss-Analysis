"""
Truss Analysis
------------------------------------------------
The definitive educational tool for structural analysis.

Features:
- Quarter-View Dashboard: 4 Independent plots (Deflection, Shear, Moment, Axial).
- Glass-Box Report: Full Stiffness Matrix + Stability Checks + Axial Status Table.
- Interactive: Independent Zoom/Pan for each diagram.
- Export: Copy any diagram as SVG.
- Visuals: Axial Tension (Red) vs Compression (Blue) with labels.
"""

import sys
import json
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union
from collections import defaultdict
import csv
from datetime import datetime
import networkx as nx

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget, 
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QTextEdit, QCheckBox, QMessageBox, 
    QTabWidget, QGroupBox, QFormLayout, QButtonGroup,
    QFileDialog, QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QFont, QGuiApplication
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

# ==============================================================================
# 1. THEME & STYLING
# ==============================================================================
THEME = {
    "app_bg": "#182b2a",
    "sidebar_bg": "#11201f",
    "input_bg": "#223f3c",
    "border": "#33524e",
    "accent": "#2dd4bf",      # Teal
    "accent_text": "#0c1514",
    "text_main": "#F1F5F9",
    "text_dim": "#809795",
    "danger": "#f87171",      # Red (Tension/Error)
    "warning": "#fbbf24",     # Yellow
    "grid": "#33524e",
    "shear": "#fb923c",       # Orange
    "moment": "#38bdf8",      # Light Blue
    "compression": "#3b82f6"  # Blue (Compression)
}

STYLESHEET = f"""
QMainWindow {{ background-color: {THEME['app_bg']}; }}
QWidget {{ color: {THEME['text_main']}; font-family: 'Segoe UI', sans-serif; font-size: 10pt; }}
QDockWidget {{ border: 1px solid {THEME['border']}; }}
QDockWidget::title {{ background: {THEME['sidebar_bg']}; padding: 6px; font-weight: bold; }}
QLineEdit, QTextEdit, QCheckBox {{ 
    background-color: {THEME['input_bg']}; 
    border: 1px solid {THEME['border']}; 
    border-radius: 4px; padding: 4px; color: {THEME['text_main']}; 
}}
QLineEdit:read-only {{ background-color: {THEME['sidebar_bg']}; color: {THEME['text_dim']}; }}
QPushButton {{ 
    background-color: {THEME['input_bg']}; color: {THEME['accent']}; 
    border: 1px solid {THEME['accent']}; padding: 6px; border-radius: 4px; font-weight: bold; 
}}
QPushButton:hover {{ background-color: {THEME['accent']}; color: {THEME['accent_text']}; }}
QTabWidget::pane {{ border: 1px solid {THEME['border']}; }}
QTabBar::tab {{ background: {THEME['sidebar_bg']}; padding: 8px 16px; margin-right: 2px; }}
QTabBar::tab:selected {{ background: {THEME['accent']}; color: {THEME['accent_text']}; }}
QGroupBox {{ margin-top: 1.2em; border: 1px solid {THEME['border']}; padding-top: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; color: {THEME['text_dim']}; }}
QMenuBar {{ background-color: {THEME['sidebar_bg']}; color: {THEME['text_main']}; }}
QMenuBar::item:selected {{ background-color: {THEME['accent']}; color: {THEME['accent_text']}; }}
"""

# ==============================================================================
# 2. PHYSICS ENGINE (Model)
# ==============================================================================

@dataclass
class Node:
    id: int
    x: float
    y: float
    fixity: List[bool] = field(default_factory=lambda: [False, False, False])
    loads: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    disp: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    reaction: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    dof_indices: List[int] = field(default_factory=list)

class Element:
     def __init__(self, id: int, node_i: Node, node_j: Node, E: float, A: float, I: float):
         self.id = id
         self.node_i = node_i
         self.node_j = node_j
         self.E = E
         self.A = A
         self.I = I
         self.local_forces: np.ndarray = np.zeros(6, dtype=np.float64)

     @property
     def length(self) -> float:
         return math.hypot(self.node_j.x - self.node_i.x, self.node_j.y - self.node_i.y)

     def get_local_stiffness(self) -> np.ndarray:
         L = self.length
         if L < 1e-6: return np.zeros((6,6))
         E, A, I = self.E, self.A, self.I
         k = np.zeros((6, 6))
         w_A = (E * A) / L
         w_I1 = (12 * E * I) / (L**3)
         w_I2 = (6 * E * I) / (L**2)
         w_I3 = (4 * E * I) / L
         w_I4 = (2 * E * I) / L
         k[0,0] = w_A;   k[0,3] = -w_A
         k[3,0] = -w_A; k[3,3] = w_A
         k[1,1] = w_I1;  k[1,2] = w_I2;  k[1,4] = -w_I1; k[1,5] = w_I2
         k[2,1] = w_I2; k[2,2] = w_I3;  k[2,4] = -w_I2; k[2,5] = w_I4
         k[4,1] = -w_I1; k[4,2] = -w_I2; k[4,4] = w_I1;  k[4,5] = -w_I2
         k[5,1] = w_I2;  k[5,2] = w_I4;  k[5,4] = -w_I2; k[5,5] = w_I3
         return k

     def get_transform(self) -> np.ndarray:
         dx = self.node_j.x - self.node_i.x
         dy = self.node_j.y - self.node_i.y
         angle = math.atan2(dy, dx)
         c, s = math.cos(angle), math.sin(angle)
         T = np.zeros((6, 6))
         rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
         T[0:3, 0:3] = rot
         T[3:6, 3:6] = rot
         return T

class Model:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.elements: Dict[int, Element] = {}
        self.global_K: Optional[np.ndarray] = None
        self.report_html = ""
        
        # Validation Thresholds
        self.max_deflection_ratio = 1/200
        self.min_element_length = 1e-3
        self.min_section_area = 1e-6
        self.min_inertia = 1e-8
        self.max_E = 500e9
        self.min_E = 1e9

    def clear(self):
        self.nodes.clear()
        self.elements.clear()
        self.global_K = None
        self.report_html = ""

    def add_node(self, x, y, nid=None):
        for n in self.nodes.values():
            if math.hypot(n.x - x, n.y - y) < 0.5:
                return n.id
        if nid is None: nid = max(self.nodes.keys(), default=0) + 1
        self.nodes[nid] = Node(nid, x, y)
        return nid

    def add_element(self, n1_id, n2_id, E, A, I, eid=None):
        if n1_id == n2_id: return
        if eid is None: eid = max(self.elements.keys(), default=0) + 1
        self.elements[eid] = Element(eid, self.nodes[n1_id], self.nodes[n2_id], E, A, I)

    def merge_duplicates(self):
        position_to_nodes = defaultdict(list)
        for n in list(self.nodes.values()):
            pos = (round(n.x, 6), round(n.y, 6))
            position_to_nodes[pos].append(n)

        merged_nodes = {}
        new_nid = 1
        for pos, nodes in position_to_nodes.items():
            main_n = nodes[0]
            main_n.id = new_nid
            for other_n in nodes[1:]:
                for i in range(3):
                    main_n.fixity[i] = main_n.fixity[i] or other_n.fixity[i]
                    main_n.loads[i] += other_n.loads[i]
            merged_nodes[pos] = main_n
            new_nid += 1

        self.nodes = {n.id: n for n in merged_nodes.values()}

        new_elements = {}
        new_eid = 1
        for el in list(self.elements.values()):
            pos1 = (round(el.node_i.x, 6), round(el.node_i.y, 6))
            pos2 = (round(el.node_j.x, 6), round(el.node_j.y, 6))
            if pos1 != pos2:
                el.node_i = merged_nodes[pos1]
                el.node_j = merged_nodes[pos2]
                el.id = new_eid
                new_elements[new_eid] = el
                new_eid += 1
        self.elements = new_elements

    def delete_nearest(self, x, y):
        target_n = None
        min_d = 0.5
        for n in self.nodes.values():
            d = math.hypot(n.x - x, n.y - y)
            if d < min_d:
                min_d = d
                target_n = n.id
        
        if target_n:
            to_remove = [eid for eid, el in self.elements.items() if el.node_i.id == target_n or el.node_j.id == target_n]
            for eid in to_remove: del self.elements[eid]
            del self.nodes[target_n]
            return

        target_e = None
        min_d = 0.5
        for el in self.elements.values():
            p1 = np.array([el.node_i.x, el.node_i.y])
            p2 = np.array([el.node_j.x, el.node_j.y])
            p3 = np.array([x, y])
            l2 = np.sum((p1-p2)**2)
            if l2 == 0: continue
            t = max(0, min(1, np.dot(p3-p1, p2-p1) / l2))
            proj = p1 + t * (p2 - p1)
            d = np.linalg.norm(p3 - proj)
            if d < min_d:
                min_d = d
                target_e = el.id
        
        if target_e:
            del self.elements[target_e]

    def solve(self):
        if not self.nodes: raise ValueError("No nodes defined.")
        if not self.elements: raise ValueError("No members defined.")

        dof = 0
        for n in self.nodes.values():
            n.dof_indices = [dof, dof+1, dof+2]
            dof += 3
        
        K = np.zeros((dof, dof))
        F = np.zeros(dof)
        
        for el in self.elements.values():
            k_glob = el.get_transform().T @ el.get_local_stiffness() @ el.get_transform()
            indices = el.node_i.dof_indices + el.node_j.dof_indices
            for i_loc, i_glob in enumerate(indices):
                for j_loc, j_glob in enumerate(indices):
                    K[i_glob, j_glob] += k_glob[i_loc, j_loc]
        
        self.global_K = K 

        for n in self.nodes.values():
            for i in range(3): F[n.dof_indices[i]] = n.loads[i]
            
        free, fixed = [], []
        for n in self.nodes.values():
            for i in range(3):
                if n.fixity[i]: fixed.append(n.dof_indices[i])
                else: free.append(n.dof_indices[i])
        
        if not free: raise ValueError("Structure is fully fixed (No free DOFs).")
        if not fixed: raise ValueError("Unstable: No supports defined.")
        
        try:
            U_f = np.linalg.solve(K[np.ix_(free, free)], F[free])
        except np.linalg.LinAlgError:
            raise ValueError("Structure is unstable (Singular Matrix). Check connectivity.")
            
        U = np.zeros(dof)
        U[free] = U_f
        
        R = K @ U - F
        for n in self.nodes.values():
            n.disp = [float(U[i]) for i in n.dof_indices]
            n.reaction = [float(R[i]) if n.fixity[k] else 0.0 for k, i in enumerate(n.dof_indices)]
            
        for el in self.elements.values():
            u_el = np.concatenate([el.node_i.disp, el.node_j.disp])
            u_loc = el.get_transform() @ u_el
            el.local_forces = (el.get_local_stiffness() @ u_loc).copy()

        self.generate_report(K, F, U)

    def generate_report(self, K, F, U):
        # HTML Generation
        h = "<h2>Analysis Report</h2>"
        
        # 1. Analysis Summary
        h += "<h3>Analysis Summary</h3>"
        h += "<p>This analysis uses the direct stiffness method for 2D frame structures. Key metrics:</p>"
        h += "<ul>"
        h += f"<li>Total Degrees of Freedom: {K.shape[0]}</li>"
        h += f"<li>Number of Nodes: {len(self.nodes)}</li>"
        h += f"<li>Number of Elements: {len(self.elements)}</li>"
        h += f"<li>Max Displacement: {max(abs(d) for n in self.nodes.values() for d in n.disp):.6f} rad or m</li>"
        h += "</ul>"
        
        # 2. Global K Preview
        dof_count = K.shape[0]
        h += f"<h3>Global Stiffness Matrix (K) [{dof_count}x{dof_count}]</h3>"
        h += "<p>The stiffness matrix represents how the structure resists deformation. Each entry K[i,j] indicates the force required at DOF i to produce a unit displacement at DOF j.</p>"
        
        limit = dof_count if dof_count <= 24 else 12
        
        h += "<table border='1' cellspacing='0' cellpadding='4' style='border-color:#444; font-family: Consolas, monospace;'>"
        for i in range(limit):
            h += "<tr>"
            for j in range(limit):
                val = K[i, j]
                color = "#2dd4bf" if abs(val) > 1e-3 else "#555"
                weight = "bold" if abs(val) > 1e-3 else "normal"
                h += f"<td style='color:{color}; font-weight:{weight}'>{val:.2e}</td>"
            h += "</tr>"
        h += "</table>"
        if dof_count > limit:
            h += f"<p>... (Matrix truncated. Showing top-left {limit}x{limit}) ...</p>"

        # 3. Nodal Results
        h += "<h3>Nodal Displacements & Reactions</h3>"
        h += "<p>Displacements show how the structure deforms under applied loads. Reactions represent support forces maintaining equilibrium.</p>"
        h += "<table border='1' cellspacing='0' cellpadding='4' style='font-family: Consolas, monospace; width:100%; border-collapse: collapse;'>"
        h += "<tr style='background-color:#11201f'><th>Node</th><th>Ux (mm)</th><th>Uy (mm)</th><th>Rot (rad)</th><th>Rx (N)</th><th>Ry (N)</th><th>Mz (Nm)</th></tr>"
        for n in self.nodes.values():
            row_style = "background-color:#223f3c" if any(n.fixity) else ""
            fix_icon = "🔒" if any(n.fixity) else ""
            h += f"<tr style='{row_style}'><td>{n.id} {fix_icon}</td>"
            h += f"<td>{n.disp[0]*1000:.3f}</td><td>{n.disp[1]*1000:.3f}</td><td>{n.disp[2]:.5f}</td>"
            h += f"<td>{n.reaction[0]:.1f}</td><td>{n.reaction[1]:.1f}</td><td>{n.reaction[2]:.1f}</td></tr>"
        h += "</table>"

        # 4. Member Forces (Standard)
        h += "<h3>Member Local Forces</h3>"
        h += "<p>Internal forces in each element. Positive axial forces indicate tension, negative indicate compression.</p>"
        h += "<table border='1' cellspacing='0' cellpadding='4' style='font-family: Consolas, monospace; width:100%; border-collapse: collapse;'>"
        h += "<tr style='background-color:#11201f'><th>Elem</th><th>Axial (N)</th><th>Shear Start (N)</th><th>Moment Start (Nm)</th><th>Moment End (Nm)</th></tr>"
        for el in self.elements.values():
            f = el.local_forces
            h += f"<tr><td>{el.id}</td>"
            h += f"<td>{f[3]:.1f}</td>"
            h += f"<td>{f[1]:.1f}</td>"
            h += f"<td>{f[2]:.1f}</td>"
            h += f"<td>{-f[5]:.1f}</td></tr>"
        h += "</table>"

        # 5. NEW: Member Axial Status Table (Integrated here)
        h += "<h3>Member Axial Status (Tension/Compression)</h3>"
        h += "<table border='1' cellspacing='0' cellpadding='4' style='font-family: Consolas, monospace; width:100%; border-collapse: collapse;'>"
        h += "<tr style='background-color:#11201f'><th>Element</th><th>Nodes</th><th>Force Magnitude (N)</th><th>Status</th></tr>"
        
        for el in self.elements.values():
            # Index 3 is axial force at node J.
            axial_force = el.local_forces[3]
            
            if axial_force > 1e-5:
                status = "TENSION"
                style = f"color:{THEME['danger']}; font-weight:bold;"
            elif axial_force < -1e-5:
                status = "COMPRESSION"
                style = f"color:{THEME['compression']}; font-weight:bold;"
            else:
                status = "ZERO"
                style = "color:#777;"
            
            h += f"<tr><td>{el.id}</td><td>{el.node_i.id}-{el.node_j.id}</td>"
            h += f"<td>{abs(axial_force):.2f}</td>"
            h += f"<td style='{style}'>{status}</td></tr>"
        h += "</table>"
        
        # 6. Analysis Assumptions
        h += "<h3>Analysis Assumptions</h3>"
        h += "<ul>"
        h += "<li>Linear elastic behavior (Hooke's law applies).</li>"
        h += "<li>Small deformations (no geometric nonlinearity).</li>"
        h += "<li>2D plane frame; all loads in-plane.</li>"
        h += "<li>Euler-Bernoulli beams (shear deformation neglected).</li>"
        h += "<li>Static analysis only; no dynamic or environmental effects.</li>"
        h += "<li>Results are approximate; verify with hand calculations for critical designs.</li>"
        h += "</ul>"
        
        # 7. Automated Comments
        h += "<h3>Automated Analysis Comments</h3>"
        h += "<ul>"
        
        if len(self.nodes) < 2:
            h += "<li style='color:#f87171;'>❌ Error: Structure has fewer than 2 nodes. Add more for a valid frame.</li>"
        if len(self.elements) == 0:
            h += "<li style='color:#f87171;'>❌ Error: No elements defined. Connect nodes with members.</li>"
        supported_nodes = sum(1 for n in self.nodes.values() if any(n.fixity))
        if supported_nodes == 0:
            h += "<li style='color:#f87171;'>❌ Error: No supports defined. Structure is unstable—add fixities!</li>"
        elif supported_nodes == 1:
            h += "<li style='color:#fbbf24;'>⚠️ Note: Only one support defined. May be unstable for frames; verify degrees of freedom.</li>"
        else:
            h += "<li style='color:#2dd4bf;'>✓ Structure has adequate supports (" + str(supported_nodes) + " nodes fixed).</li>"
        loaded_nodes = sum(1 for n in self.nodes.values() if any(abs(float(l)) > 1e-6 for l in n.loads))
        if loaded_nodes == 0:
            h += "<li style='color:#fbbf24;'>⚠️ Note: No external loads applied. All results will be zero—add loads for meaningful analysis.</li>"
        else:
            h += "<li style='color:#2dd4bf;'>✓ " + str(loaded_nodes) + " node(s) have applied loads.</li>"
        
        invalid_elements = [el for el in self.elements.values() if el.length < self.min_element_length or el.A <= self.min_section_area or el.I <= self.min_inertia]
        if invalid_elements:
            h += "<li style='color:#f87171;'>❌ Error: " + str(len(invalid_elements)) + " invalid element(s) (zero-length, negative/zero area, or inertia). Check elements: " + ", ".join(str(el.id) for el in invalid_elements) + ". Fix sections or remove duplicates.</li>"
        
        extreme_materials = [el for el in self.elements.values() if el.E < self.min_E or el.E > self.max_E]
        if extreme_materials:
            h += "<li style='color:#fbbf24;'>⚠️ Note: Unusual material stiffness (E) in " + str(len(extreme_materials)) + " element(s). Typical steel E=200 GPa. Check elements: " + ", ".join(str(el.id) for el in extreme_materials) + ".</li>"
        
        max_disp = max((max(abs(float(d)) for d in n.disp) for n in self.nodes.values()), default=0)
        max_element_L = max((el.length for el in self.elements.values()), default=1)
        if max_disp > self.max_deflection_ratio * max_element_L:
            h += "<li style='color:#f87171;'>❌ Warning: Large deflections detected (max δ = " + f"{max_disp:.4f}" + " > L/" + str(int(1/self.max_deflection_ratio)) + "). Small deformation assumption violated—consider nonlinear analysis or stiffen structure.</li>"
        else:
            h += "<li style='color:#2dd4bf;'>✓ Deflections within limits (small deformation assumption holds).</li>"
        
        G = nx.Graph()
        for el in self.elements.values():
            G.add_edge(el.node_i.id, el.node_j.id)
        connected_components = list(nx.connected_components(G))
        if len(connected_components) > 1:
            h += "<li style='color:#f87171;'>❌ Error: Disconnected structure (" + str(len(connected_components)) + " separate components). Connect all parts or analyze separately.</li>"
        elif len(connected_components) == 0:
            h += "<li style='color:#f87171;'>❌ Error: No connected components. Add elements.</li>"
        else:
            h += "<li style='color:#2dd4bf;'>✓ Structure is fully connected (single component).</li>"
        
        fixed_dofs = sum(sum(n.fixity) for n in self.nodes.values())
        if fixed_dofs < 3:
            h += "<li style='color:#f87171;'>❌ Error: Insufficient constraints (" + str(fixed_dofs) + " fixed DOFs). Need at least 3 for stability.</li>"
        else:
            redundancies = max(0, fixed_dofs - 3)
            if redundancies == 0:
                h += "<li style='color:#2dd4bf;'>✓ Structure appears statically determinate.</li>"
            else:
                h += "<li style='color:#fbbf24;'>⚠️ Note: Structure is statically indeterminate (" + str(redundancies) + " redundancy). FEM suitable, but check for overconstraints.</li>"
        
        max_load = max((max(abs(float(l)) for l in n.loads) for n in self.nodes.values()), default=0)
        if max_load > 1e6:
            h += "<li style='color:#fbbf24;'>⚠️ Note: Very large loads detected (max = " + f"{max_load:.2e}" + " N/Nm). Check units or scale; may cause numerical issues.</li>"
        elif max_load < 1e-3 and max_load > 0:
            h += "<li style='color:#fbbf24;'>⚠️ Note: Very small loads detected. Results may be near zero—check units.</li>"
        
        h += "</ul>"
        self.report_html = h

    def to_json(self):
        data = {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "elements": []
        }
        for el in self.elements.values():
            data["elements"].append({
                "id": el.id, "n1": el.node_i.id, "n2": el.node_j.id,
                "E": el.E, "A": el.A, "I": el.I
            })
        return json.dumps(data, indent=2)

    def load_json(self, json_str):
        self.clear()
        data = json.loads(json_str)
        for n_data in data["nodes"]:
            n = Node(**n_data)
            self.nodes[n.id] = n
        for e_data in data["elements"]:
            self.add_element(e_data["n1"], e_data["n2"],
                             e_data["E"], e_data["A"], e_data["I"], eid=e_data["id"])
        self.merge_duplicates()

# ==============================================================================
# 3. INTERACTIVE EDITOR CANVAS
# ==============================================================================

class EditorCanvas(FigureCanvasQTAgg):
    def __init__(self, model: Model):
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor(THEME['app_bg'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#182b2a")
        super().__init__(self.fig)
        self.model = model
        self.mode = "SELECT"
        self.temp_node = None
        self.canvas_size = 10
        self.hover_obj = None
        self.default_material = {"E": 200e9, "A": 0.005, "I": 0.0001}
        self.mpl_connect("button_press_event", self.on_click)
        self.mpl_connect("motion_notify_event", self.on_hover)
        self.setup_plot()

    def setup_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-1, self.canvas_size+1)
        self.ax.set_ylim(-1, self.canvas_size+1)
        self.ax.grid(True, color=THEME['grid'], linestyle='--', alpha=0.5)
        self.ax.set_aspect('equal')
        for spine in self.ax.spines.values(): spine.set_color(THEME['grid'])
        self.ax.tick_params(colors=THEME['text_dim'])

    def fit_view(self):
        if not self.model.nodes: return
        xs = [n.x for n in self.model.nodes.values()]
        ys = [n.y for n in self.model.nodes.values()]
        margin = 2.0
        self.ax.set_xlim(min(xs) - margin, max(xs) + margin)
        self.ax.set_ylim(min(ys) - margin, max(ys) + margin)
        self.draw()

    def redraw(self):
        self.ax.clear()
        self.setup_plot()
        
        # Elements
        for el in self.model.elements.values():
            color = THEME['accent'] if (self.hover_obj == ('EL', el.id)) else THEME['text_dim']
            lw = 4 if (self.hover_obj == ('EL', el.id)) else 2
            self.ax.plot([el.node_i.x, el.node_j.x], [el.node_i.y, el.node_j.y], 
                         color=color, lw=lw, zorder=1)
            # Member Labels
            mid_x = (el.node_i.x + el.node_j.x) / 2
            mid_y = (el.node_i.y + el.node_j.y) / 2
            self.ax.text(mid_x, mid_y, f"{el.node_i.id}_{el.node_j.id}", 
                         color=THEME['text_dim'], fontsize=8, ha='center', 
                         bbox=dict(facecolor=THEME['app_bg'], edgecolor='none', alpha=0.7))
            
        # Nodes
        x = [n.x for n in self.model.nodes.values()]
        y = [n.y for n in self.model.nodes.values()]
        if x: 
            colors = []
            sizes = []
            for n in self.model.nodes.values():
                if self.hover_obj == ('NODE', n.id):
                    colors.append(THEME['warning'])
                    sizes.append(100)
                else:
                    colors.append(THEME['accent'])
                    sizes.append(50)
            self.ax.scatter(x, y, c=colors, s=sizes, zorder=2, edgecolors=THEME['app_bg'])
            
            # Node Labels
            for n in self.model.nodes.values():
                self.ax.text(n.x, n.y + 0.3, str(n.id), color=THEME['text_main'], ha='center', fontsize=9, fontweight='bold')
        
        # Supports
        for n in self.model.nodes.values():
            if any(n.fixity):
                self.ax.plot(n.x, n.y-0.3, marker='^', color=THEME['danger'], markersize=8, zorder=3)
        
        # Loads
        for n in self.model.nodes.values():
            if abs(n.loads[0]) > 0 or abs(n.loads[1]) > 0:
                self.ax.arrow(n.x, n.y+0.8, 0, -0.4, head_width=0.2, color=THEME['warning'], zorder=4)

        self.draw()

    def on_click(self, event):
        if event.inaxes != self.ax: return
        x, y = round(event.xdata*2)/2, round(event.ydata*2)/2
        
        if self.mode == "NODE":
            self.model.add_node(x, y)
            self.redraw()
        elif self.mode == "MEMBER":
            nid = self.model.add_node(x, y)
            if self.temp_node is None:
                self.temp_node = nid
            else:
                self.model.add_element(self.temp_node, nid, 
                                     self.default_material["E"], 
                                     self.default_material["A"], 
                                     self.default_material["I"])
                self.temp_node = None
                self.redraw()
        elif self.mode == "DELETE":
            self.model.delete_nearest(event.xdata, event.ydata)
            self.redraw()

    def on_hover(self, event):
        if event.inaxes != self.ax: return
        prev_hover = self.hover_obj
        self.hover_obj = None
        
        for n in self.model.nodes.values():
            if math.hypot(n.x - event.xdata, n.y - event.ydata) < 0.4:
                self.hover_obj = ('NODE', n.id)
                break
        
        if not self.hover_obj:
            for el in self.model.elements.values():
                p1 = np.array([el.node_i.x, el.node_i.y])
                p2 = np.array([el.node_j.x, el.node_j.y])
                p3 = np.array([event.xdata, event.ydata])
                l2 = np.sum((p1-p2)**2)
                if l2 == 0: continue
                t = max(0, min(1, np.dot(p3-p1, p2-p1) / l2))
                dist = np.linalg.norm(p3 - (p1 + t * (p2 - p1)))
                if dist < 0.3:
                    self.hover_obj = ('EL', el.id)
                    break
        
        if self.hover_obj != prev_hover:
            self.redraw()

# ==============================================================================
# 4. RESULT VISUALIZER (Quarter Format)
# ==============================================================================

class QuarterPlot(QWidget):
    """A custom widget containing a single plot, toolbar, and export button."""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(self.layout)
        
        # Figure and Canvas
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.fig.patch.set_facecolor(THEME['app_bg'])
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setup_ax(title)
        
        self.layout.addWidget(self.canvas)
        
        # Toolbar and Button Layout
        btn_layout = QHBoxLayout()
        
        # Navigation Toolbar (Zoom/Pan)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet(f"background-color: {THEME['input_bg']}; color: black;")
        btn_layout.addWidget(self.toolbar)
        
        # Export Button
        self.btn_copy = QPushButton("Copy SVG")
        self.btn_copy.setStyleSheet(f"background-color: {THEME['input_bg']}; color: {THEME['text_main']}; border: 1px solid {THEME['border']};")
        self.btn_copy.clicked.connect(self.copy_svg)
        btn_layout.addWidget(self.btn_copy)
        
        self.layout.addLayout(btn_layout)

    def setup_ax(self, title):
        self.ax.clear()
        self.ax.set_facecolor(THEME['app_bg'])
        self.ax.set_title(title, color=THEME['text_main'], fontsize=10, pad=10)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        for spine in self.ax.spines.values(): spine.set_color(THEME['grid'])
        self.ax.set_aspect('equal')

    def copy_svg(self):
        import io
        svg_buffer = io.BytesIO()
        self.fig.savefig(svg_buffer, format='svg', facecolor=self.fig.get_facecolor(), bbox_inches='tight')
        svg_string = svg_buffer.getvalue().decode('utf-8')
        svg_buffer.close()
        
        app = QApplication.instance()
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(svg_string)
        QMessageBox.information(self, "Copied", "Diagram copied to clipboard as SVG.")

class ResultCanvas(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        
        # Create 4 Independent Quarter Plots
        self.plot_def = QuarterPlot("Deflected Shape")
        self.plot_shear = QuarterPlot("Shear Force Diagram (SFD)")
        self.plot_moment = QuarterPlot("Bending Moment Diagram (BMD)")
        self.plot_axial = QuarterPlot("Axial Force (Tension/Compression)")
        
        # Add to Grid (2x2)
        self.layout.addWidget(self.plot_def, 0, 0)
        self.layout.addWidget(self.plot_shear, 0, 1)
        self.layout.addWidget(self.plot_moment, 1, 0)
        self.layout.addWidget(self.plot_axial, 1, 1)

    def plot_results(self):
        # Clear all axes
        for plot in [self.plot_def, self.plot_shear, self.plot_moment, self.plot_axial]:
            plot.ax.clear()
            plot.ax.set_facecolor(THEME['app_bg'])
            plot.ax.set_xticks([]); plot.ax.set_yticks([])
            for spine in plot.ax.spines.values(): spine.set_color(THEME['grid'])
            plot.ax.set_aspect('equal')

        # Set Titles
        self.plot_def.ax.set_title("Deflected Shape", color=THEME['text_main'])
        self.plot_shear.ax.set_title("Shear Force", color=THEME['shear'])
        self.plot_moment.ax.set_title("Bending Moment", color=THEME['moment'])
        self.plot_axial.ax.set_title("Axial (Red=T, Blue=C)", color=THEME['text_main'])

        if self.model.global_K is None: return

        # Auto-Scaling
        max_disp = max((max(abs(float(n.disp[0])), abs(float(n.disp[1]))) for n in self.model.nodes.values()), default=0)
        scale_def = 2.0 / max_disp if max_disp > 1e-6 else 1.0
        
        max_shear = max((max(abs(float(el.local_forces[1])), abs(float(el.local_forces[4]))) for el in self.model.elements.values()), default=0)
        scale_shear = 1.0 / max_shear if max_shear > 1e-6 else 1.0
        
        max_moment = max((max(abs(float(el.local_forces[2])), abs(float(el.local_forces[5]))) for el in self.model.elements.values()), default=0)
        scale_moment = 1.0 / max_moment if max_moment > 1e-6 else 1.0
        
        max_axial = max((abs(float(el.local_forces[3])) for el in self.model.elements.values()), default=0)
        scale_axial = 1.0 / max_axial if max_axial > 1e-6 else 1.0

        # Helper to draw labels
        def draw_labels(ax, n1, n2, el_id):
            # Node Labels
            ax.text(n1.x, n1.y, str(n1.id), color=THEME['text_dim'], fontsize=7, ha='center', va='center', bbox=dict(facecolor=THEME['app_bg'], edgecolor='none', alpha=0.5))
            ax.text(n2.x, n2.y, str(n2.id), color=THEME['text_dim'], fontsize=7, ha='center', va='center', bbox=dict(facecolor=THEME['app_bg'], edgecolor='none', alpha=0.5))
            # Member Label
            mx, my = (n1.x + n2.x)/2, (n1.y + n2.y)/2
            ax.text(mx, my, f"{n1.id}_{n2.id}", color=THEME['text_dim'], fontsize=6, ha='center', alpha=0.7)

        for el in self.model.elements.values():
            n1, n2 = el.node_i, el.node_j
            L = el.length
            dx, dy = n2.x - n1.x, n2.y - n1.y
            ux, uy = dx/L, dy/L
            vx, vy = -uy, ux
            
            # --- 1. Deflection ---
            ax = self.plot_def.ax
            ax.plot([n1.x, n2.x], [n1.y, n2.y], ':', color=THEME['text_dim'], alpha=0.5)
            d1x, d1y = n1.x + n1.disp[0]*scale_def, n1.y + n1.disp[1]*scale_def
            d2x, d2y = n2.x + n2.disp[0]*scale_def, n2.y + n2.disp[1]*scale_def
            ax.plot([d1x, d2x], [d1y, d2y], color=THEME['accent'], lw=2)
            draw_labels(ax, n1, n2, el.id)

            # --- Helper for SFD/BMD ---
            def draw_poly(ax, v1, v2, scale, color, invert=False):
                s = -1 if invert else 1
                p1 = (n1.x, n1.y)
                p2 = (n1.x + v1*scale*s*vx, n1.y + v1*scale*s*vy)
                p3 = (n2.x + v2*scale*s*vx, n2.y + v2*scale*s*vy)
                p4 = (n2.x, n2.y)
                ax.add_patch(Polygon([p1, p2, p3, p4], closed=True, color=color, alpha=0.6))
                ax.plot([n1.x, n2.x], [n1.y, n2.y], color=THEME['text_dim'], lw=1, alpha=0.3)
                
                peak_val = max(abs(v1), abs(v2))
                if peak_val > 0.1:
                    ax.text((n1.x+n2.x)/2, (n1.y+n2.y)/2, f"{peak_val:.1f}", color=color, fontsize=7, ha='center', fontweight='bold')

            # --- 2. Shear ---
            ax = self.plot_shear.ax
            draw_poly(ax, el.local_forces[1], -el.local_forces[4], scale_shear, THEME['shear'])
            draw_labels(ax, n1, n2, el.id)

            # --- 3. Moment ---
            ax = self.plot_moment.ax
            draw_poly(ax, el.local_forces[2], -el.local_forces[5], scale_moment, THEME['moment'], invert=True)
            draw_labels(ax, n1, n2, el.id)

            # --- 4. Axial (Tension/Compression) ---
            ax = self.plot_axial.ax
            axial_val = el.local_forces[3] # Force at node J (Positive = Tension)
            
            # Color Logic: Red for Tension (+), Blue for Compression (-)
            if axial_val > 0:
                col_ax = THEME['danger'] # Red
            else:
                col_ax = THEME['compression'] # Blue
                
            thickness = max(0.05, abs(axial_val) * scale_axial * 0.5)
            
            # Draw Axial Box
            p1 = (n1.x + thickness*vx, n1.y + thickness*vy)
            p2 = (n1.x - thickness*vx, n1.y - thickness*vy)
            p3 = (n2.x - thickness*vx, n2.y - thickness*vy)
            p4 = (n2.x + thickness*vx, n2.y + thickness*vy)
            
            if abs(axial_val) > 1e-3:
                ax.add_patch(Polygon([p1, p2, p3, p4], closed=True, color=col_ax, alpha=0.6))
                ax.text((n1.x+n2.x)/2, (n1.y+n2.y)/2, f"{abs(axial_val):.1f}", color=col_ax, fontsize=7, ha='center', fontweight='bold')
            
            ax.plot([n1.x, n2.x], [n1.y, n2.y], color=THEME['text_dim'], lw=1, alpha=0.3)
            draw_labels(ax, n1, n2, el.id)

        # Refresh Canvases
        self.plot_def.canvas.draw()
        self.plot_shear.canvas.draw()
        self.plot_moment.canvas.draw()
        self.plot_axial.canvas.draw()

# ==============================================================================
# 5. MAIN APPLICATION
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Truss Analysis")
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)
        
        self.model = Model()
        self.setup_menu()
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.editor = EditorCanvas(self.model)
        self.tabs.addTab(self.editor, "Structure Editor")
        
        self.results = ResultCanvas(self.model)
        self.tabs.addTab(self.results, "Visual Results")
        
        report_widget = QWidget()
        report_layout = QVBoxLayout()
        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.report_view.setFont(font)
        self.report_view.setStyleSheet(f"background-color: {THEME['input_bg']};")
        report_layout.addWidget(self.report_view)
        
        export_btn = QPushButton("Export Calculations to Spreadsheet")
        export_btn.clicked.connect(self.export_calculations)
        report_layout.addWidget(export_btn)
        
        report_widget.setLayout(report_layout)
        self.tabs.addTab(report_widget, "Calculation Report")
        
        self.setup_ui()
        self.editor.mpl_connect("button_press_event", self.on_canvas_select)

    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        save_act = QAction("Save Project", self)
        save_act.triggered.connect(self.save_project)
        file_menu.addAction(save_act)
        load_act = QAction("Open Project", self)
        load_act.triggered.connect(self.load_project)
        file_menu.addAction(load_act)

    def setup_ui(self):
        dock = QDockWidget("Inspector", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        widget = QWidget()
        layout = QVBoxLayout()
        
        grp_tools = QGroupBox("Editor Tools")
        l_tools = QHBoxLayout()
        self.btn_grp = QButtonGroup()
        for mode in ["SELECT", "NODE", "MEMBER", "DELETE"]:
            btn = QPushButton(mode)
            btn.setCheckable(True)
            if mode == "SELECT": btn.setChecked(True)
            btn.clicked.connect(lambda _, m=mode: setattr(self.editor, 'mode', m))
            self.btn_grp.addButton(btn)
            l_tools.addWidget(btn)
        grp_tools.setLayout(l_tools)
        layout.addWidget(grp_tools)
        
        grp_node = QGroupBox("Selected Node")
        f_node = QFormLayout()
        self.in_nid = QLineEdit(); self.in_nid.setReadOnly(True)
        self.chk_fx = QCheckBox("Fix X")
        self.chk_fy = QCheckBox("Fix Y")
        self.chk_mz = QCheckBox("Fix Rot")
        self.in_lx = QLineEdit("0")
        self.in_ly = QLineEdit("0")
        self.in_lm = QLineEdit("0")
        self.in_lx.returnPressed.connect(self.update_node)
        self.in_ly.returnPressed.connect(self.update_node)
        self.in_lm.returnPressed.connect(self.update_node)
        btn_upd_node = QPushButton("Update Node")
        btn_upd_node.clicked.connect(self.update_node)
        f_node.addRow("ID:", self.in_nid)
        f_node.addRow(self.chk_fx, self.chk_fy)
        f_node.addRow(self.chk_mz)
        f_node.addRow("Load X (N):", self.in_lx)
        f_node.addRow("Load Y (N):", self.in_ly)
        f_node.addRow("Moment (Nm):", self.in_lm)
        f_node.addRow(btn_upd_node)
        grp_node.setLayout(f_node)
        layout.addWidget(grp_node)
        
        grp_el = QGroupBox("Member Properties")
        f_el = QFormLayout()
        self.in_eid = QLineEdit(); self.in_eid.setReadOnly(True)
        self.in_E = QLineEdit("200")
        self.in_A = QLineEdit("0.005")
        self.in_I = QLineEdit("0.0001")
        self.in_E.textChanged.connect(self.update_defaults)
        self.in_A.textChanged.connect(self.update_defaults)
        self.in_I.textChanged.connect(self.update_defaults)
        btn_upd_el = QPushButton("Update Member")
        btn_upd_el.clicked.connect(self.update_element)
        f_el.addRow("ID:", self.in_eid)
        f_el.addRow("E (GPa):", self.in_E)
        f_el.addRow("Area (m²):", self.in_A)
        f_el.addRow("I (m⁴):", self.in_I)
        f_el.addRow(btn_upd_el)
        grp_el.setLayout(f_el)
        layout.addWidget(grp_el)
        
        grp_sz = QGroupBox("Grid Settings")
        l_sz = QHBoxLayout()
        self.in_sz = QLineEdit("10")
        btn_sz = QPushButton("Set Size")
        btn_sz.clicked.connect(lambda: setattr(self.editor, 'canvas_size', float(self.in_sz.text())) or self.editor.setup_plot())
        l_sz.addWidget(self.in_sz); l_sz.addWidget(btn_sz)
        grp_sz.setLayout(l_sz)
        layout.addWidget(grp_sz)

        btn_run = QPushButton("SOLVE STRUCTURE")
        btn_run.setStyleSheet(f"background-color: {THEME['accent']}; color: black; padding: 12px; font-weight: bold;")
        btn_run.clicked.connect(self.run_analysis)
        layout.addStretch()
        layout.addWidget(btn_run)
        
        widget.setLayout(layout)
        dock.setWidget(widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def on_canvas_select(self, event):
        if self.editor.mode != "SELECT": return
        obj_type, obj_id = self.editor.hover_obj if self.editor.hover_obj else (None, None)
        
        if obj_type == 'NODE':
            n = self.model.nodes[obj_id]
            self.in_nid.setText(str(n.id))
            self.chk_fx.setChecked(n.fixity[0])
            self.chk_fy.setChecked(n.fixity[1])
            self.chk_mz.setChecked(n.fixity[2])
            self.in_lx.setText(str(n.loads[0]))
            self.in_ly.setText(str(n.loads[1]))
            self.in_lm.setText(str(n.loads[2]))
            self.in_eid.clear()
        elif obj_type == 'EL':
            el = self.model.elements[obj_id]
            self.in_eid.setText(str(el.id))
            self.in_E.setText(str(el.E / 1e9))
            self.in_A.setText(str(el.A))
            self.in_I.setText(str(el.I))
            self.in_nid.clear()

    def update_defaults(self):
        try:
            self.editor.default_material["E"] = float(self.in_E.text()) * 1e9
            self.editor.default_material["A"] = float(self.in_A.text())
            self.editor.default_material["I"] = float(self.in_I.text())
        except ValueError:
            pass

    def update_node(self):
        try:
            nid = int(self.in_nid.text())
            n = self.model.nodes[nid]
            n.fixity = [self.chk_fx.isChecked(), self.chk_fy.isChecked(), self.chk_mz.isChecked()]
            n.loads = [float(self.in_lx.text() or 0), float(self.in_ly.text() or 0), float(self.in_lm.text() or 0)]
            self.editor.redraw()
        except ValueError: pass

    def update_element(self):
        try:
            eid = int(self.in_eid.text())
            el = self.model.elements[eid]
            el.E = float(self.in_E.text()) * 1e9
            el.A = float(self.in_A.text())
            el.I = float(self.in_I.text())
            QMessageBox.information(self, "Success", f"Member {eid} properties updated.")
        except ValueError: pass

    def run_analysis(self):
        try:
            E = self.safe_float(self.in_E.text(), 200) * 1e9
            A = self.safe_float(self.in_A.text(), 0.005)
            I = self.safe_float(self.in_I.text(), 0.0001)
            for el in self.model.elements.values():
                el.E, el.A, el.I = E, A, I

            self.model.merge_duplicates()
            self.model.solve()
            self.results.plot_results()
            self.report_view.setHtml(self.model.report_html)
            self.tabs.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", str(e))

    def safe_float(self, text, default):
        try: return float(text)
        except ValueError: return default

    def export_calculations(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Calculations", f"calculations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                writer.writerow(["Truss Analysis Report"])
                writer.writerow(["Generated on:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow([])
                
                # Analysis Summary
                writer.writerow(["ANALYSIS SUMMARY"])
                writer.writerow(["Total Degrees of Freedom:", self.model.global_K.shape[0] if self.model.global_K is not None else 0])
                writer.writerow(["Number of Nodes:", len(self.model.nodes)])
                writer.writerow(["Number of Elements:", len(self.model.elements)])
                max_disp = max(abs(d) for n in self.model.nodes.values() for d in n.disp) if self.model.nodes else 0
                writer.writerow(["Max Displacement (m or rad):", f"{max_disp:.6f}"])
                writer.writerow([])
                
                # Global Stiffness Matrix
                if self.model.global_K is not None:
                    dof_count = self.model.global_K.shape[0]
                    writer.writerow([f"GLOBAL STIFFNESS MATRIX (K) [{dof_count}x{dof_count}]"])
                    
                    # Write column headers
                    col_headers = ["Row/Col"]
                    for j in range(min(dof_count, 24)):  # Limit to first 24 columns to avoid huge CSV
                        col_headers.append(f"Col {j}")
                    if dof_count > 24:
                        col_headers.append("...")
                    writer.writerow(col_headers)
                    
                    # Write matrix rows
                    for i in range(min(dof_count, 24)):  # Limit to first 24 rows
                        row = [f"Row {i}"]
                        for j in range(min(dof_count, 24)):
                            row.append(f"{self.model.global_K[i, j]:.2e}")
                        if dof_count > 24:
                            row.append("...")
                        writer.writerow(row)
                    
                    if dof_count > 24:
                        writer.writerow(["...", "..."])
                        writer.writerow([f"... (Matrix truncated. Showing top-left 24x24 of {dof_count}x{dof_count}) ..."])
                    writer.writerow([])
                
                # Nodal Displacements & Reactions
                writer.writerow(["NODAL DISPLACEMENTS & REACTIONS"])
                writer.writerow(["Node", "Ux (mm)", "Uy (mm)", "Rot (rad)", "Rx (N)", "Ry (N)", "Mz (Nm)"])
                for n in self.model.nodes.values():
                    fix_icon = "🔒" if any(n.fixity) else ""
                    node_id_with_fix = f"{n.id} {fix_icon}"
                    writer.writerow([node_id_with_fix, f"{n.disp[0]*1000:.6f}", f"{n.disp[1]*1000:.6f}", f"{n.disp[2]:.8f}", f"{n.reaction[0]:.4f}", f"{n.reaction[1]:.4f}", f"{n.reaction[2]:.4f}"])
                writer.writerow([])
                
                # Member Forces
                writer.writerow(["MEMBER LOCAL FORCES"])
                writer.writerow(["Element", "Nodes", "Axial (N)", "Shear Start (N)", "Moment Start (Nm)", "Moment End (Nm)"])
                for el in self.model.elements.values():
                    f = el.local_forces
                    nodes = f"{el.node_i.id}-{el.node_j.id}"
                    writer.writerow([el.id, nodes, f"{f[3]:.2f}", f"{f[1]:.2f}", f"{f[2]:.2f}", f"{-f[5]:.2f}"])
                writer.writerow([])
                
                # Member Axial Status Table
                writer.writerow(["MEMBER AXIAL STATUS (TENSION/COMPRESSION)"])
                writer.writerow(["Element", "Nodes", "Force Magnitude (N)", "Status"])
                for el in self.model.elements.values():
                    axial_force = el.local_forces[3]
                    nodes = f"{el.node_i.id}-{el.node_j.id}"
                    
                    if axial_force > 1e-5:
                        status = "TENSION"
                    elif axial_force < -1e-5:
                        status = "COMPRESSION"
                    else:
                        status = "ZERO"
                    
                    writer.writerow([el.id, nodes, f"{abs(axial_force):.2f}", status])
                writer.writerow([])
                
                # Analysis Comments
                writer.writerow(["AUTOMATED ANALYSIS COMMENTS"])
                comments = []
                
                if len(self.model.nodes) < 2:
                    comments.append("❌ Error: Structure has fewer than 2 nodes. Add more for a valid frame.")
                if len(self.model.elements) == 0:
                    comments.append("❌ Error: No elements defined. Connect nodes with members.")
                
                supported_nodes = sum(1 for n in self.model.nodes.values() if any(n.fixity))
                if supported_nodes == 0:
                    comments.append("❌ Error: No supports defined. Structure is unstable—add fixities!")
                elif supported_nodes == 1:
                    comments.append("⚠️ Note: Only one support defined. May be unstable for frames; verify degrees of freedom.")
                else:
                    comments.append(f"✓ Structure has adequate supports ({supported_nodes} nodes fixed).")
                
                loaded_nodes = sum(1 for n in self.model.nodes.values() if any(abs(float(l)) > 1e-6 for l in n.loads))
                if loaded_nodes == 0:
                    comments.append("⚠️ Note: No external loads applied. All results will be zero—add loads for meaningful analysis.")
                else:
                    comments.append(f"✓ {loaded_nodes} node(s) have applied loads.")
                
                invalid_elements = [el for el in self.model.elements.values() if el.length < self.model.min_element_length or el.A <= self.model.min_section_area or el.I <= self.model.min_inertia]
                if invalid_elements:
                    comments.append(f"❌ Error: {len(invalid_elements)} invalid element(s) (zero-length, negative/zero area, or inertia). Check elements: " + ", ".join(str(el.id) for el in invalid_elements) + ". Fix sections or remove duplicates.")
                
                extreme_materials = [el for el in self.model.elements.values() if el.E < self.model.min_E or el.E > self.model.max_E]
                if extreme_materials:
                    comments.append(f"⚠️ Note: Unusual material stiffness (E) in {len(extreme_materials)} element(s). Typical steel E=200 GPa. Check elements: " + ", ".join(str(el.id) for el in extreme_materials) + ".")
                
                max_disp = max((max(abs(float(d)) for d in n.disp) for n in self.model.nodes.values()), default=0)
                max_element_L = max((el.length for el in self.model.elements.values()), default=1)
                if max_disp > self.model.max_deflection_ratio * max_element_L:
                    comments.append("❌ Warning: Large deflections detected (max δ = " + f"{max_disp:.4f}" + " > L/" + str(int(1/self.model.max_deflection_ratio)) + "). Small deformation assumption violated—consider nonlinear analysis or stiffen structure.")
                else:
                    comments.append("✓ Deflections within limits (small deformation assumption holds).")
                
                G = nx.Graph()
                for el in self.model.elements.values():
                    G.add_edge(el.node_i.id, el.node_j.id)
                connected_components = list(nx.connected_components(G))
                if len(connected_components) > 1:
                    comments.append(f"❌ Error: Disconnected structure ({len(connected_components)} separate components). Connect all parts or analyze separately.")
                elif len(connected_components) == 0:
                    comments.append("❌ Error: No connected components. Add elements.")
                else:
                    comments.append("✓ Structure is fully connected (single component).")
                
                fixed_dofs = sum(sum(n.fixity) for n in self.model.nodes.values())
                if fixed_dofs < 3:
                    comments.append(f"❌ Error: Insufficient constraints ({fixed_dofs} fixed DOFs). Need at least 3 for stability.")
                else:
                    redundancies = max(0, fixed_dofs - 3)
                    if redundancies == 0:
                        comments.append("✓ Structure appears statically determinate.")
                    else:
                        comments.append(f"⚠️ Note: Structure is statically indeterminate ({redundancies} redundancy). FEM suitable, but check for overconstraints.")
                
                max_load = max((max(abs(float(l)) for l in n.loads) for n in self.model.nodes.values()), default=0)
                if max_load > 1e6:
                    comments.append("⚠️ Note: Very large loads detected (max = " + f"{max_load:.2e}" + " N/Nm). Check units or scale; may cause numerical issues.")
                elif max_load < 1e-3 and max_load > 0:
                    comments.append("⚠️ Note: Very small loads detected. Results may be near zero—check units.")
                
                for comment in comments:
                    writer.writerow([comment])
            
            QMessageBox.information(self, "Export Complete", f"All analysis tables and information exported successfully to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON Files (*.json)")
        if path:
            with open(path, 'w') as f: f.write(self.model.to_json())

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "JSON Files (*.json)")
        if path:
            with open(path, 'r') as f: self.model.load_json(f.read())
            self.editor.fit_view()
            self.editor.redraw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())