"""Periodic Shell unit-cell builder for axial pipe dispersion solves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


SIMPLE_ROOT = Path(__file__).resolve().parents[2]
if str(SIMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_ROOT))

import simple_shell_common as shell


STUDY_NAME = "pipe dispersion eigenfrequency"
MODE_INTEGRATION_OPERATOR = "intop_disp"
MODE_EDGE_OPERATOR = "intop_edge"


@dataclass(frozen=True)
class DispersionCellConfig:
    cell_length_mm: float = 5.0
    reference_mid_radius_mm: float = 155.0
    reference_wall_thickness_mm: float = 10.0
    initial_thickness_mm: float = 10.0
    max_frequency_hz: float = 110_000.0
    min_wave_speed_m_s: float = 2_500.0
    elements_per_wavelength: float = 8.0
    axial_elements: int = 4
    mode_count: int = 24
    eigen_shift_hz: float = 60_000.0

    @property
    def mesh_hmax_mm(self) -> float:
        wave_hmax = self.min_wave_speed_m_s / self.max_frequency_hz * 1000.0
        wave_hmax /= self.elements_per_wavelength
        return min(wave_hmax, self.cell_length_mm / self.axial_elements)

    def validate(self) -> None:
        positive = (
            self.cell_length_mm,
            self.reference_mid_radius_mm,
            self.reference_wall_thickness_mm,
            self.initial_thickness_mm,
            self.max_frequency_hz,
            self.min_wave_speed_m_s,
            self.elements_per_wavelength,
            self.axial_elements,
            self.mode_count,
            self.eigen_shift_hz,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Dispersion cell configuration values must be positive")


def _box_selection(model, name: str, entity_dim: int, zmin: str, zmax: str):
    selection = (model / "selections").create("Box", name=name)
    selection.property("entitydim", entity_dim)
    selection.property("xmin", "-Rm-1[mm]")
    selection.property("xmax", "Rm+1[mm]")
    selection.property("ymin", "-Rm-1[mm]")
    selection.property("ymax", "Rm+1[mm]")
    selection.property("zmin", zmin)
    selection.property("zmax", zmax)
    selection.property("condition", "inside")
    return selection


def _create_selections(model):
    cap0 = _box_selection(model, "cell cap z0", 2, "-0.01[mm]", "0.01[mm]")
    cap1 = _box_selection(model, "cell cap zL", 2, "L_cell-0.01[mm]", "L_cell+0.01[mm]")
    caps = (model / "selections").create("Union", name="cell caps")
    caps.property("entitydim", 2)
    caps.property("input", [cap0, cap1])
    side = (model / "selections").create("Complement", name="cell shell surface")
    side.property("entitydim", 2)
    side.property("input", [caps])

    edge0 = _box_selection(model, "periodic edge z0", 1, "-0.01[mm]", "0.01[mm]")
    edge1 = _box_selection(model, "periodic edge zL", 1, "L_cell-0.01[mm]", "L_cell+0.01[mm]")
    edges = (model / "selections").create("Union", name="periodic end edges")
    edges.property("entitydim", 1)
    edges.property("input", [edge0, edge1])
    return side, edges, edge0


def _create_shell_physics(model, geometry, side, edges):
    physics = (model / "physics").create("Shell", geometry, name="shell mechanics")
    physics.select(side)
    found = set()
    for child in physics.children():
        if child.type() == "Elastic":
            found.add("Elastic")
            child.property("E_mat", "userdef")
            child.property("E", "E_al")
            child.property("nu_mat", "userdef")
            child.property("nu", "nu_al")
            child.property("rho_mat", "userdef")
            child.property("rho", "rho_al")
        elif child.type() == "ThicknessOffset":
            found.add("ThicknessOffset")
            child.property("d", "h_cell")
            child.property("OffsetDefinition", "RelativeDistance")
            child.property("z_offset_rel", "-(h_ref-h_cell)/h_cell")
    if found != {"Elastic", "ThicknessOffset"}:
        raise RuntimeError(f"Unexpected Shell default nodes: found {sorted(found)}")

    periodic = physics.create("PeriodicCondition", 1, name="axial Floquet periodicity")
    periodic.select(edges)
    periodic.property("PeriodicType", "Floquet")
    periodic.property("kFloquet", ["0", "0", "kz"])
    periodic.property("constraintMethod", "nodal")
    return physics, periodic


def _create_mode_integration_operator(model, side, edge0) -> None:
    component = model.java.component("comp1")
    coupling = component.cpl()
    for tag, label, selection in (
        (MODE_INTEGRATION_OPERATOR, "dispersion mode shell integration", side),
        (MODE_EDGE_OPERATOR, "dispersion source-edge integration", edge0),
    ):
        coupling.create(tag, "Integration")
        operator = component.cpl(tag)
        operator.label(label)
        operator.selection().named(selection.tag())


def build_periodic_shell_model(client, model_name: str, config: DispersionCellConfig):
    config.validate()
    model = client.create(model_name)
    parameters = {
        "L_cell": f"{config.cell_length_mm:.12g}[mm]",
        "Rm": f"{config.reference_mid_radius_mm:.12g}[mm]",
        "h_ref": f"{config.reference_wall_thickness_mm:.12g}[mm]",
        "h_cell": f"{config.initial_thickness_mm:.12g}[mm]",
        "rho_al": "2700[kg/m^3]",
        "E_al": "70[GPa]",
        "nu_al": "0.33",
        "kz": "0[1/m]",
        "mesh_hmax": f"{config.mesh_hmax_mm:.12g}[mm]",
    }
    for name, value in parameters.items():
        model.parameter(name, value)

    (model / "components").create(True, name="component")
    geometry = (model / "geometries").create(3, name="periodic pipe cell")
    geometry.java.lengthUnit("mm")
    cylinder = geometry.create("Cylinder", name="reference midsurface cylinder")
    cylinder.property("r", "Rm")
    cylinder.property("h", "L_cell")
    cylinder.property("selresult", "on")
    model.build(geometry)

    side, edges, edge0 = _create_selections(model)
    _create_shell_physics(model, geometry, side, edges)
    _create_mode_integration_operator(model, side, edge0)
    mesh = (model / "meshes").create(geometry, name="periodic shell mesh")
    size = mesh.create("Size", name="dispersion mesh size")
    size.select(side)
    size.property("custom", "on")
    size.property("hmax", "mesh_hmax")
    size.property("hmin", "mesh_hmax/5")
    triangle = mesh.create("FreeTri", name="free triangular shell mesh")
    triangle.select(side)
    model.mesh(mesh)

    study = (model / "studies").create(name=STUDY_NAME)
    study.java.setGenPlots(False)
    study.java.setGenConv(False)
    eigen = study.create("Eigenfrequency", name="eigenfrequency near target band")
    eigen.property("neigsactive", True)
    eigen.property("neigs", config.mode_count)
    eigen.property("shift", f"{config.eigen_shift_hz:.12g}[Hz]")
    study.java.createAutoSequences("sol")
    return model


def solve_eigenfrequencies(model, thickness_mm: float, kz_rad_m: float) -> np.ndarray:
    model.parameter("h_cell", f"{thickness_mm:.12g}[mm]")
    model.parameter("kz", f"{kz_rad_m:.12g}[1/m]")
    model.solve(STUDY_NAME)
    values = np.atleast_1d(np.asarray(model.evaluate("freq", unit="Hz"), dtype=complex))
    valid = np.isfinite(values.real) & np.isfinite(values.imag) & (values.real > 0.0)
    values = values[valid]
    return values[np.argsort(values.real)]


def validate_model_tree(model) -> dict[str, object]:
    periodic = model / "physics" / "shell mechanics" / "axial Floquet periodicity"
    return {
        "periodic_type": periodic.property("PeriodicType"),
        "k_floquet": list(periodic.property("kFloquet")),
        "periodic_selection": str(periodic.selection()),
        "problems": model.problems(),
    }
