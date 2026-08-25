"""Axisymmetric Shell cell with prescribed circumferential and axial wavenumbers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


SIMPLE_ROOT = Path(__file__).resolve().parents[2]
if str(SIMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_ROOT))

import simple_shell_common as shell


STUDY_NAME = "axisymmetric pipe dispersion eigenfrequency"
INTEGRATION_OPERATOR = "intop_axi"


@dataclass(frozen=True)
class AxisymmetricCellConfig:
    cell_length_mm: float = 5.0
    mid_radius_mm: float = 155.0
    reference_thickness_mm: float = 10.0
    initial_thickness_mm: float = 10.0
    support_width_mm: float = 1.0
    mesh_hmax_mm: float = 1.25
    mode_count: int = 8
    eigen_shift_hz: float = 60_000.0

    def validate(self) -> None:
        if any(value <= 0 for value in vars(self).values()):
            raise ValueError("Axisymmetric cell configuration values must be positive")


def _box_selection(model, name: str, entity_dim: int, ymin: str, ymax: str):
    selection = (model / "selections").create("Box", name=name)
    selection.property("entitydim", entity_dim)
    selection.property("xmin", "R_mid-0.01[mm]")
    selection.property("xmax", "R_mid+0.01[mm]")
    selection.property("ymin", ymin)
    selection.property("ymax", ymax)
    selection.property("condition", "inside")
    return selection


def _create_selections(model):
    side = _box_selection(model, "axisymmetric shell midsurface", 1, "-0.01[mm]", "L_cell+0.01[mm]")
    point0 = _box_selection(model, "periodic point z0", 0, "-0.01[mm]", "0.01[mm]")
    point1 = _box_selection(model, "periodic point zL", 0, "L_cell-0.01[mm]", "L_cell+0.01[mm]")
    endpoints = (model / "selections").create("Union", name="periodic axial endpoints")
    endpoints.property("entitydim", 0)
    endpoints.property("input", [point0, point1])
    return side, endpoints


def _create_physics(model, geometry, side, endpoints):
    physics = (model / "physics").create("Shell", geometry, name="axisymmetric shell mechanics")
    physics.select(side)
    mode_extension = physics.java.prop("Mode2Daxi")
    mode_extension.set("ModeExtension", "1")
    mode_extension.set("mk", "n_circ")
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
        raise RuntimeError(f"Unexpected axisymmetric Shell defaults: {sorted(found)}")
    periodic = physics.create("PeriodicCondition", 0, name="axial Floquet periodicity")
    periodic.select(endpoints)
    periodic.property("PeriodicType", "Floquet")
    periodic.property("kFloquet", ["0", "0", "kz"])
    periodic.property("constraintMethod", "nodal")
    return physics, periodic


def build_axisymmetric_model(client, model_name: str, config: AxisymmetricCellConfig):
    config.validate()
    model = client.create(model_name)
    parameters = {
        "L_cell": f"{config.cell_length_mm:.12g}[mm]",
        "R_mid": f"{config.mid_radius_mm:.12g}[mm]",
        "h_ref": f"{config.reference_thickness_mm:.12g}[mm]",
        "h_cell": f"{config.initial_thickness_mm:.12g}[mm]",
        "support_width": f"{config.support_width_mm:.12g}[mm]",
        "mesh_hmax": f"{config.mesh_hmax_mm:.12g}[mm]",
        "rho_al": "2700[kg/m^3]",
        "E_al": "70[GPa]",
        "nu_al": "0.33",
        "n_circ": "0",
        "kz": "0[1/m]",
    }
    for name, value in parameters.items():
        model.parameter(name, value)
    (model / "components").create(True, name="component")
    geometry = (model / "geometries").create(2, name="axisymmetric periodic cell")
    geometry.java.axisymmetric(True)
    geometry.java.lengthUnit("mm")
    support = geometry.create("Rectangle", name="selection support strip")
    support.property("size", ["support_width", "L_cell"])
    support.property("pos", ["R_mid", "0"])
    model.build(geometry)
    side, endpoints = _create_selections(model)
    _create_physics(model, geometry, side, endpoints)

    component = model.java.component("comp1")
    component.cpl().create(INTEGRATION_OPERATOR, "Integration")
    operator = component.cpl(INTEGRATION_OPERATOR)
    operator.label("axisymmetric Shell modal integration")
    operator.selection().named(side.tag())

    mesh = (model / "meshes").create(geometry, name="axisymmetric cell mesh")
    size = mesh.create("Size", name="dispersion mesh size")
    size.property("custom", "on")
    size.property("hmax", "mesh_hmax")
    size.property("hmin", "mesh_hmax/5")
    triangle = mesh.create("FreeTri", name="free triangular support mesh")
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


def solve_axisymmetric_modes(
    model, thickness_mm: float, kz_rad_m: float, circumferential_order: int
) -> tuple[np.ndarray, np.ndarray]:
    model.parameter("h_cell", f"{thickness_mm:.12g}[mm]")
    model.parameter("kz", f"{kz_rad_m:.12g}[1/m]")
    model.parameter("n_circ", str(int(circumferential_order)))
    model.solve(STUDY_NAME)
    expressions = ["freq", "intop_axi(abs(u)^2)", "intop_axi(abs(v)^2)", "intop_axi(abs(w)^2)"]
    values = np.asarray(model.evaluate(expressions), dtype=complex)
    if values.ndim != 2 or values.shape[1] != len(expressions):
        raise RuntimeError(f"Unexpected axisymmetric modal result shape {values.shape}")
    frequency = values[:, 0]
    energy = np.maximum(values[:, 1:].real, 0.0)
    valid = np.isfinite(frequency) & (frequency.real > 0.0) & (energy.sum(axis=1) > 0.0)
    frequency = frequency[valid]
    polarization = energy[valid] / energy[valid].sum(axis=1, keepdims=True)
    order = np.argsort(frequency.real)
    return frequency[order], polarization[order]


def validate_model_tree(model) -> dict[str, object]:
    physics = model / "physics" / "axisymmetric shell mechanics"
    periodic = physics / "axial Floquet periodicity"
    mode = physics.java.prop("Mode2Daxi")
    return {
        "axisymmetric": bool((model / "geometries" / "axisymmetric periodic cell").java.isAxisymmetric()),
        "mode_extension": mode.getString("ModeExtension"),
        "mode_parameter": mode.getString("mk"),
        "periodic_type": periodic.property("PeriodicType"),
        "k_floquet": list(periodic.property("kFloquet")),
        "problems": model.problems(),
    }
