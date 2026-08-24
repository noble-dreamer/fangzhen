"""Axisymmetric full-solid pipe cell for limited Shell dispersion calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


SIMPLE_ROOT = Path(__file__).resolve().parents[2]
if str(SIMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_ROOT))

import simple_shell_common as shell


STUDY_NAME = "axisymmetric solid dispersion eigenfrequency"
INTEGRATION_OPERATOR = "intop_solid"


@dataclass(frozen=True)
class SolidCellConfig:
    cell_length_mm: float = 5.0
    inner_radius_mm: float = 150.0
    thickness_mm: float = 10.0
    mesh_hmax_mm: float = 1.25
    mode_count: int = 4
    eigen_shift_hz: float = 60_000.0

    def validate(self) -> None:
        if any(value <= 0.0 for value in vars(self).values()):
            raise ValueError("Solid cell configuration values must be positive")
        if self.inner_radius_mm + self.thickness_mm <= 0.0:
            raise ValueError("Solid cell outer radius must be positive")


def _box(model, name: str, entity_dim: int, ymin: str, ymax: str):
    selection = (model / "selections").create("Box", name=name)
    selection.property("entitydim", entity_dim)
    selection.property("xmin", "Ri-0.01[mm]")
    selection.property("xmax", "Ri+h_cell+0.01[mm]")
    selection.property("ymin", ymin)
    selection.property("ymax", ymax)
    selection.property("condition", "inside")
    return selection


def _selections(model):
    domain = _box(model, "solid wall domain", 2, "-0.01[mm]", "L_cell+0.01[mm]")
    edge0 = _box(model, "periodic edge z0", 1, "-0.01[mm]", "0.01[mm]")
    edge1 = _box(model, "periodic edge zL", 1, "L_cell-0.01[mm]", "L_cell+0.01[mm]")
    ends = (model / "selections").create("Union", name="periodic axial edges")
    ends.property("entitydim", 1)
    ends.property("input", [edge0, edge1])
    return domain, ends


def _physics(model, geometry, domain, ends):
    physics = (model / "physics").create("SolidMechanics", geometry, name="solid mechanics")
    physics.select(domain)
    mode = physics.java.prop("Mode2Daxi")
    mode.set("ModeExtension", "1")
    mode.set("mk", "n_circ")
    elastic = [child for child in physics.children() if child.type() == "LinearElasticModel"]
    if len(elastic) != 1:
        raise RuntimeError(f"Expected one LinearElasticModel, found {len(elastic)}")
    for key, value in (
        ("E_mat", "userdef"),
        ("E", "E_al"),
        ("nu_mat", "userdef"),
        ("nu", "nu_al"),
        ("rho_mat", "userdef"),
        ("rho", "rho_al"),
    ):
        elastic[0].property(key, value)
    periodic = physics.create("PeriodicCondition", 1, name="axial Floquet periodicity")
    periodic.select(ends)
    periodic.property("PeriodicType", "Floquet")
    periodic.property("kFloquet", ["0", "0", "kz"])
    periodic.property("constraintMethod", "nodal")
    return physics, periodic


def build_solid_model(client, model_name: str, config: SolidCellConfig):
    config.validate()
    model = client.create(model_name)
    parameters = {
        "L_cell": f"{config.cell_length_mm:.12g}[mm]",
        "Ri": f"{config.inner_radius_mm:.12g}[mm]",
        "h_cell": f"{config.thickness_mm:.12g}[mm]",
        "mesh_hmax": f"{config.mesh_hmax_mm:.12g}[mm]",
        "E_al": "70[GPa]",
        "nu_al": "0.33",
        "rho_al": "2700[kg/m^3]",
        "kz": "0[1/m]",
        "n_circ": "0",
    }
    for name, value in parameters.items():
        model.parameter(name, value)
    (model / "components").create(True, name="component")
    geometry = (model / "geometries").create(2, name="axisymmetric solid cell")
    geometry.java.axisymmetric(True)
    geometry.java.lengthUnit("mm")
    wall = geometry.create("Rectangle", name="full solid wall")
    wall.property("size", ["h_cell", "L_cell"])
    wall.property("pos", ["Ri", "0"])
    model.build(geometry)
    domain, ends = _selections(model)
    _physics(model, geometry, domain, ends)
    component = model.java.component("comp1")
    component.cpl().create(INTEGRATION_OPERATOR, "Integration")
    component.cpl(INTEGRATION_OPERATOR).selection().named(domain.tag())
    mesh = (model / "meshes").create(geometry, name="solid cell mesh")
    size = mesh.create("Size", name="solid calibration mesh size")
    size.select(domain)
    size.property("custom", "on")
    size.property("hmax", "mesh_hmax")
    size.property("hmin", "mesh_hmax/5")
    triangle = mesh.create("FreeTri", name="free triangular solid mesh")
    triangle.select(domain)
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


def solve_modes(model, kz_rad_m: float, circumferential_order: int) -> tuple[np.ndarray, np.ndarray]:
    model.parameter("kz", f"{kz_rad_m:.12g}[1/m]")
    model.parameter("n_circ", str(int(circumferential_order)))
    model.solve(STUDY_NAME)
    expressions = ["freq", *(f"{INTEGRATION_OPERATOR}(abs({component})^2)" for component in "uvw")]
    values = np.asarray(model.evaluate(expressions), dtype=complex)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid solid eigenmode result shape/content: {values.shape}")
    frequency = values[:, 0]
    energy = np.maximum(values[:, 1:].real, 0.0)
    valid = (frequency.real > 0.0) & (energy.sum(axis=1) > 0.0)
    frequency, energy = frequency[valid], energy[valid]
    polarization = energy / energy.sum(axis=1, keepdims=True)
    order = np.argsort(frequency.real)
    return frequency[order], polarization[order]


def validate_model_tree(model) -> dict[str, object]:
    physics = model / "physics" / "solid mechanics"
    periodic = physics / "axial Floquet periodicity"
    mode = physics.java.prop("Mode2Daxi")
    return {
        "axisymmetric": bool((model / "geometries" / "axisymmetric solid cell").java.isAxisymmetric()),
        "mode_extension": mode.getString("ModeExtension"),
        "mode_parameter": mode.getString("mk"),
        "periodic_type": periodic.property("PeriodicType"),
        "k_floquet": list(periodic.property("kFloquet")),
        "problems": model.problems(),
    }
