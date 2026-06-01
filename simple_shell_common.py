"""Shared COMSOL shell-model builder for lightweight guided-wave datasets.

The simplified model removes all PZT solid domains. Transducers are represented
by smooth equivalent face-load windows on a cylindrical shell midsurface, and
receivers are exported from point datasets on the receiver ring.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import mph


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / 'output'


@dataclass(frozen=True)
class PipeConfig:
    length_mm: float = 1000.0
    outer_radius_mm: float = 160.0
    inner_radius_mm: float = 150.0

    @property
    def mid_radius_mm(self) -> float:
        return 0.5 * (self.outer_radius_mm + self.inner_radius_mm)

    @property
    def wall_thickness_mm(self) -> float:
        return self.outer_radius_mm - self.inner_radius_mm


@dataclass(frozen=True)
class TransducerConfig:
    count_per_ring: int = 16
    tx_z_mm: float = 100.0
    rx_z_mm: float = 900.0
    patch_width_mm: float = 6.0
    patch_length_mm: float = 27.0
    total_force_n: float = 1.0
    center_frequency_hz: float = 50_000.0
    cycles: int = 5


@dataclass(frozen=True)
class MaterialConfig:
    density_kg_m3: float = 2700.0
    young_gpa: float = 70.0
    poisson: float = 0.33
    rayleigh_alpha_1_s: float = 0.0
    rayleigh_beta_s: float = 5e-8


@dataclass(frozen=True)
class AbsorbingLayerConfig:
    enabled: bool = False
    length_mm: float = 75.0
    rayleigh_beta_s: float = 2e-6
    exponent: float = 2.0


@dataclass(frozen=True)
class DefectModelConfig:
    corrosion_surface: str = 'outer'


@dataclass(frozen=True)
class SolverConfig:
    t_end_ms: float = 0.8
    dt_out_us: float = 0.7
    relative_tolerance: float = 1e-4
    direct_linear_solver: str = 'pardiso'
    cudss_precision: str = 'single'
    cudss_hybrid_memory: str = 'auto'
    cudss_use_hybrid_compute: bool = False
    cudss_use_multiple_gpus: bool = False
    pardiso_use_cluster: bool = False
    solve: bool = False


@dataclass(frozen=True)
class MeshConfig:
    min_wave_speed_m_s: float = 2500.0
    max_frequency_hz: float = 60_000.0
    elements_per_wavelength: float = 8.0
    hmin_mm: float = 2.0
    build_mesh: bool = True

    @property
    def hmax_mm(self) -> float:
        wavelength_mm = self.min_wave_speed_m_s / self.max_frequency_hz * 1000.0
        return wavelength_mm / self.elements_per_wavelength


@dataclass(frozen=True)
class SweepConfig:
    transmitter_indices: tuple[int, ...] = tuple(range(1, 17))
    frequencies_hz: tuple[float, ...] = (40_000.0, 50_000.0, 60_000.0)
    use_parametric_sweep: bool = True


@dataclass(frozen=True)
class DefectConfig:
    theta_deg: float
    z_mm: float
    radius_mm: float
    depth_mm: float
    radius_theta_mm: float | None = None
    radius_z_mm: float | None = None


@dataclass(frozen=True)
class DefectLobeConfig:
    parent_index: int
    theta_deg: float
    z_mm: float
    radius_mm: float
    depth_mm: float
    radius_theta_mm: float | None = None
    radius_z_mm: float | None = None


PIPE = PipeConfig()
TRANSDUCER = TransducerConfig()
MATERIAL = MaterialConfig()
ABSORBING_LAYER = AbsorbingLayerConfig()
DEFECT_MODEL = DefectModelConfig()
SOLVER = SolverConfig()
MESH = MeshConfig()
SWEEP = SweepConfig()
RECEIVER_INDICES: tuple[int, ...] = tuple(range(17, 33))
POSITION_PERTURBATIONS: dict[int, dict[str, float]] = {}
AMPLITUDE_SCALE: dict[int, float] = {}
MODEL_FAMILY = 'simple shell guided-wave model'
DATASET_NOTES: list[str] = []
CREATE_RECEIVER_DATASETS = True
CREATE_VISUAL_MARKER_DATASETS = True


def set_if_possible(node, name: str, value) -> bool:
    try:
        node.property(name, value)
        return True
    except Exception:
        return False


def transducer_positions() -> list[dict]:
    step = 360.0 / TRANSDUCER.count_per_ring
    positions: list[dict] = []
    index = 1
    for ring, z_mm in [('tx', TRANSDUCER.tx_z_mm), ('rx', TRANSDUCER.rx_z_mm)]:
        for n in range(TRANSDUCER.count_per_ring):
            perturb = POSITION_PERTURBATIONS.get(index, {})
            theta_deg = n * step + perturb.get('dtheta_deg', 0.0)
            actual_z_mm = z_mm + perturb.get('dz_mm', 0.0)
            theta = math.radians(theta_deg)
            positions.append({
                'index': index,
                'ring': ring,
                'n': n,
                'base_theta_deg': n * step,
                'theta_deg': theta_deg,
                'x_mm': PIPE.mid_radius_mm * math.cos(theta),
                'y_mm': PIPE.mid_radius_mm * math.sin(theta),
                'z_mm': actual_z_mm,
                'dz_mm': perturb.get('dz_mm', 0.0),
                'dtheta_deg': perturb.get('dtheta_deg', 0.0),
                'amplitude_scale': AMPLITUDE_SCALE.get(index, 1.0),
            })
            index += 1
    return positions


def transmitter_positions() -> list[dict]:
    allowed = set(SWEEP.transmitter_indices)
    return [item for item in transducer_positions() if item['index'] in allowed]


def receiver_positions() -> list[dict]:
    allowed = set(RECEIVER_INDICES)
    return [item for item in transducer_positions() if item['index'] in allowed]


def add_parameters(model, damaged: bool) -> None:
    params = {
        'L_pipe': (f'{PIPE.length_mm:.9g}[mm]', 'Pipe length'),
        'Ro': (f'{PIPE.outer_radius_mm:.9g}[mm]', 'Outer radius, kept for metadata'),
        'Ri': (f'{PIPE.inner_radius_mm:.9g}[mm]', 'Inner radius, kept for metadata'),
        'Rm': (f'{PIPE.mid_radius_mm:.9g}[mm]', 'Shell midsurface radius'),
        'h0': (f'{PIPE.wall_thickness_mm:.9g}[mm]', 'Nominal shell thickness'),
        'h_min': ('1[mm]', 'Minimum shell thickness used in defect expressions'),
        'rho_al': (f'{MATERIAL.density_kg_m3:.9g}[kg/m^3]', 'Aluminum density'),
        'E_al': (f'{MATERIAL.young_gpa:.9g}[GPa]', 'Aluminum Young modulus'),
        'nu_al': (f'{MATERIAL.poisson:.9g}', 'Aluminum Poisson ratio'),
        'rayleigh_alpha': (f'{MATERIAL.rayleigh_alpha_1_s:.9g}[1/s]', 'Rayleigh mass damping'),
        'rayleigh_beta': (f'{MATERIAL.rayleigh_beta_s:.9g}[s]', 'Rayleigh stiffness damping'),
        'absorb_enabled': ('1' if ABSORBING_LAYER.enabled else '0', '1 when shell end absorbing layers are active'),
        'absorb_len': (f'{ABSORBING_LAYER.length_mm:.9g}[mm]', 'Axial length of each shell absorbing end layer'),
        'absorb_beta': (f'{ABSORBING_LAYER.rayleigh_beta_s:.9g}[s]', 'Added Rayleigh beta at the pipe ends'),
        'absorb_exp': (f'{ABSORBING_LAYER.exponent:.9g}', 'Power-law ramp exponent for absorbing end layers'),
        'pzt_w': (f'{TRANSDUCER.patch_width_mm:.9g}[mm]', 'Equivalent transducer circumferential width'),
        'pzt_l': (f'{TRANSDUCER.patch_length_mm:.9g}[mm]', 'Equivalent transducer axial length'),
        'pzt_A': ('pzt_w*pzt_l', 'Equivalent transducer patch area'),
        'F0': (f'{TRANSDUCER.total_force_n:.9g}[N]', 'Equivalent total force amplitude'),
        'pzt_fc': (f'{TRANSDUCER.center_frequency_hz:.9g}[Hz]', 'Excitation center frequency'),
        'pzt_cycles': (f'{TRANSDUCER.cycles}', 'Hanning-windowed sine cycles'),
        'tx': (f'{SWEEP.transmitter_indices[0]}', 'Active transmitter index'),
        't_end': (f'{SOLVER.t_end_ms:.9g}[ms]', 'Transient end time'),
        'dt_out': (f'{SOLVER.dt_out_us:.9g}[us]', 'Output time step'),
        'mesh_hmax': (f'{MESH.hmax_mm:.9g}[mm]', 'Shell mesh target from wavelength rule'),
        'mesh_hmin': (f'{MESH.hmin_mm:.9g}[mm]', 'Shell mesh minimum size'),
        'defect_enabled': ('1' if damaged else '0', '1 when thickness-loss defects are active'),
    }
    for key, (value, description) in params.items():
        model.parameter(key, value)
        model.description(key, description)


def create_geometry(model):
    (model / 'components').create(True, name='component')
    geometry = (model / 'geometries').create(3, name='shell midsurface geometry')
    geometry.java.lengthUnit('mm')
    support = geometry.create('Cylinder', name='pipe midsurface support cylinder')
    support.property('r', 'Rm')
    support.property('h', 'L_pipe')
    support.property('pos', ['0', '0', '0'])
    support.property('axis', ['0', '0', '1'])
    support.property('selresult', 'on')
    support.property('selresultshow', 'bnd')
    model.build(geometry)
    return geometry


def create_shell_side_selection(model):
    selections = model / 'selections'
    z0 = selections.create('Box', name='open pipe end z0 cap boundaries')
    z0.property('entitydim', 2)
    z0.property('xmin', '-Rm-1[mm]')
    z0.property('xmax', 'Rm+1[mm]')
    z0.property('ymin', '-Rm-1[mm]')
    z0.property('ymax', 'Rm+1[mm]')
    z0.property('zmin', '-0.1[mm]')
    z0.property('zmax', '0.1[mm]')
    z0.property('condition', 'inside')

    zL = selections.create('Box', name='open pipe end zL cap boundaries')
    zL.property('entitydim', 2)
    zL.property('xmin', '-Rm-1[mm]')
    zL.property('xmax', 'Rm+1[mm]')
    zL.property('ymin', '-Rm-1[mm]')
    zL.property('ymax', 'Rm+1[mm]')
    zL.property('zmin', 'L_pipe-0.1[mm]')
    zL.property('zmax', 'L_pipe+0.1[mm]')
    zL.property('condition', 'inside')

    caps = selections.create('Union', name='open pipe cap boundaries')
    caps.property('entitydim', 2)
    caps.property('input', [z0, zL])

    side = selections.create('Complement', name='open pipe cylindrical shell boundary')
    side.property('entitydim', 2)
    side.property('input', [caps])
    return side


def create_material(model) -> None:
    material = (model / 'materials').create('Common', name='aluminum shell material')
    material.comment(
        'Reference material data. The Shell elastic node also uses E_al, '
        'nu_al, and rho_al explicitly so the shell solve does not depend on '
        'boundary-to-domain material lookup.'
    )
    group = material.java.propertyGroup('def')
    group.set('density', 'rho_al')
    group.set('youngsmodulus', 'E_al')
    group.set('poissonsratio', 'nu_al')


def angular_offset_expr(theta_deg: float) -> str:
    theta = math.radians(theta_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    return f'atan2(({(-s):.12g})*x+({c:.12g})*y,({c:.12g})*x+({s:.12g})*y)'


def patch_window_expr(theta_deg: float, z_mm: float, width_name: str = 'pzt_w', length_name: str = 'pzt_l') -> str:
    ds = f'Rm*({angular_offset_expr(theta_deg)})'
    dz = f'(z-{z_mm:.9g}[mm])'
    return f'exp(-(({ds})/({width_name}/2))^8-(({dz})/({length_name}/2))^8)'


def defect_window_expr(defect: DefectConfig | DefectLobeConfig) -> str:
    ds = f'Rm*({angular_offset_expr(defect.theta_deg)})'
    dz = f'(z-{defect.z_mm:.9g}[mm])'
    radius_theta_mm = defect.radius_theta_mm or defect.radius_mm
    radius_z_mm = defect.radius_z_mm or defect.radius_mm
    rt = f'{radius_theta_mm:.9g}[mm]'
    rz = f'{radius_z_mm:.9g}[mm]'
    return f'exp(-((({ds})/({rt}))^2+(({dz})/({rz}))^2)^4)'


def thickness_loss_expression(defects: list[DefectConfig], lobes: list[DefectLobeConfig]) -> str:
    if not defects and not lobes:
        return '0[mm]'
    loss_terms = [
        f'({defect.depth_mm:.9g}[mm])*({defect_window_expr(defect)})'
        for defect in defects
    ]
    loss_terms.extend(
        f'({lobe.depth_mm:.9g}[mm])*({defect_window_expr(lobe)})'
        for lobe in lobes
    )
    return '+'.join(loss_terms)


def thickness_expression(defects: list[DefectConfig], lobes: list[DefectLobeConfig]) -> str:
    if not defects and not lobes:
        return 'h0'
    return f'max(h_min,h0-({thickness_loss_expression(defects, lobes)}))'


def shell_offset_relative_expression(thickness: str) -> str:
    surface = DEFECT_MODEL.corrosion_surface.lower()
    if surface == 'outer':
        return f'-((h0-({thickness}))/({thickness}))'
    if surface == 'inner':
        return f'((h0-({thickness}))/({thickness}))'
    return '0'


def rayleigh_beta_expression() -> str:
    if not ABSORBING_LAYER.enabled:
        return 'rayleigh_beta'
    left = 'if(z<absorb_len, ((absorb_len-z)/absorb_len)^absorb_exp, 0)'
    right = 'if(z>L_pipe-absorb_len, ((z-(L_pipe-absorb_len))/absorb_len)^absorb_exp, 0)'
    profile = f'min(1,({left})+({right}))'
    return f'rayleigh_beta + if(absorb_enabled>0.5, absorb_beta*({profile}), 0)'


def load_vector_expression() -> list[str]:
    x_terms: list[str] = []
    y_terms: list[str] = []
    for item in transmitter_positions():
        theta = math.radians(item['theta_deg'])
        gate = f'if(tx=={item["index"]},1,0)'
        window = patch_window_expr(item['theta_deg'], item['z_mm'])
        scale = f'{item["amplitude_scale"]:.9g}*{gate}*F0/pzt_A*pztpulse(t)*({window})'
        x_terms.append(f'({math.cos(theta):.12g})*({scale})')
        y_terms.append(f'({math.sin(theta):.12g})*({scale})')
    return ['+'.join(x_terms) or '0', '+'.join(y_terms) or '0', '0']


def create_functions(model) -> None:
    pulse = (model / 'functions').create('Analytic', name='five-cycle Hanning sine')
    pulse.property('funcname', 'pztpulse')
    pulse.property('args', 't')
    pulse.property(
        'expr',
        'if(t<=pzt_cycles/pzt_fc, '
        'sin(2*pi*pzt_fc*t)*0.5*(1-cos(2*pi*t/(pzt_cycles/pzt_fc))), 0)',
    )
    set_if_possible(pulse, 'argunit', 's')
    set_if_possible(pulse, 'fununit', '1')
    pulse.property('plotargs', ['t', '0', '150[us]'])


def create_shell_physics(model, geometry, shell_selection, defects: list[DefectConfig], lobes: list[DefectLobeConfig]):
    shell = (model / 'physics').create('Shell', geometry, name='shell mechanics')
    shell.select(shell_selection)
    for child in shell.children():
        if child.type() == 'Elastic':
            try:
                child.rename('explicit aluminum shell elastic material')
            except Exception:
                pass
            set_if_possible(child, 'E_mat', 'userdef')
            set_if_possible(child, 'E', 'E_al')
            set_if_possible(child, 'nu_mat', 'userdef')
            set_if_possible(child, 'nu', 'nu_al')
            set_if_possible(child, 'rho_mat', 'userdef')
            set_if_possible(child, 'rho', 'rho_al')
            child.comment(
                'Uses explicit global parameters E_al, nu_al, and rho_al. '
                'This avoids missing-material-property errors when Shell is '
                'applied only to the open cylindrical boundary.'
            )
            try:
                damping = child.create('Damping', name='light Rayleigh damping')
                set_if_possible(damping, 'DampingType', 'Rayleigh')
                set_if_possible(damping, 'alpha_dM', 'rayleigh_alpha')
                set_if_possible(damping, 'beta_dK', rayleigh_beta_expression())
            except Exception:
                pass
        elif child.type() == 'ThicknessOffset':
            try:
                child.rename('shell thickness and defect wall loss')
            except Exception:
                pass
            thickness = thickness_expression(defects, lobes)
            set_if_possible(child, 'd', thickness)
            if defects or lobes:
                offset = shell_offset_relative_expression(thickness)
                if offset != '0':
                    set_if_possible(child, 'OffsetDefinition', 'RelativeDistance')
                    set_if_possible(child, 'z_offset_rel', offset)
            else:
                set_if_possible(child, 'OffsetDefinition', 'NoOffset')
                set_if_possible(child, 'z_offset_rel', '0')
            child.comment(
                'Shell thickness. Healthy models use h0. Damaged models use '
                f'the thickness-loss expression: {thickness}. '
                f'corrosion_surface={DEFECT_MODEL.corrosion_surface}.'
            )

    load = shell.create('FaceLoad', 2, name='equivalent transducer face load')
    load.select(shell_selection)
    load_vector = load_vector_expression()
    set_if_possible(load, 'forceType', 'ForceArea')
    set_if_possible(load, 'forceReferenceArea', load_vector)
    # Older COMSOL/MPh combinations accepted these field names.  Keep them as
    # compatibility writes, but COMSOL 6.4 Shell ForceArea displays/evaluates
    # forceReferenceArea in the GUI as f_A.
    # set_if_possible(load, 'LoadTypeForce', 'ForceAreaFace')
    # set_if_possible(load, 'F', load_vector)
    load.comment(
        'Equivalent transducer load. Smooth spatial windows replace PZT solid '
        'domains, so mesh size is controlled by wavelength rather than PZT size. '
        'The active transmitter is selected by global parameter tx; the window '
        'centers are listed in the generated metadata and build log.'
    )
    return shell


def create_mesh(model, geometry, shell_selection):
    mesh = (model / 'meshes').create(geometry, name='wavelength controlled shell mesh')
    size = mesh.create('Size', name='shell wavelength size')
    size.select(shell_selection)
    size.property('custom', 'on')
    size.property('hmax', 'mesh_hmax')
    size.property('hmin', 'mesh_hmin')
    size.comment(
        'hmax = min_wave_speed / max_frequency / elements_per_wavelength. '
        'No PZT-specific local refinement is used.'
    )
    tri = mesh.create('FreeTri', name='free triangular shell mesh')
    tri.select(shell_selection)
    if MESH.build_mesh:
        model.mesh(mesh)
    return mesh


def create_study(model):
    study = (model / 'studies').create(name='simple shell displacement transient')
    study.java.setGenPlots(False)
    study.java.setGenConv(False)
    step = study.create('Transient', name='time dependent')
    step.property('tlist', 'range(0, dt_out, t_end)')
    step.property('rtol', SOLVER.relative_tolerance)
    if SWEEP.use_parametric_sweep:
        parametric = study.create('Parametric', name='tx and frequency sweep')
        parametric.property('pname', ['tx', 'pzt_fc'])
        parametric.property('sweeptype', 'filled')
        parametric.property('plistarr', [
            ' '.join(str(index) for index in SWEEP.transmitter_indices),
            ' '.join(f'{freq:.9g}[Hz]' for freq in SWEEP.frequencies_hz),
        ])
        parametric.property('punit', ['', 'Hz'])
    study.java.createAutoSequences('sol')
    tune_solver(model)
    return study


def tune_solver(model) -> None:
    for solution in model / 'solutions':
        try:
            solution.rename('simple shell displacement solution')
        except Exception:
            pass
        for feature in walk_solver_features(solution):
            if feature.type() == 'Time':
                for name, value in [
                    ('tlist', 'range(0, dt_out, t_end)'),
                    ('rtol', SOLVER.relative_tolerance),
                    ('timemethod', 'genalpha'),
                    ('tstepsgenalpha', 'strict'),
                    ('maxstepconstraintgenalpha', 'expr'),
                    ('maxstepexpressiongenalpha', 'dt_out'),
                    ('tstepsstore', 1),
                    ('plot', 'off'),
                    ('probefreq', 'tsteps'),
                ]:
                    set_if_possible(feature, name, value)
            elif feature.type() == 'Direct':
                configure_direct_solver(feature)


def walk_solver_features(node):
    for child in node.children():
        yield child
        yield from walk_solver_features(child)


def configure_direct_solver(feature) -> None:
    solver_name = SOLVER.direct_linear_solver.lower()
    if solver_name:
        set_if_possible(feature, 'linsolver', solver_name)
    if solver_name == 'cudss':
        set_if_possible(feature, 'cudssprecision', SOLVER.cudss_precision)
        set_if_possible(feature, 'cudsshybridmemory', SOLVER.cudss_hybrid_memory)
        set_if_possible(feature, 'cudssusehybridcompute', 'on' if SOLVER.cudss_use_hybrid_compute else 'off')
        set_if_possible(feature, 'cudssmultigpusinglenode', 'on' if SOLVER.cudss_use_multiple_gpus else 'off')
        set_if_possible(feature, 'cudssreorder', 'auto')
        set_if_possible(feature, 'cudssmatching', 'auto')
        set_if_possible(feature, 'cudssfactor', 'auto')
        feature.comment(
            'Direct linear solver set by simple_shell_common.py. '
            'Uses NVIDIA cuDSS for CUDA GPU acceleration when a compatible '
            'GPU and driver are available to the COMSOL process.'
        )
    elif solver_name == 'pardiso':
        set_if_possible(feature, 'clusterpardiso', 'on' if SOLVER.pardiso_use_cluster else 'off')
        feature.comment(
            'Direct linear solver set by simple_shell_common.py. '
            'PARDISO in COMSOL Direct solver does not expose a cuDSS-like '
            'single-precision factorization switch here; use SolverConfig.relative_tolerance '
            'for comparable transient tolerance studies.'
        )


def create_receiver_datasets(model) -> None:
    if not CREATE_RECEIVER_DATASETS:
        return
    datasets = model / 'datasets'
    for item in receiver_positions():
        dataset = datasets.create('CutPoint3D', name=f'receiver PZT {item["index"]:02d} point')
        dataset.property('data', 'dset1')
        dataset.property('pointx', f'{item["x_mm"]:.12g}[mm]')
        dataset.property('pointy', f'{item["y_mm"]:.12g}[mm]')
        dataset.property('pointz', f'{item["z_mm"]:.12g}[mm]')
        dataset.comment('Receiver point for radial shell displacement export.')


def create_marker_dataset(model, name: str, positions: list[dict], comment: str) -> None:
    if not positions:
        return
    dataset = (model / 'datasets').create('CutPoint3D', name=name)
    dataset.property('data', 'dset1')
    dataset.property('pointx', [f'{item["x_mm"]:.12g}[mm]' for item in positions])
    dataset.property('pointy', [f'{item["y_mm"]:.12g}[mm]' for item in positions])
    dataset.property('pointz', [f'{item["z_mm"]:.12g}[mm]' for item in positions])
    dataset.comment(comment)


def create_visual_marker_datasets(model) -> None:
    if not CREATE_VISUAL_MARKER_DATASETS:
        return
    create_marker_dataset(
        model,
        'transmitter PZT marker points',
        transmitter_positions(),
        'Visual-only transmitter marker points. These datasets are for COMSOL result plotting and do not affect physics, mesh, or loads.',
    )
    create_marker_dataset(
        model,
        'receiver PZT marker points',
        receiver_positions(),
        'Visual-only receiver marker points. Individual receiver datasets are still used for waveform export.',
    )


def build_model_object(
    client,
    model_name: str,
    defects: list[DefectConfig] | None = None,
    lobes: list[DefectLobeConfig] | None = None,
):
    defects = defects or []
    lobes = lobes or []
    model = client.create(model_name)
    add_parameters(model, damaged=bool(defects or lobes))
    geometry = create_geometry(model)
    shell_selection = create_shell_side_selection(model)
    create_material(model)
    create_functions(model)
    create_shell_physics(model, geometry, shell_selection, defects, lobes)
    create_mesh(model, geometry, shell_selection)
    create_study(model)
    create_receiver_datasets(model)
    create_visual_marker_datasets(model)
    problems = model.problems()
    if SOLVER.solve:
        model.solve()
    return model, problems


def build_model(
    client,
    model_name: str,
    output_dir: Path,
    defects: list[DefectConfig] | None = None,
    lobes: list[DefectLobeConfig] | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    try:
        model, problems = build_model_object(client, model_name, defects=defects, lobes=lobes)
        path = output_dir / f'{model_name}.mph'
        model.save(path)
        return path, problems
    finally:
        if model is not None:
            client.remove(model)


def write_build_log(path: Path, saved: Iterable[Path], problems: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = '\n'.join(f'- {note}' for note in DATASET_NOTES) or '- None'
    receivers = '\n'.join(
        '| {index:02d} | {theta_deg:.1f} | {x_mm:.3f} | {y_mm:.3f} | {z_mm:.3f} |'.format(**item)
        for item in receiver_positions()
    )
    path.write_text(
        f"""# Simple Shell Model Build Log

## Generated files

{chr(10).join(f'- `{item}`' for item in saved)}

## Model family

{MODEL_FAMILY}

## Notes

{notes}

## Simplifications

- Pipe is a cylindrical shell midsurface at `Rm = {(PIPE.mid_radius_mm):.3f} mm`.
- Wall loss defects are represented by spatially varying shell thickness, not Boolean corrosion cuts.
- PZT solids are removed. Excitation is an equivalent face load with a smooth transducer window.
- Receivers are 16 shell displacement points on the receiver ring.
- Mesh is controlled by wavelength: `hmax = {MESH.hmax_mm:.3f} mm`, not by PZT block dimensions.

## Where to find the important settings in COMSOL Model Builder

- Thickness: `Component 1 > Shell Mechanics > shell thickness and defect wall loss`.
- Explicit shell material: `Component 1 > Shell Mechanics > explicit aluminum shell elastic material`.
- Equivalent excitation: `Component 1 > Shell Mechanics > equivalent transducer face load`.
- Active transmitter/frequency: `Global Definitions > Parameters`, then `tx` and `pzt_fc`.
- Excitation pulse: `Global Definitions > Functions > five-cycle Hanning sine`.
- Receiver points: `Results > Datasets > receiver PZT 17 point` through `receiver PZT 32 point`.

The excitation patches are not separate geometric PZT faces. Their positions are encoded in the face-load expression as smooth spatial windows so they do not force local mesh refinement.

## Receiver points

| Channel | theta_deg | x_mm | y_mm | z_mm |
| --- | ---: | ---: | ---: | ---: |
{receivers}

## COMSOL self-check

```json
{json.dumps(problems, ensure_ascii=False, indent=2, default=str)}
```
""",
        encoding='utf-8',
    )


def model_metadata(dataset: str, defect_state: str, model_path: Path | str | None, problems) -> dict:
    return {
        'dataset': dataset,
        'defect_state': defect_state,
        'model_family': MODEL_FAMILY,
        'model_path': None if model_path is None else str(model_path),
        'pipe': asdict(PIPE),
        'transducer': asdict(TRANSDUCER),
        'material': asdict(MATERIAL),
        'absorbing_layer': asdict(ABSORBING_LAYER),
        'defect_model': asdict(DEFECT_MODEL),
        'solver': asdict(SOLVER),
        'mesh': {
            **asdict(MESH),
            'hmax_mm': MESH.hmax_mm,
        },
        'sweep': asdict(SWEEP),
        'receiver_indices': list(RECEIVER_INDICES),
        'create_receiver_datasets': CREATE_RECEIVER_DATASETS,
        'create_visual_marker_datasets': CREATE_VISUAL_MARKER_DATASETS,
        'position_perturbations': POSITION_PERTURBATIONS,
        'amplitude_scale': AMPLITUDE_SCALE,
        'problems': problems,
    }


def start_client(cores: int | None = None):
    if cores is None:
        return mph.start()
    return mph.start(cores=cores)
