"""Image-textured irregular single-defect fields on an unfolded pipe surface."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np


PIPE_LENGTH_MM = 1000.0
PIPE_MID_RADIUS_MM = 155.0
MAX_WALL_LOSS_MM = 5.0
SIMPLE_ROOT = Path(__file__).resolve().parents[2]
TEXTURE_IMAGE_PATH = SIMPLE_ROOT / 'Defect _models' / '1.jpg'
GENERATOR_NAME = 'irregular_polygon_image_texture_v4_multiscale'
SIZE_CLASS_RADIUS_RANGES_MM = {
    'small': (40.0, 70.0),
    'medium': (70.0, 110.0),
    'large': (110.0, 170.0),
}


@dataclass(frozen=True)
class IrregularDefectField:
    seed: int
    size_class: str
    arc_mm: np.ndarray
    z_mm: np.ndarray
    depth_mm: np.ndarray
    center_theta_deg: float
    center_z_mm: float
    radius_theta_mm: float
    radius_z_mm: float
    texture_source: str
    texture_sha256: str
    texture_rotation_deg: int

    def metadata(self) -> dict[str, object]:
        active = self.depth_mm > 0.01
        cell_area_mm2 = float((self.arc_mm[1] - self.arc_mm[0]) * (self.z_mm[1] - self.z_mm[0]))
        return {
            'generator': GENERATOR_NAME,
            'seed': self.seed,
            'size_class': self.size_class,
            'coordinate_system': 'unfolded pipe midsurface arc-z',
            'center_theta_deg': self.center_theta_deg,
            'center_z_mm': self.center_z_mm,
            'radius_theta_mm': self.radius_theta_mm,
            'radius_z_mm': self.radius_z_mm,
            'texture_source': self.texture_source,
            'texture_sha256': self.texture_sha256,
            'texture_rotation_deg': self.texture_rotation_deg,
            'texture_method': 'opencv rotate-crop, polygon mask, inverted grayscale, Gaussian sigma=2',
            'grid_shape_z_arc': list(self.depth_mm.shape),
            'arc_range_mm': [float(self.arc_mm[0]), float(self.arc_mm[-1])],
            'z_range_mm': [float(self.z_mm[0]), float(self.z_mm[-1])],
            'max_depth_mm': float(self.depth_mm.max()),
            'mean_active_depth_mm': float(self.depth_mm[active].mean()),
            'active_area_fraction': float(active.mean()),
            'active_area_mm2': float(active.sum() * cell_area_mm2),
        }


def _image_texture(path: Path, grid_count: int, rotation_deg: int):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError('OpenCV is required for image-textured irregular defects') from error
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Cannot read defect texture image: {path}')
    height, width = image.shape[:2]
    transform = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), rotation_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    grayscale = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    grayscale[grayscale < 5] = 255
    interpolation = cv2.INTER_AREA if min(grayscale.shape) > grid_count else cv2.INTER_CUBIC
    resized = cv2.resize(grayscale, (grid_count, grid_count), interpolation=interpolation)
    return cv2, resized.astype(np.float32)


def _polygon_mask(cv2, rng: np.random.Generator, grid_count: int) -> np.ndarray:
    vertex_count = int(rng.integers(20, 51))
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, vertex_count))
    radius_px = (grid_count - 1) / 2.8
    variation_px = max(5, int(np.rint(radius_px * rng.uniform(0.0, 0.2))))
    low = max(1, int(np.rint(radius_px)) - variation_px)
    radii = rng.integers(low, int(np.floor(radius_px)) + 1, vertex_count)
    center = 0.5 * (grid_count - 1)
    points = np.column_stack((center + np.cos(angles) * radii, center + np.sin(angles) * radii))
    mask = np.zeros((grid_count, grid_count), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


def generate_irregular_defect(
    seed: int,
    *,
    grid_count: int = 257,
    radius_range_mm: tuple[float, float] | None = None,
    depth_range_mm: tuple[float, float] = (2.5, 4.6),
    texture_path: Path | str | None = None,
) -> IrregularDefectField:
    """Generate one image-textured polygon wall-loss field."""

    if grid_count < 33 or grid_count % 2 == 0:
        raise ValueError('grid_count must be odd and at least 33')
    rng = np.random.default_rng(seed)
    if radius_range_mm is None:
        size_class = tuple(SIZE_CLASS_RADIUS_RANGES_MM)[seed % len(SIZE_CLASS_RADIUS_RANGES_MM)]
        radius_range_mm = SIZE_CLASS_RADIUS_RANGES_MM[size_class]
    else:
        size_class = 'custom'
    radius = rng.uniform(*radius_range_mm)
    aspect = rng.uniform(0.65, 1.45)
    radius_theta = radius * np.sqrt(aspect)
    radius_z = radius / np.sqrt(aspect)
    max_arc = np.pi * PIPE_MID_RADIUS_MM
    center_arc = rng.uniform(-max_arc, max_arc)
    center_z = rng.uniform(330.0, 670.0)
    extent_arc = 1.4 * radius_theta
    extent_z = 1.4 * radius_z
    arc = np.linspace(center_arc - extent_arc, center_arc + extent_arc, grid_count)
    z_mm = np.linspace(center_z - extent_z, center_z + extent_z, grid_count)
    source_path = (TEXTURE_IMAGE_PATH if texture_path is None else Path(texture_path)).resolve()
    rotation_deg = int(rng.integers(1, 361))
    cv2, texture = _image_texture(source_path, grid_count, rotation_deg)
    mask = _polygon_mask(cv2, rng, grid_count)
    masked = texture * mask
    masked[masked == 0.0] = float(masked.max()) + 5.0
    depth_unit = (float(masked.max()) - masked) / max(float(masked.max() - masked.min()), 1e-12)
    peak_depth = rng.uniform(*depth_range_mm)
    depth = cv2.GaussianBlur(
        (peak_depth * depth_unit).astype(np.float32),
        (0, 0),
        sigmaX=2.0,
        sigmaY=2.0,
        borderType=cv2.BORDER_REPLICATE,
    )
    depth = np.clip(depth, 0.0, MAX_WALL_LOSS_MM)
    depth *= peak_depth / max(float(depth.max()), 1e-12)
    depth[depth < 1e-5] = 0.0
    center_theta = float(np.degrees(center_arc / PIPE_MID_RADIUS_MM) % 360.0)
    try:
        texture_source = source_path.relative_to(SIMPLE_ROOT).as_posix()
    except ValueError:
        texture_source = str(source_path)
    return IrregularDefectField(
        seed=seed,
        size_class=size_class,
        arc_mm=arc,
        z_mm=z_mm,
        depth_mm=depth.astype(np.float32),
        center_theta_deg=center_theta,
        center_z_mm=float(center_z),
        radius_theta_mm=float(radius_theta),
        radius_z_mm=float(radius_z),
        texture_source=texture_source,
        texture_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        texture_rotation_deg=rotation_deg,
    )


def _resample_periodic_depth(
    field: IrregularDefectField,
    arc_mm: np.ndarray,
    z_mm: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample one local field with periodic pipe-arc wrapping."""

    max_arc = np.pi * PIPE_MID_RADIUS_MM
    circumference = 2.0 * max_arc
    center_arc = 0.5 * float(field.arc_mm[0] + field.arc_mm[-1])
    local_arc = center_arc + (arc_mm - center_arc + max_arc) % circumference - max_arc
    ds = float(field.arc_mm[1] - field.arc_mm[0])
    dz = float(field.z_mm[1] - field.z_mm[0])
    js = (local_arc - field.arc_mm[0]) / ds
    iz = (z_mm - field.z_mm[0]) / dz
    valid_s = (js >= 0.0) & (js <= field.arc_mm.size - 1)
    valid_z = (iz >= 0.0) & (iz <= field.z_mm.size - 1)
    j0 = np.clip(np.floor(js).astype(int), 0, field.arc_mm.size - 2)
    i0 = np.clip(np.floor(iz).astype(int), 0, field.z_mm.size - 2)
    ws = np.clip(js - j0, 0.0, 1.0)[None, :]
    wz = np.clip(iz - i0, 0.0, 1.0)[:, None]
    values = field.depth_mm
    v0 = values[i0[:, None], j0[None, :]] * (1.0 - ws) + values[i0[:, None], j0[None, :] + 1] * ws
    v1 = values[i0[:, None] + 1, j0[None, :]] * (1.0 - ws) + values[i0[:, None] + 1, j0[None, :] + 1] * ws
    return (v0 * (1.0 - wz) + v1 * wz) * (valid_z[:, None] & valid_s[None, :])


def unfolded_depth_map(
    field: IrregularDefectField,
    *,
    theta_count: int = 512,
    z_count: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample the local COMSOL table onto the standard EDM theta-z label grid."""

    theta_deg = np.linspace(0.0, 360.0, theta_count, endpoint=False)
    z_mm = np.linspace(0.0, PIPE_LENGTH_MM, z_count)
    wrapped_deg = (theta_deg + 180.0) % 360.0 - 180.0
    arc_mm = PIPE_MID_RADIUS_MM * np.deg2rad(wrapped_deg)
    depth = _resample_periodic_depth(field, arc_mm, z_mm)
    return theta_deg.astype(np.float32), z_mm.astype(np.float32), depth.astype(np.float32)


def write_comsol_table(field: IrregularDefectField, path: Path) -> Path:
    """Write arc-z-depth rows accepted by a COMSOL 2D interpolation function."""

    path.parent.mkdir(parents=True, exist_ok=True)
    max_arc = np.pi * PIPE_MID_RADIUS_MM
    arc_mm = np.linspace(-max_arc, max_arc, 513)
    depth_mm = _resample_periodic_depth(field, arc_mm, field.z_mm)
    arc_grid, z_grid = np.meshgrid(arc_mm, field.z_mm)
    table = np.column_stack((arc_grid.ravel(), z_grid.ravel(), depth_mm.ravel()))
    np.savetxt(path, table, fmt='%.9g', header='arc_mm z_mm wall_loss_mm', comments='% ')
    return path


def write_label_package(
    label_module,
    output_dir: Path,
    sample_id: str,
    field: IrregularDefectField,
    *,
    write_preview_png: bool = True,
) -> dict[str, object]:
    """Write files compatible with the existing Dataset A label layout."""

    output_dir.mkdir(parents=True, exist_ok=True)
    theta_deg, z_mm, depth_mm = unfolded_depth_map(field)
    field_metadata = field.metadata()
    max_wall_loss_mm = 9.0
    depth_norm = np.clip(depth_mm / max_wall_loss_mm, 0.0, 1.0).astype(np.float32)
    mask = (depth_mm >= 0.01).astype(np.uint8)
    stem = label_module.safe_stem(sample_id, 'irregular_sample')
    files = {
        'depth_mm_npy': output_dir / f'{stem}_defect_depth_mm.npy',
        'depth_norm_npy': output_dir / f'{stem}_defect_depth_norm.npy',
        'mask_npy': output_dir / f'{stem}_defect_mask.npy',
        'preview_png': output_dir / f'{stem}_defect_label.png',
        'metadata_json': output_dir / f'{stem}_defect_label_metadata.json',
    }
    np.save(files['depth_mm_npy'], depth_mm)
    np.save(files['depth_norm_npy'], depth_norm)
    np.save(files['mask_npy'], mask)
    if write_preview_png:
        label_module.save_depth_png(
            depth_mm,
            files['preview_png'],
            theta_deg,
            z_mm,
            f'{stem} irregular defect depth label',
            preview_max_mm=MAX_WALL_LOSS_MM,
        )
        preview = str(files['preview_png'])
    else:
        preview = None
    metadata = {
        'sample_id': stem,
        'generator': GENERATOR_NAME,
        'size_class': field.size_class,
        'active_area_mm2': field_metadata['active_area_mm2'],
        'texture_source': field.texture_source,
        'texture_sha256': field.texture_sha256,
        'texture_rotation_deg': field.texture_rotation_deg,
        'coordinate_system': 'unfolded outer pipe surface, theta-z',
        'array_shape': ['z_index', 'theta_index'],
        'depth_units': 'mm',
        'theta_axis': {'units': 'deg', 'range': [0.0, 360.0], 'count': len(theta_deg), 'endpoint_included': False},
        'z_axis': {'units': 'mm', 'range': [0.0, PIPE_LENGTH_MM], 'count': len(z_mm)},
        'h_min_mm': 1.0,
        'max_wall_loss_mm': max_wall_loss_mm,
        'defect_loss_max_mm': MAX_WALL_LOSS_MM,
        'normalization_denominator_mm': max_wall_loss_mm,
        'mask_threshold_mm': 0.01,
        'formula': 'OpenCV image texture inside a random polygon with Gaussian-smoothed boundary; periodic linear interpolation in arc-z',
        'files': {key: preview if key == 'preview_png' else str(value) for key, value in files.items()},
        'defect_count': 1,
        'lobe_count': 0,
    }
    files['metadata_json'].write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


@contextmanager
def irregular_shell_field(shell_module, field: IrregularDefectField, table_path: Path) -> Iterator[None]:
    """Temporarily make the shared Shell builder consume one tabulated field."""

    original_create_functions = shell_module.create_functions
    original_thickness_expression = shell_module.thickness_expression

    def create_functions(model) -> None:
        original_create_functions(model)
        function = (model / 'functions').create('Interpolation', name='irregular wall-loss depth')
        function.property('source', 'file')
        function.property('filename', table_path.resolve())
        function.property('argunit', ['mm', 'mm'])
        function.property('fununit', 'mm')
        function.property('interp', 'linear')
        function.property('extrap', 'const')
        function.java.importData()
        function.comment(
            'Tabulated outer-surface wall loss on unfolded coordinates '
            '(arc=Rm*atan2(y,x), z). The same field produces the EDM label.'
        )

    def thickness_expression(_defects, _lobes) -> str:
        # COMSOL spreadsheet interpolation registers its imported value as int1.
        loss = 'int1(Rm*atan2(y,x),z)'
        return f'max(h_min,h0-min(defect_loss_max,{loss}))'

    shell_module.create_functions = create_functions
    shell_module.thickness_expression = thickness_expression
    try:
        yield
    finally:
        shell_module.create_functions = original_create_functions
        shell_module.thickness_expression = original_thickness_expression
