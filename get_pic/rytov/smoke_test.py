"""Fast algebraic smoke tests that do not start COMSOL or read formal labels."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simple.get_pic.rytov.basis import (  # noqa: E402
    build_basis_grid,
    difference_operator,
    expand_rotational_response,
)
from simple.get_pic.rytov.common import (  # noqa: E402
    complex_rytov,
    sha256_file,
    sha256_json,
    write_json,
)
from simple.get_pic.rytov.config import RytovConfig  # noqa: E402
from simple.get_pic.rytov.formal_selection import (  # noqa: E402
    batch_output_root,
    select_formal_sample_ids,
)
from simple.get_pic.rytov.inversion import invert_observation  # noqa: E402
from simple.get_pic.rytov.rytov_operator import (  # noqa: E402
    ARTIFACT_KIND,
    SCHEMA_VERSION,
    FullWaveRytovOperator,
    operator_array_hashes,
    save_operator_npz,
)


def _expect_error(function, error_type: type[BaseException], message: str) -> None:
    try:
        function()
    except error_type:
        return
    raise AssertionError(message)


def _smoke_config() -> RytovConfig:
    config = RytovConfig(
        output_root="simple/get_pic/rytov/_smoke_output",
        frequencies_hz=(20_000.0, 32_500.0),
        jacobian_assembly_mode="all_tx",
        training_tx_indices=tuple(range(1, 17)),
        theta_basis_count=16,
        z_basis_count=2,
        z_min_mm=350.0,
        z_max_mm=650.0,
        basis_radius_theta_mm=30.0,
        basis_radius_z_mm=45.0,
        perturbation_depths_mm=(0.25,),
        image_size=32,
        ridge_weight=0.0,
        tv_weight=0.0,
        irls_iterations=1,
        lsq_tolerance=1.0e-9,
        lsq_max_iterations=500,
        formal_sample_ids=(1,),
        validation_sample_ids=(1,),
    )
    config.validate()
    return config


def test_zero_rytov() -> None:
    rng = np.random.default_rng(10)
    healthy = 0.5 + rng.normal(size=(2, 3, 4)) + 1j * rng.normal(size=(2, 3, 4))
    floor = np.full(4, 0.05, dtype=np.float64)
    values = complex_rytov(healthy, healthy, floor, np.arange(4, dtype=np.float64))
    if not np.allclose(values, 0.0, rtol=0.0, atol=1.0e-14):
        raise AssertionError("H=H0 did not produce a zero complex-Rytov observation")


def test_rotational_indices() -> None:
    z_count, ring_count, frequency_count = 2, 16, 3
    reference = np.empty(
        (z_count, ring_count, ring_count, frequency_count), dtype=np.complex128
    )
    for z_index in range(z_count):
        for theta_index in range(ring_count):
            for rx_index in range(ring_count):
                for frequency_index in range(frequency_count):
                    code = 10_000 * z_index + 100 * theta_index + 10 * rx_index + frequency_index
                    reference[z_index, theta_index, rx_index, frequency_index] = code + 1j * (code + 1)
    expanded = expand_rotational_response(reference)
    for tx_index in range(ring_count):
        for rx_index in range(ring_count):
            for z_index in range(z_count):
                for theta_index in range(ring_count):
                    column = z_index * ring_count + theta_index
                    expected = reference[
                        z_index,
                        (theta_index - tx_index) % ring_count,
                        (rx_index - tx_index) % ring_count,
                    ]
                    if not np.array_equal(expanded[tx_index, rx_index, :, column], expected):
                        raise AssertionError("Rotational virtual-TX index mapping is incorrect")


def test_periodic_difference() -> None:
    differences = difference_operator(2, 16, theta_spacing_mm=2.0, z_spacing_mm=3.0)
    constant = np.ones(32, dtype=np.float64)
    if not np.allclose(differences @ constant, 0.0):
        raise AssertionError("Periodic/nonperiodic difference operator rejects a constant field")
    ramp = np.tile(np.arange(16, dtype=np.float64), 2)
    seam_value = float((differences @ ramp)[15])
    if not np.isclose(seam_value, (ramp[0] - ramp[15]) / 2.0):
        raise AssertionError("Theta difference does not close across the periodic seam")


def synthetic_operator(config: RytovConfig) -> tuple[FullWaveRytovOperator, np.ndarray]:
    rng = np.random.default_rng(20260724)
    measurement_shape = (16, 16, len(config.frequencies_hz))
    coefficient_count = config.theta_basis_count * config.z_basis_count
    jacobian = (
        rng.normal(size=(*measurement_shape, coefficient_count))
        + 1j * rng.normal(size=(*measurement_shape, coefficient_count))
    ) / np.sqrt(2.0 * np.prod(measurement_shape))
    grid = build_basis_grid(config)
    true_coefficients = np.zeros(coefficient_count, dtype=np.float64)
    true_coefficients[3] = 1.25
    true_coefficients[16 + 12] = 2.0
    metadata = {
        "config_sha256": sha256_json(config.to_dict()),
        "synthetic_smoke_only": True,
    }
    operator = FullWaveRytovOperator(
        metadata_path=HERE / "_synthetic_operator.json",
        model_path=HERE / "_synthetic_operator.npz",
        model_sha256="synthetic-smoke",
        assembly_mode=config.jacobian_assembly_mode,
        jacobian=jacobian,
        data_weights=np.ones(measurement_shape, dtype=np.float64),
        healthy_h=np.ones(measurement_shape, dtype=np.complex128),
        frequency_floor=np.full(len(config.frequencies_hz), 0.1, dtype=np.float64),
        tx_indices=np.arange(1, 17, dtype=np.int32),
        rx_indices=np.arange(17, 33, dtype=np.int32),
        frequencies_hz=np.asarray(config.frequencies_hz, dtype=np.float64),
        theta_centers_deg=grid.theta_centers_deg,
        z_centers_mm=grid.z_centers_mm,
        basis_radius_theta_mm=grid.radius_theta_mm,
        basis_radius_z_mm=grid.radius_z_mm,
        depth_limit_mm=config.depth_limit_mm,
        normalization_denominator_mm=config.normalization_denominator_mm,
        training_linearity=np.zeros(coefficient_count, dtype=np.float64),
        metadata=metadata,
    )
    operator.assert_compatible(config)
    return operator, true_coefficients


def test_synthetic_inverse() -> None:
    config = _smoke_config()
    operator, truth = synthetic_operator(config)
    observation = operator.predict(truth)
    result = invert_observation(
        config=config,
        operator=operator,
        observation=observation,
        data_weights=np.ones(operator.measurement_shape, dtype=np.float64),
        observation_diagnostics={"synthetic": True},
    )
    relative_error = float(
        np.linalg.norm(np.asarray(result.coefficients_mm) - truth)
        / max(np.linalg.norm(truth), 1.0e-12)
    )
    if relative_error > 2.0e-4:
        raise AssertionError(f"Synthetic coefficient recovery error is too high: {relative_error}")
    if result.image_mm.shape != (config.image_size, config.image_size):
        raise AssertionError("Synthetic inversion returned an unexpected image shape")
    if not np.all(np.isfinite(result.image_mm)):
        raise AssertionError("Synthetic prediction contains non-finite values")
    if np.min(result.image_mm) < 0.0 or np.max(result.image_mm) > config.depth_limit_mm:
        raise AssertionError("Synthetic prediction violates physical bounds")
    if not np.allclose(
        result.image_norm,
        result.image_mm / config.normalization_denominator_mm,
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise AssertionError("Synthetic millimeter normalization is inconsistent")
    predicted = operator.predict(result.coefficients_mm)
    if np.linalg.norm(predicted - observation) / np.linalg.norm(observation) > 2.0e-4:
        raise AssertionError("Synthetic inversion does not close in complex data space")


def test_operator_artifact_round_trip() -> None:
    config = _smoke_config()
    operator, _truth = synthetic_operator(config)
    with tempfile.TemporaryDirectory(prefix="rytov_operator_smoke_") as directory:
        root = Path(directory)
        model_path = root / "operator.npz"
        metadata_path = root / "operator.json"
        save_operator_npz(
            model_path,
            jacobian=operator.jacobian,
            data_weights=operator.data_weights,
            healthy_h=operator.healthy_h,
            frequency_floor=operator.frequency_floor,
            tx_indices=operator.tx_indices,
            rx_indices=operator.rx_indices,
            frequencies_hz=operator.frequencies_hz,
            theta_centers_deg=operator.theta_centers_deg,
            z_centers_mm=operator.z_centers_mm,
            basis_radius_theta_mm=operator.basis_radius_theta_mm,
            basis_radius_z_mm=operator.basis_radius_z_mm,
            training_linearity=operator.training_linearity,
            raw_dh_slope=operator.jacobian,
            amplitude_correction=np.ones(operator.measurement_shape, dtype=np.float64),
        )
        external_contract: dict[str, str] = {}
        for key in (
            "formal_healthy_npz",
            "formal_healthy_metadata",
            "training_healthy_npz",
            "training_plan",
        ):
            path = root / f"{key}.txt"
            path.write_text(f"synthetic {key}\n", encoding="ascii")
            external_contract[key] = str(path)
            external_contract[f"{key}_sha256"] = sha256_file(path)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "real_formal_sample_labels_used": False,
            "model_npz": model_path.name,
            "model_npz_sha256": sha256_file(model_path),
            "config_sha256": sha256_json(config.to_dict()),
            "jacobian_assembly_mode": operator.assembly_mode,
            "physical_scale": {
                "depth_limit_mm": config.depth_limit_mm,
                "normalization_denominator_mm": config.normalization_denominator_mm,
            },
            "rytov_stability": {
                "minimum_data_weight": config.minimum_data_weight,
                "minimum_ratio_magnitude": config.minimum_ratio_magnitude,
                "phase_branch_margin_rad": config.phase_branch_margin_rad,
            },
            "data_contract": external_contract,
            "array_sha256": operator_array_hashes(
                jacobian=operator.jacobian,
                data_weights=operator.data_weights,
                healthy_h=operator.healthy_h,
                frequency_floor=operator.frequency_floor,
                tx_indices=operator.tx_indices,
                rx_indices=operator.rx_indices,
                frequencies_hz=operator.frequencies_hz,
                theta_centers_deg=operator.theta_centers_deg,
                z_centers_mm=operator.z_centers_mm,
                training_linearity=operator.training_linearity,
            ),
        }
        write_json(metadata_path, metadata)
        loaded = FullWaveRytovOperator.load(metadata_path, config=config)
        if loaded.model_sha256 != sha256_file(model_path):
            raise AssertionError("Round-tripped operator model hash differs")
        if not np.allclose(loaded.jacobian, operator.jacobian, rtol=1.0e-6, atol=1.0e-7):
            raise AssertionError("Round-tripped complex Jacobian differs")
        if not np.array_equal(loaded.tx_indices, operator.tx_indices):
            raise AssertionError("Round-tripped operator axes differ")


def test_output_scope_guard() -> None:
    config = _smoke_config()
    escaped = replace(
        config,
        output_root="simple/get_pic/physical_inversion/forbidden_rytov_output",
    )
    _expect_error(
        escaped.validate,
        ValueError,
        "Config accepted an output path outside simple/get_pic/rytov",
    )


def test_formal_source_id_batch_selection() -> None:
    with tempfile.TemporaryDirectory(prefix="rytov_formal_selection_smoke_") as directory:
        dataset_path = Path(directory) / "formal_dataset"
        response_dir = dataset_path / "frequency_response"
        response_dir.mkdir(parents=True)
        for sample_id in (641, 642):
            (response_dir / f"dataset_a_frequency_sample_{sample_id:04d}_H_complex.npz").touch()

        selection = select_formal_sample_ids(
            dataset_path=dataset_path,
            configured_ids=(1,),
            start_id=641,
            end_id=642,
        )
        if selection.sample_ids != (641, 642) or not selection.is_source_range:
            raise AssertionError("Formal source-ID batch selection is incorrect")
        expected_root = Path(directory) / "output_dataset" / "batches" / "ids_000641_000642"
        if batch_output_root(Path(directory) / "output_dataset", 641, 642) != expected_root:
            raise AssertionError("Formal source-ID batch root is not deterministic")

        _expect_error(
            lambda: select_formal_sample_ids(
                dataset_path=dataset_path,
                configured_ids=(1,),
                start_id=641,
                end_id=643,
            ),
            FileNotFoundError,
            "Missing source IDs did not reject an incomplete formal batch",
        )
        _expect_error(
            lambda: select_formal_sample_ids(
                dataset_path=dataset_path,
                configured_ids=(1,),
                explicit_ids=("641",),
                start_id=641,
                end_id=641,
            ),
            ValueError,
            "Mixed explicit and source-ID range selection was accepted",
        )


def main() -> None:
    tests = (
        test_zero_rytov,
        test_rotational_indices,
        test_periodic_difference,
        test_synthetic_inverse,
        test_operator_artifact_round_trip,
        test_output_scope_guard,
        test_formal_source_id_batch_selection,
    )
    for test in tests:
        test()
        print(f"[pass] {test.__name__}")
    print("Full-wave Rytov algebraic smoke tests passed; COMSOL and formal labels were not used.")


if __name__ == "__main__":
    main()
