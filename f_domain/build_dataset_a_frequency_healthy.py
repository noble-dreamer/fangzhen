"""Build the Dataset A healthy frequency-domain shell model for inspection."""

from __future__ import annotations

import json

import frequency_domain_common as fcommon


OUTPUT_DIR = fcommon.OUTPUT_ROOT / 'dataset_a_frequency_shell'
MODEL_NAME = 'pipe_shell_frequency_healthy'


def main() -> None:
    fcommon.configure_dataset_a_frequency(use_parametric_sweep=True)
    fcommon.shell.CREATE_RECEIVER_DATASETS = True
    fcommon.shell.CREATE_VISUAL_MARKER_DATASETS = True
    client = fcommon.shell.start_client()
    saved = []
    problems = {}
    try:
        path, model_problems = fcommon.shell.build_model(client, MODEL_NAME, OUTPUT_DIR)
        saved.append(path)
        problems[MODEL_NAME] = model_problems
        metadata = fcommon.model_metadata('A_frequency', 'healthy_no_defect', path, model_problems)
        (OUTPUT_DIR / 'metadata').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'metadata' / f'{MODEL_NAME}.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8',
        )
        fcommon.write_frequency_build_log(
            OUTPUT_DIR / 'dataset_a_frequency_shell_build_log.md',
            saved,
            problems,
        )
    finally:
        client.clear()
    print(f'Saved {saved[0]}')
    print('No frequency-domain study was solved.')


if __name__ == '__main__':
    main()
