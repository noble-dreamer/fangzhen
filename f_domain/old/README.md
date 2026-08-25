# Archived high-contrast defect sampling

`high_contrast_defect_sampling.py` freezes both effective configurations found in the existing
high-contrast frequency sample metadata:

- `v1`: Commit `da4e12b`, used by samples `0001-1040`;
- `v2`: the stronger lobe defaults from Commit `609afe5`, used by samples `1041-1200`.

The streaming entry point kept the same four sampling overrides until Commit `83463b8`. The
module defaults to `profile='auto'`, which follows the existing dataset cutoff above.

This profile is retained only for reproducibility. It uses:

- 1/2/3-defect weights of 30%/45%/25%;
- small, medium, and large primary defects;
- frequent `large + small` and `large + medium + small` mixtures;
- an aspect-ratio range of 0.75-1.45;
- the historical seed rule `710000 + sample_id`.

These choices can produce weak defects beside much stronger defects. The current frequency
streaming entry point does not import this module and continues to use
`balanced_detectable_v1`.

To reconstruct legacy metadata without running COMSOL:

```python
from f_domain.old.high_contrast_defect_sampling import generate_legacy_sample
from simple_defect_common import sample_to_dict

metadata = sample_to_dict(generate_legacy_sample(1))
v2_metadata = sample_to_dict(generate_legacy_sample(1201, profile='v2'))
```
