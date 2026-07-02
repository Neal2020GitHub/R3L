# Third-Party License Notices

This repository vendors third-party code that is distributed under licenses
separate from the project's own MIT License (see the repository root
`LICENSE`). The third-party licenses are reproduced below and preserved in
their respective directories.

## Holodeck / Holodeck 2.0 — Apache License 2.0

The `planners/holodeck_v2/`, `retrievers/holodeck_v2/`, and
`utils/holodeck_v2/` directories, and the `holodeck.py` entry point that
aggregates them, are derived from **AllenAI Holodeck**
([https://github.com/allenai/Holodeck](https://github.com/allenai/Holodeck);
Yang et al., *CVPR 2024*) and **Holodeck 2.0** (Bian et al., 2025,
arXiv:2508.05899).

- Upstream copyright: `Copyright 2023 Allen Institute for Artificial Intelligence`
- License: Apache License, Version 2.0
- Full text: see the `LICENSE` file in each of the three `holodeck_v2/`
  directories listed above (`planners/holodeck_v2/LICENSE`,
  `retrievers/holodeck_v2/LICENSE`, `utils/holodeck_v2/LICENSE`).

In accordance with Section 4 of the Apache License, Version 2.0:

- The original copyright notice and the Apache 2.0 license are retained.
- Each derived source file carries a prominent notice stating that it was
  modified for the R3L pipeline.
- The modifications made by the R3L authors (Yuqi Wang and Zhifeng Gu,
  Copyright (c) 2026) are additionally licensed under the project's MIT
  License (see the repository root `LICENSE`).

Holodeck and Holodeck 2.0 are works of the Allen Institute for AI and their
respective authors; the name "Holodeck" and the AllenAI name belong to their
respective owners and are used here only for attribution and to describe the
origin of the derived code, as permitted by Section 6 of the Apache License.

## Rotated IoU — MIT License

The vendored package at `utils/third_party/Rotated_IoU/` provides
differentiable rotated-box IoU/GIoU/DIoU utilities. It is derived from
[`lilanxiao/Rotated_IoU`](https://github.com/lilanxiao/Rotated_IoU)
(Copyright (c) 2020 Lanxiao Li), MIT licensed. The compiled CUDA/C++ vertex
sort extension was replaced with a vectorized pure-PyTorch implementation for
reproducibility without a CUDA toolchain. See
`utils/third_party/Rotated_IoU/LICENSE` for the full MIT text and
`utils/third_party/Rotated_IoU/README.md` for details.