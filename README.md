# R<sup>3</sup>L: Reasoning 3D Layouts from Relative Spatial Relations (ICML 2026)

## Get Started

This code has been tested to run on MacOS and Ubuntu. 
The Blender renderer is configured to route to METAL or CUDA based on the platform.

If you haven't already, install `uv`: 
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, download the objathor data and retriever models
```
uv run python scripts/download_data.py
uv run python scripts/download_models.py
```

Set your API keys: 
```
cp scripts/env.example.sh scripts/env.sh
vim scripts/env.sh
```

Finally, run the code simply by: 
```
uv sync
source scripts/env.sh
uv run -q python main.py
```

If you encountered any issue, please file an issue.


## BibTeX
