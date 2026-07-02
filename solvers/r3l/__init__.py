"""
    ____ _____ __
   / __ \__  // /
  / /_/ //_ </ /
 / _, _/__/ / /___
/_/ |_/____/_____/

R³L: Reasoning 3D Layouts from Relative Spatial Relations

Flow:  requirement ─► generate ─► compile ─► optimize ─► assemble ─► R3LSolution
       (Pipeline.run drives it; render emits images/GIFs along the way.)

Entry
    pipeline      Pipeline.run() — drives the whole flow; read its docstring first.

    Stages (verbs)
        generate      requirement ─► constraint JSON  (LLM frontend, or reparse a cached llm_output.py).
        compile       constraint JSON ─► CompiledConstraints  (the differentiable objective).
        optimize      gradient-descent the layout  (base + finetune stages).

The Objective  (built by `compile`, consumed by `optimize`)
    constraints   CompiledConstraints: frozen, queryable product (loss terms + scene + params).
    builders      make_*: bind a loss kernel + config into a LossTerm.
    losses        ~24 pure loss kernels (collision / wall / facing / gap / ...) + clamp_param.
    activations   Softplus2 / ReLU2: nonlinearity primitives used by the kernels.
    cluster       ClusterMeta / SceneIndex / AugmentedState: cluster topology + reparam frame.
    physics       physics(): collision/wall/aesthetics assembly; regularize(): Var-param priors.

Inputs & I/O
    assets        blueprint tree + disk annotations ─► asset tables (info / id maps / flatten).
    layout        init_layout: random initial poses.
    render        render() orchestrator: _render_image (static stills) + _render_2d (top-down GIF) + _render_3d (Cycles video).
    report        LossDashboard (console) + constraint-param report + loss_curve.csv/.png.
    schedulers    learning-rate / physics-weight schedules.

Config
    config        frozen Pydantic schema; validates config.yaml at load.

Subpackages
    dsl/          parse the Python-DSL constraint program ─► JSON (+ cogmap).
    prompts/      LLM prompt templates.

Shared types live in utils/r3l/ (PoseVec, BBoxVec, AssetInfo, LossTerm, ParamTable, ...).
"""
