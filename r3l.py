from typing import Callable, Union
from adapters.protocols import R3LSolution, R3LBlueprint, HolodeckBlueprint
from adapters.blueprints import holodeck_to_r3l_blueprint
from solvers.r3l.pipeline import Pipeline, Stage

def create_r3l_layout(
    blueprint: Union[R3LBlueprint, HolodeckBlueprint],
    save_dir: str,
    asset_dir: str,
    on_stage: Callable[[Stage], None] = lambda _stage: None,
) -> R3LSolution:
    if isinstance(blueprint, HolodeckBlueprint):
        blueprint = holodeck_to_r3l_blueprint(blueprint)

    return Pipeline(blueprint, save_dir, asset_dir).run(on_stage)
