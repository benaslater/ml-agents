from typing import List, Tuple
from mlagents_envs.base_env import ObservationSpec, DimensionProperty, ObservationType


def create_observation_specs_with_shapes(
    shapes: List[Tuple[int, ...]]
) -> List[ObservationSpec]:
    obs_specs: List[ObservationSpec] = []
    for i, shape in enumerate(shapes):
        dim_prop = (DimensionProperty.UNSPECIFIED,) * len(shape)
        if len(shape) == 2:
            dim_prop = (DimensionProperty.VARIABLE_SIZE, DimensionProperty.NONE)
        spec = ObservationSpec(
            name=f"observation {i} with shape {shape}",
            shape=shape,
            dimension_property=dim_prop,
            observation_type=ObservationType.DEFAULT,
        )
        obs_specs.append(spec)
    return obs_specs
