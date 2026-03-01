from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    scales: Float[Tensor, "batch gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"] | Float[Tensor, "batch gaussian 3"]
    opacities: Float[Tensor, "batch gaussian"]
    semantics: Float[Tensor, "batch gaussian dim"] | None
