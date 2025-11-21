from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    # harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    scales: Float[Tensor, "batch gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"]
    rgbs: Float[Tensor, "batch gaussian 3"]
    semantic: Float[Tensor, "batch gaussian dim"]
    opacities: Float[Tensor, "batch gaussian"]
