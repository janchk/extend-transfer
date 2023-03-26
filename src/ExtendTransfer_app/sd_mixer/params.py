from dataclasses import dataclass

@dataclass
class MixerParams:
    id: str
    scale: float
    device: str
    n_samples: int
    seed: int
    steps: int
    h: int
    w: int
    sampler: str
    imgs: list([float, str])
