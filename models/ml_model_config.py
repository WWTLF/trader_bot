from dataclasses import dataclass

@dataclass
class MlModelConfig:
    id: int
    model_name: str
    config: dict
    path: str