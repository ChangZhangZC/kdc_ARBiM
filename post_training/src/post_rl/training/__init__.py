from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

from .stage_gate_workspace import TrainACTWorkspace

__all__ = ["TrainACTWorkspace"]
