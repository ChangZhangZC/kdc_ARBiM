from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

from .checkpoint_compat_workspace import TrainACTWorkspace

__all__ = ["TrainACTWorkspace"]
