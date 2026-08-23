import math
from dataclasses import dataclass, fields
from lerobot.configs.policies import PreTrainedConfig
from kuavo_train.wrapper.policy.act.ACTConfigWrapper import CustomACTConfigWrapper

@PreTrainedConfig.register_subclass("stochastic_act")
@dataclass
class StochasticACTConfigWrapper(CustomACTConfigWrapper):
    init_log_std: float = -3.5
    log_std_min: float = -5.0
    log_std_max: float = -2.3
    
    def __post_init__(self):
        super().__post_init__()
        for name in ("init_log_std", "log_std_min", "log_std_max"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number, got {type(value).__name__}")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if not self.log_std_min < self.init_log_std < self.log_std_max:
            raise ValueError(
                f"Expected log_std_min < init_log_std < log_std_max, got "
                f"{self.log_std_min} < {self.init_log_std} < {self.log_std_max}"
            )
            
    @classmethod
    def from_il_pretrained(
        cls,
        pretrained_name_or_path,
        *,
        init_log_std: float | None = None,
        log_std_min: float | None = None,
        log_std_max: float | None = None,
        **kwargs,
    ) -> "StochasticACTConfigWrapper":
        il_config = CustomACTConfigWrapper.from_pretrained(pretrained_name_or_path, **kwargs)
        stochastic_names = {"init_log_std", "log_std_min", "log_std_max"}
        config_kwargs = {
            f.name: getattr(il_config, f.name)
            for f in fields(cls)
            if f.init and f.name not in stochastic_names and hasattr(il_config, f.name)
        }
        if init_log_std is not None:
            config_kwargs["init_log_std"] = init_log_std
        if log_std_min is not None:
            config_kwargs["log_std_min"] = log_std_min
        if log_std_max is not None:
            config_kwargs["log_std_max"] = log_std_max
        return cls(**config_kwargs)