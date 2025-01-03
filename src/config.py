import yaml
from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: Union[str, list]  # Accepts a single string or a list of strings
    use_dora: bool
    lora_plus_lr_ratio: int

@dataclass
class FineTuningConfig:
    epochs: int
    batch_size: int
    warmup_ratio: float
    lr: float
    lr_scheduler_type: str
    eval_strategy: str
    save_strategy: str
    logging_steps: int
    save_total_limit: int
    save_dir: str

@dataclass
class Config:
    model_repo: str
    data_config_type: str  # Options: [pair, triplets, pair_score, pair_class]
    lora: LoRAConfig
    finetuning: FineTuningConfig

    @staticmethod
    def from_yaml(file_path: str) -> "Config":
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return Config(
            model_repo=config_dict["model_repo"],
            data_config_type=config_dict["data_config_type"],
            lora=LoRAConfig(**config_dict["lora"]),
            finetuning=FineTuningConfig(**config_dict["finetuning"]),
        )
