from .normal_trainer import HardParameterSharingTrainer

TRAINER_REGISTRY = {
    "normal_trainer": HardParameterSharingTrainer,
}