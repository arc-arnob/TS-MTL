from .normal_trainer import HardParameterSharingTrainer
from .ca_grad_trainer import CAGradTrainer
from .gradient_balancing_trainer import GradientBalancingTrainer
TRAINER_REGISTRY = {
    "normal_trainer": HardParameterSharingTrainer,
    "cagrad_trainer": CAGradTrainer,
    "gradient_balancing_trainer": GradientBalancingTrainer
}