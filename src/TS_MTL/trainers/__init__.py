from .ts_mtl.normal_trainer import HardParameterSharingTrainer
from .ts_mtl.ca_grad_trainer import CAGradTrainer
from .ts_mtl.gradient_balancing_trainer import GradientBalancingTrainer
from .ts_mtl.pc_grad_trainer import PCGradTrainer
from .ts_diff.ts_diff_trainer import TSDiffTrainer
from .ts_diff.grad_bal_trainer import TSDiffGradBalTrainer  # Placeholder for future implementation
TRAINER_REGISTRY = {
    "normal_trainer": HardParameterSharingTrainer,
    "cagrad_trainer": CAGradTrainer,
    "gradient_balancing_trainer": GradientBalancingTrainer,
    "pc_grad_trainer": PCGradTrainer,
    "ts_diff_trainer": TSDiffTrainer,
    "ts_diff_grad_bal_trainer": TSDiffGradBalTrainer
}