from .non_fed_baselines.ts_mtl import HardParameterSharingModel
from .non_fed_baselines.ts_diff import TSDiffModel, DiffusionProcess
from .non_fed_baselines.arimax_models import ARIMAXIndependentModel, ARIMAXGlobalModel

MODEL_REGISTRY = {
    "hard_sharing": HardParameterSharingModel,
    "ts_diff": TSDiffModel,
    "diffusion_process": DiffusionProcess,
    "arimax_independent": ARIMAXIndependentModel,
    "arimax_global": ARIMAXGlobalModel,
}