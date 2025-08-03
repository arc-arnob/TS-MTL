from .ts_mtl import HardParameterSharingModel
from .ts_diff import TSDiffModel, DiffusionProcess
MODEL_REGISTRY = {
    "hard_sharing": HardParameterSharingModel,
    "ts_diff": TSDiffModel,
    "diffusion_process": DiffusionProcess,
}