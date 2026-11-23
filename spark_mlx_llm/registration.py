import importlib
import sys
from types import ModuleType

MODEL_MODULE = "mlx_lm.models.spark2_5"
SPARK3_MODEL_MODULE = "mlx_lm.models.spark3"


def _register(module_name: str, local_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise

    model = importlib.import_module(local_name, package=__package__)
    sys.modules[module_name] = model
    return model


def register_model() -> ModuleType:
    """Expose both Spark architectures; retain the Spark2_5 return value."""
    spark2_5 = _register(MODEL_MODULE, ".spark2_5")
    _register(SPARK3_MODEL_MODULE, ".spark3")
    return spark2_5
