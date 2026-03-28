from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf
from monai import transforms
import importlib

def get_obj_from_str(string: str, reload: bool = False):
    module, cls = string.rsplit(".", 1)
    module_imp = importlib.import_module(module)
    if reload:
        importlib.reload(module_imp)
    return getattr(module_imp, cls)

def instantiate_from_config(config, *args, **kwargs):
    if isinstance(config, (DictConfig, ListConfig)):
        config = OmegaConf.to_container(config, resolve=True)

    if config is None:
        return None

    if isinstance(config, list):
        return [instantiate_from_config(item) for item in config]

    if isinstance(config, dict):
        if "_target_" in config:
            cls = get_obj_from_str(config["_target_"])
            init_args = {
                k: instantiate_from_config(v) for k, v in config.items() if k != "_target_"
            }
            return cls(*args, **init_args, **kwargs)
        else:
            return {k: instantiate_from_config(v) for k, v in config.items()}

    return config

def get_monai_transforms(cfg):
    trfs = []
    for t in cfg:
        transform_cls = instantiate_from_config(t)
        trfs.append(transform_cls)
    trfs = transforms.Compose(trfs)

    return trfs