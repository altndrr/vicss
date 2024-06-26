import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict


def test_train_config(cfg_train: DictConfig):
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    data = hydra.utils.instantiate(cfg_train.data)
    with open_dict(cfg_train):
        cfg_train.model.task = data.task
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig):
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    data = hydra.utils.instantiate(cfg_eval.data)
    with open_dict(cfg_eval):
        cfg_eval.model.task = data.task
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)
