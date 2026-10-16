import torch.nn as nn
from abc import ABC, abstractmethod


def init_attack_model_fun(perturb_config: str, **kwargs):

    if perturb_config in init_attack_registry:

        return init_attack_registry[perturb_config](**kwargs)
    else:
        raise ValueError(f"Unknown attack name {perturb_config}")

class BaseAttack(nn.Module, ABC):
    def __init__(self, **build_kwargs):
        super().__init__()

        self.at_model, self.attack_name = self._build(**build_kwargs)

    @abstractmethod
    def _build(self, **build_kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, mil_model, features, labels):
        raise NotImplementedError

class PGD_AT(BaseAttack):

    def __init__(self, eps=1.0, alpha=0.2, steps=5, random_start=True, **build_kwargs):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

        super().__init__(**build_kwargs)

    def _build(self):

        from models.attack_models.pgd_at import PGD_AT_Model, PGD_AT_BaseConfig
        self.attack_name = 'pgd_at'

        cfg = PGD_AT_BaseConfig(
            eps=self.eps,
            alpha=self.alpha,
            steps=self.steps,
            random_start=self.random_start
        )

        model = PGD_AT_Model(config=cfg)

        return model, self.attack_name

    def forward(self, mil_model,features, labels):

        adv_features = self.at_model(mil_model, features, labels)

        return adv_features

init_attack_registry = {
    'pgd_at': PGD_AT
}