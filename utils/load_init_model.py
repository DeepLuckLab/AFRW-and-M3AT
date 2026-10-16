import torch
from abc import abstractmethod

embed_type = "mlp"

def init_model_fun(model_name: str, **kwargs) -> torch.nn.Module:

    if model_name in init_model_registry:

        return init_model_registry[model_name](**kwargs)
    else:

        raise ValueError(f"Unknown encoder name {model_name}")

class BaseSlideEncoder(torch.nn.Module):

    def __init__(self, **build_kwargs: dict) -> None:

        super().__init__()

        self.enc_name = None
        self.model, self.precision, self.embedding_dim = self._build(**build_kwargs)

    def forward(self, features, return_raw_attention, return_intermediate_feats, label, loss_fn, attn_mask):
        pass

    @abstractmethod
    def _build(self, **build_kwargs):
        pass

class ABMILSlideEncoder(BaseSlideEncoder):

    def __init__(self, **build_kwargs):

        super().__init__(**build_kwargs)

    def _build(
            self,
            input_feature_dim: int = 1024,
            embed_dim: int = 512,
            num_classes: int = 2
    ):

        from models.slide_only_models.abmil import ABMILModel, ABMILGatedBaseConfig
        self.enc_name = 'abmil'

        cfg = ABMILGatedBaseConfig(in_dim=input_feature_dim,
                                   embed_dim=embed_dim,
                                   num_fc_layers=1,
                                   dropout=0.25,
                                   gate=True,
                                   attn_dim=128,
                                   num_classes=num_classes,
                                   embed_type = embed_type,
                                   extra_args = None)

        model = ABMILModel(config=cfg)
        precision = torch.float32
        embedding_dim = embed_dim
        return model, precision, embedding_dim

    def forward(self, features, return_raw_attention, return_intermediate_feats, label, loss_fn, attn_mask):
        results_dict, log_dict = self.model(features,
                                            return_raw_attention,
                                            return_intermediate_feats,
                                            label,
                                            loss_fn,
                                            attn_mask=None)
        return results_dict, log_dict

class CLAMSlideEncoder(BaseSlideEncoder):

    def __init__(self, **build_kwargs):
        super().__init__(**build_kwargs)

    def _build(
            self,
            input_feature_dim: int = 1024,
            embed_dim: int = 512,
            num_classes: int = 2
    ):

        from models.slide_only_models.clam import CLAMModel, CLAMConfig
        self.enc_name = 'clam'

        cfg = CLAMConfig(in_dim=input_feature_dim,
                         embed_dim=embed_dim,
                         num_fc_layers=1,
                         dropout=0.25,
                         k_sample=8,
                         subtyping=False,
                         instance_loss_fn='svm',
                         gate=True,
                         bag_weight=0.7,
                         attention_dim=256,
                         num_classes=num_classes,
                         embed_type = embed_type,
                         extra_args = None)
        model = CLAMModel(config=cfg)

        precision = torch.float32
        embedding_dim = embed_dim
        return model, precision, embedding_dim

    def forward(self, features, return_raw_attention, return_intermediate_feats, label, loss_fn, attn_mask):
        results_dict, log_dict = self.model(features,
                                            return_raw_attention,
                                            return_intermediate_feats,
                                            label,
                                            loss_fn,
                                            attn_mask=None)
        return results_dict, log_dict


init_model_registry = {
    'abmil': ABMILSlideEncoder,
    'clam':CLAMSlideEncoder
}
