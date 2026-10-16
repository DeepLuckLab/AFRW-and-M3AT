

from models.slide_only_models.mil_template import MIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.slide_only_models.some_layers import GlobalAttention, GlobalGatedAttention, build_embed

MODEL_TYPE = 'abmil'

class ABMIL(MIL):

    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            num_fc_layers: int = 1,
            dropout: float = 0.25,
            attn_dim: int = 384,
            gate: int = True,
            num_classes: int = 2,
            embed_type: str = "mlp",
            extra_args=None
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        self.patch_embed = build_embed(embed_type, in_dim, embed_dim, num_fc_layers, dropout, extra_args)

        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1
        )

        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only=True) -> torch.Tensor:

        h = self.patch_embed(h)
        A = self.global_attn(h)
        A = torch.transpose(A, -2, -1)
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min

        if attn_only:
            return A
        return h, A

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_raw_attention: bool = True) -> torch.Tensor:

        h, A_raw = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)
        A = F.softmax(A_raw, dim=-1)
        h = torch.bmm(A, h).squeeze(dim=1)
        log_dict = {'attention_raw': A_raw if return_raw_attention else None}
        return h, log_dict

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:

        logits = self.classifier(h)
        return logits

    def forward(self, h: torch.Tensor,
                return_raw_attention: bool = False,
                return_intermediate_feats: bool = False,
                label: torch.LongTensor = None,
                loss_fn: torch.nn.Module = None,
                attn_mask=None):

        wsi_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_raw_attention=return_raw_attention)
        logits = self.forward_head(wsi_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits}

        log_dict['loss'] = cls_loss.item() if cls_loss is not None else None
        log_dict['slide_feats'] = wsi_feats if return_intermediate_feats else None

        return results_dict, log_dict


class ABMILGatedBaseConfig:
    def __init__(self,
                 gate: bool = True,
                 embed_dim: int = 512,
                 attn_dim: int = 384,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 in_dim: int = 1024,
                 num_classes: int = 2,
                 embed_type="mlp",
                 extra_args=None):
        self.gate = gate
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.embed_type = embed_type
        self.extra_args = extra_args

class ABMILModel(nn.Module):
    def __init__(self, config: ABMILGatedBaseConfig):
        super().__init__()

        self.config = config
        self.model = ABMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            attn_dim=config.attn_dim,
            gate=config.gate,
            num_classes=config.num_classes,
            embed_type=config.embed_type,
            extra_args=config.extra_args
        )
        self.forward = self.model.forward

if __name__ == '__main__':
    x = torch.randn(1,  666, 1024)
    y = torch.tensor([2])
    swe_kwargs = {
        'num_ref_points': 128,
        'return_attn': False
    }
    cfg = ABMILGatedBaseConfig(in_dim = 1024,
                      embed_dim = 512,
                      num_fc_layers = 1,
                      dropout = 0.25,
                      gate=True,
                      attn_dim = 128,
                      num_classes=4,
                      embed_type = "mlp",
                      extra_args = swe_kwargs
    )
    loss = nn.CrossEntropyLoss()
    model = ABMILModel(config=cfg)

    results_dict, log_dict = model(h=x, label=y, return_raw_attention=True, return_intermediate_feats=True, loss_fn=loss)
    print(model)
    print(results_dict['logits'].shape)
    print(log_dict['attention_raw'].shape)
    print(log_dict['loss'])
    print(log_dict['slide_feats'].shape)

