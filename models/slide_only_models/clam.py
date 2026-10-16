

import torch
from typing import Union
import torch.nn as nn
import torch.nn.functional as F

from models.slide_only_models.mil_template import MIL
from models.slide_only_models.some_layers import GlobalGatedAttention, GlobalAttention, build_embed


class CLAMSB(MIL):

    def __init__(self,
                 in_dim: int = 1024,
                 embed_dim: int = 512,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 gate: bool = True,
                 attention_dim: int = 384,
                 num_classes: int = 2,
                 k_sample: int = 8,
                 subtyping: bool = False,
                 instance_loss_fn: str = 'svm',
                 bag_weight: float = 0.7,
                 embed_type: str = "mlp",
                 extra_args=None):

        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        self.k_sample = k_sample
        self.subtyping = subtyping
        self.bag_weight = bag_weight

        self.patch_embed = build_embed(embed_type, in_dim, embed_dim, num_fc_layers, dropout, extra_args)

        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(L=embed_dim,
                                       D=attention_dim,
                                       dropout=dropout,
                                       num_classes=1)

        self.classifier = nn.Linear(embed_dim, num_classes)

        instance_classifiers = [nn.Linear(embed_dim, 2) for _ in range(num_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        if instance_loss_fn == 'svm':
            try:
                import topk
                from topk.svm import SmoothTop1SVM
            except:
                print(f'SmoothTop1SVM not found. '
                      f'Please install it via pip: pip install git+https://github.com/oval-group/smooth-topk')

            try:
                self.instance_loss_fn = SmoothTop1SVM(n_classes=2).cuda()
            except AssertionError:
                raise AssertionError('SmoothTop1SVM requires CUDA to be available. If not available, please use the ce loss function')
        else:
            self.instance_loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def create_positive_targets(length: int, device: torch.device) -> torch.LongTensor:
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length: int, device: torch.device) -> torch.LongTensor:
        return torch.full((length,), 0, device=device).long()

    def _check_inputs(self, features, loss_fn, label):
        if features.dim() == 3 and features.shape[0] > 1:
            raise ValueError(f'CLAM does not currently support batch size > 1')
        if label is None:
            raise ValueError(f'Label must be provided for CLAM')

    def inst_eval(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(A.shape) == 1:
            A = A.view(1, -1)

        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)

        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)

        p_targets = self.create_positive_targets(self.k_sample, h.device)
        n_targets = self.create_negative_targets(self.k_sample, h.device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)

        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)

        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:

        if len(A.shape) == 1:
            A = A.view(1, -1)

        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)

        p_targets = self.create_negative_targets(self.k_sample, h.device)

        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward_attention(self, h: torch.Tensor, attention_only: bool = False) -> Union[torch.Tensor, tuple[
        torch.Tensor, torch.Tensor]]:

        h = self.patch_embed(h).squeeze()


        A = self.global_attn(h)
        A = torch.transpose(A, 1, 0)

        if attention_only:
            return A
        else:
            return h, A

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:

        logits = self.classifier(h)
        return logits


    def forward_features(self, h: torch.Tensor, return_raw_attention: bool = True) -> tuple[
        torch.Tensor, dict]:

        h_embedded, attention_raw = self.forward_attention(h)
        intermeds = {'instance_feats': h_embedded, 'attention_raw': attention_raw}

        attention_scaled = F.softmax(attention_raw, dim=-1)

        M = torch.mm(attention_scaled, h_embedded)


        return M, intermeds


    def forward_instance_heads(self, h: torch.Tensor, attention_scores: torch.Tensor,
                               label: torch.LongTensor = None) -> Union[torch.Tensor, None]:

        if label is None:
            return None
        total_inst_loss = 0.0

        inst_labels = F.one_hot(label, num_classes=self.num_classes).squeeze(0).to(label.device)

        for i in range(len(self.instance_classifiers)):
            inst_label_for_class = inst_labels[i].item()
            classifier = self.instance_classifiers[i]

            if inst_label_for_class == 1:
                instance_loss, _, _ = self.inst_eval(attention_scores, h, classifier)
                total_inst_loss += instance_loss
            else:
                if self.subtyping:
                    instance_loss, _, _ = self.inst_eval_out(attention_scores, h, classifier)
                    total_inst_loss += instance_loss

        if self.subtyping and len(self.instance_classifiers) > 0:
            total_inst_loss /= len(self.instance_classifiers)
        elif not self.subtyping and inst_labels.sum().item() > 0:
            total_inst_loss /= inst_labels.sum().item()
        elif total_inst_loss == 0 and inst_labels.sum().item() == 0:
            return None

        return total_inst_loss

    def compute_total_loss(self, logits: torch.Tensor, label: torch.LongTensor, loss_fn: nn.Module,
                           inst_loss: torch.Tensor) -> torch.Tensor:

        cls_loss = self.compute_loss(loss_fn, logits, label)

        if inst_loss is not None and cls_loss is not None:

            loss = cls_loss * self.bag_weight + (1 - self.bag_weight) * inst_loss
        elif inst_loss is not None:
            loss = inst_loss
        else:
            loss = cls_loss
        return loss

    def forward(self, h: torch.Tensor,
                return_raw_attention: bool = False,
                return_intermediate_feats: bool = False,
                label: torch.LongTensor = None,
                loss_fn: torch.nn.Module = None,
                attn_mask=None):

        self._check_inputs(h, loss_fn, label)

        slide_feats, intermeds = self.forward_features(h, return_raw_attention=return_raw_attention)

        logits = self.forward_head(slide_feats)

        inst_loss = self.forward_instance_heads(intermeds['instance_feats'], intermeds['attention_raw'], label)

        total_loss = self.compute_total_loss(logits, label, loss_fn, inst_loss)

        log_dict = {
            'instance_loss': inst_loss if inst_loss is not None else None,
            'cls_loss': loss_fn(logits, label) if loss_fn is not None and label is not None else None,
            'loss': total_loss.item()
        }

        results_dict = {'logits': logits}
        log_dict['attention_raw'] = intermeds['attention_raw'].unsqueeze(0) if return_raw_attention else None
        log_dict['slide_feats'] = slide_feats if return_intermediate_feats else None

        return results_dict, log_dict

class CLAMConfig:
    def __init__(self,
        gate: bool = True,
        in_dim: int = 1024,
        num_classes: int = 2,
        embed_dim: int = 512,
        attention_dim: int = 384,
        num_fc_layers: int = 1,
        dropout: float = 0.25,
        k_sample: int = 8,
        subtyping: bool = False,
        instance_loss_fn: str = 'svm',
        bag_weight: float = 0.7,
        embed_type="mlp",
        extra_args=None):

        self.gate = gate
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.k_sample = k_sample
        self.subtyping = subtyping
        self.instance_loss_fn = instance_loss_fn
        self.bag_weight = bag_weight
        self.embed_type = embed_type
        self.extra_args = extra_args

class CLAMModel(nn.Module):
    def __init__(self, config: CLAMConfig):
        self.config = config
        super().__init__()

        self.model = CLAMSB(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            gate=config.gate,
            attention_dim=config.attention_dim,
            num_classes=config.num_classes,
            k_sample=config.k_sample,
            subtyping=config.subtyping,
            instance_loss_fn=config.instance_loss_fn,
            bag_weight=config.bag_weight,
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
    cfg = CLAMConfig(in_dim=1024,
                      embed_dim=1024,
                      num_fc_layers=1,
                      dropout=0.25,
                      k_sample=8,
                      subtyping=False,
                      instance_loss_fn='svm',
                      gate=True,
                      bag_weight=0.7,
                      attention_dim=128,
                      num_classes=4,
                      embed_type = "mlp",
                      extra_args = swe_kwargs)
    loss = nn.CrossEntropyLoss()
    model = CLAMModel(config=cfg)
    results_dict, log_dict = model(h=x, label=y, return_raw_attention=True, return_intermediate_feats=True, loss_fn=loss)
    print(results_dict['logits'].shape)
    print(log_dict['attention_raw'].shape)
    print(log_dict['loss'])
    print(log_dict['slide_feats'].shape)

