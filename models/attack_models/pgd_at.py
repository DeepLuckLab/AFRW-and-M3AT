import torch
import torch.nn as nn
from losses.nll_loss import NLLSurvLoss

class PGD_AT(nn.Module):

    def __init__(self, eps=1, alpha=0.2, steps=5, random_start=True):
        super().__init__()
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.random_start = random_start

    def forward(self, mil_model, features, labels):

        try:
            device = next(mil_model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Warning: Failed to set device automatically, defaulting to {device}")

        with torch.enable_grad():

            ori_features = features.clone().detach().to(device)

            if isinstance(labels, list):
                y_bins = labels[0].to(device)
                y_bins = y_bins % 4
                y_event = labels[1]['OS_event'].to(device)
                loss = NLLSurvLoss(alpha=0.5, eps=1e-7, reduction="mean")
                labels = labels[0].clone().detach().to(device)
                task_type = "survival"
            else:
                labels = labels.clone().detach().to(device)
                loss = nn.CrossEntropyLoss()
                task_type = "classification"

            min_act = torch.min(ori_features)
            max_act = torch.max(ori_features)

            activation_std = torch.std(ori_features, dim=1, keepdim=True) + 1e-6
            mean_std = activation_std.mean()
            normalization = (activation_std / mean_std).detach()

            if self.random_start:
                current_perturbation = torch.empty_like(ori_features).normal_().to(device)
                n = torch.norm(current_perturbation, p=2, dim=-1, keepdim=True)
                r = torch.zeros_like(n).uniform_(0, 1)
                current_perturbation *= ((r / (n + 1e-8)) * self.eps)
            else:
                current_perturbation = torch.zeros_like(ori_features).to(device)

            def project_scale_clip(_current_perturbation):
                perturbation_norms = torch.norm(_current_perturbation, p=2, dim=-1, keepdim=True)
                factor = self.eps / (perturbation_norms + 1e-8)
                factor = torch.min(factor, torch.ones_like(perturbation_norms))
                _current_perturbation = _current_perturbation * factor
                adv_latents = torch.clamp(ori_features + _current_perturbation,
                                          min=min_act,
                                          max=max_act).detach()
                _current_perturbation = (adv_latents - ori_features).detach()
                return _current_perturbation

            current_perturbation = project_scale_clip(current_perturbation)

            for _ in range(self.steps):

                current_perturbation.requires_grad = True

                adv_features = ori_features + current_perturbation * normalization

                results_dict, log_dict = mil_model(features=adv_features,
                                         label=labels,
                                         return_raw_attention=False,
                                         return_intermediate_feats=False,
                                         loss_fn=None,
                                         attn_mask=None)

                outputs = results_dict['logits']

                if task_type == "survival":
                    cost = loss(outputs, y_bins.unsqueeze(0), y_event.unsqueeze(0))
                elif task_type == "classification":
                    cost = loss(outputs, labels)
                else:
                    raise ValueError(f"Unknown task type: {task_type}")

                grad = torch.autograd.grad(
                    cost, current_perturbation, retain_graph=False, create_graph=False
                )[0]

                grad_norm = torch.norm(grad, p=2, dim=-1, keepdim=True) + 1e-8
                grad = grad / grad_norm

                current_perturbation = current_perturbation.detach() + self.alpha * grad
                current_perturbation = project_scale_clip(current_perturbation)

            adv_features = (ori_features + current_perturbation * normalization).detach()

        return adv_features

class PGD_AT_BaseConfig:
    def __init__(self,
                 eps: float = 1.0,
                 steps: int = 5,
                 alpha: float = 0.2,
                 random_start: bool = True):

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start


class PGD_AT_Model(nn.Module):
    def __init__(self, config: PGD_AT_BaseConfig):
        super().__init__()
        self.config = config
        self.model = PGD_AT(
            eps=config.eps,
            steps=config.steps,
            alpha=config.alpha,
            random_start=config.random_start
        )
        self.forward = self.model.forward
