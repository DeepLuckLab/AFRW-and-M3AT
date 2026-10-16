import torch
import torch.nn as nn

class NLLSurvLoss(nn.Module):

    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, logits, y_bins, y_event):


        y_event = y_event.type(torch.int64)
        y_bins = y_bins.type(torch.int64)


        y_censor = 1 - y_event


        hazards = torch.sigmoid(logits)


        S = torch.cumprod(1 - hazards, dim=1)


        S_padded = torch.cat([torch.ones_like(y_censor), S], 1)


        s_prev = torch.gather(S_padded, dim=1, index=y_bins).clamp(min=self.eps)

        h_this = torch.gather(hazards, dim=1, index=y_bins).clamp(min=self.eps)

        s_this = torch.gather(S_padded, dim=1, index=y_bins + 1).clamp(min=self.eps)


        uncensored_loss = -(1 - y_censor) * (torch.log(s_prev) + torch.log(h_this))

        censored_loss = -y_censor * torch.log(s_this)


        loss = uncensored_loss + (1 - self.alpha) * censored_loss


        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}. Must be 'mean' or 'sum'.")