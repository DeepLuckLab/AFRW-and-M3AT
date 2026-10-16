import torch.nn as nn
import torch
import numpy as np

class CoxLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, logits, times, censorships):
        return partial_ll_loss(lrisks=logits, survival_times=times, event_indicators=censorships.float())

def partial_ll_loss(lrisks, survival_times, event_indicators):


    num_uncensored = torch.sum(event_indicators, 0)

    if num_uncensored.item() == 0:
        return {'loss': torch.sum(lrisks) * 0}


    survival_times = survival_times.squeeze(1)
    event_indicators = event_indicators.squeeze(1)
    lrisks = lrisks.squeeze(1)


    sindex = torch.argsort(-survival_times)

    survival_times = survival_times[sindex]
    event_indicators = event_indicators[sindex]
    lrisks = lrisks[sindex]


    log_risk_stable = torch.logcumsumexp(lrisks, 0)


    likelihood = lrisks - log_risk_stable

    uncensored_likelihood = likelihood * event_indicators

    logL = -torch.sum(uncensored_likelihood)


    return {'loss': logL / num_uncensored}





