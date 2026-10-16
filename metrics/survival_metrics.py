import os
import json
import torch
from sksurv.metrics import concordance_index_censored

def survival_metrics_report(survival_events, survival_times, preds, saveto = None):


    survival_events = survival_events.astype(bool)


    assert preds.ndim == 1, f"Predictions must be 1D, got shape {preds.shape}"

    num_samples = preds.shape[0]

    assert survival_events.shape == (num_samples,), f"Expected shape ({num_samples},) for survival_events, got {survival_events.shape}"

    assert survival_times.shape == (num_samples,), f"Expected shape ({num_samples},) for survival_times, got {survival_times.shape}"


    metrics = {
        'cindex': concordance_index_censored(survival_events, survival_times, preds, tied_tol=1e-08)[0],
    }


    if saveto is not None:

        os.makedirs(os.path.dirname(saveto), exist_ok=True)

        with open(saveto, 'w') as f:
            json.dump(metrics, f, indent=4)


    return metrics

def calculate_risk(logits):
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1 - hazards, dim=0)
    risk = -torch.sum(survival, dim=0)
    return risk