import os
import json
import matplotlib.pyplot as plt
import numpy as np
import csv

def extract_survival_label(case_info, task_name):


    time_key = f"{task_name}_months"
    event_key = f"{task_name}_event"


    t_list = case_info[time_key]
    e_list = case_info[event_key]


    if isinstance(t_list, (list, tuple)):
        time = float(set(t_list).pop())
    else:
        time = float(t_list)


    if isinstance(e_list, (list, tuple)):
        event = int(set(e_list).pop())
    else:
        event = int(e_list)

    return time, event


def extract_classification_label(case_info, task_name):


    labels_dict = case_info.get("labels", {})


    label = labels_dict[task_name]


    if isinstance(label, (list, tuple)):

        label = set(label)
        return int(label.pop())
    else:
        return int(label)


def save_fold_prob_pred(
        output_dir: str,
        fold_idx: int,
        case_ids,
        labels,
        probs,
        preds,
        task_type: str,
        save_key: str
):



    folder_name = f"{save_key}_test_probs"
    setting_dir = os.path.join(output_dir, folder_name)
    os.makedirs(setting_dir, exist_ok=True)


    save_path = os.path.join(setting_dir, f"fold_{fold_idx}.csv")


    file_mode = "w"

    with open(save_path, file_mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)


        if task_type == "classification":
            num_classes = probs.shape[1]
            header = ["fold", "case_id", "label", "pred"] + \
                     [f"prob_{i}" for i in range(num_classes)]
        elif task_type == "survival":
            header = [
                "fold", "case_id",
                "survival_time", "survival_event",
                "risk_score"
            ]
        else:
            raise ValueError(f"Unknown task_type={task_type}")

        writer.writerow(header)


        num_samples = len(case_ids)
        for idx in range(num_samples):
            case_id = case_ids[idx]

            if task_type == "classification":
                label_i = int(labels[idx])
                pred_i = int(preds[idx])
                prob_list = probs[idx].reshape(-1).tolist()
                prob_list = [float(p) for p in prob_list]

                row = [fold_idx, case_id, label_i, pred_i] + prob_list

            elif task_type == "survival":
                surv_time = float(labels["survival_time"][idx])
                surv_event = int(labels["survival_event"][idx])
                risk_value = float(np.array(probs[idx]).reshape(-1)[0])

                row = [fold_idx, case_id, surv_time, surv_event, risk_value]
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            writer.writerow(row)

def smooth_curve(values, smooth=0.6):

    smoothed = []
    last = values[0]
    for v in values:

        last = smooth * last + (1 - smooth) * v
        smoothed.append(last)
    return smoothed

def get_loss(args, current_iter, train_losses, val_losses, smooth_alpha=0.1):

    output_dir = os.path.join(args.results_dir, "loss_metrics", f"fold_{current_iter}")
    os.makedirs(output_dir, exist_ok=True)


    loss_record_path = os.path.join(output_dir, "loss_record.json")
    with open(loss_record_path, "w") as f:
        json.dump({
            "train_loss": train_losses,
            "val_loss": val_losses
        }, f, indent=4)


    sm_train = smooth_curve(train_losses, smooth=smooth_alpha)
    epochs = np.arange(len(train_losses))

    plt.figure(figsize=(6, 6))
    ax = plt.gca()


    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_linewidth(1.8)


    ax.plot(epochs, sm_train, color="#4DBBD5", linewidth=2.2,
            label="Train Loss", marker='o', markersize=5)


    if len(val_losses) > 0:
        sm_val = smooth_curve(val_losses, smooth=smooth_alpha)
        ax.plot(epochs, sm_val, color="#E64B35", linewidth=2.2,
                label="Val Loss", marker='s', markersize=5)
        title = f"Training & Validation Loss Curve (Fold {current_iter})"
    else:
        title = f"Training Loss Curve Only (Fold {current_iter})"


    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.legend(frameon=False)


    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def bootstrap(all_labels_across_folds, all_preds_across_folds, num_bootstraps):


    all_labels_across_folds, all_preds_across_folds = np.concatenate(all_labels_across_folds), np.concatenate(all_preds_across_folds)

    bootstraps = []

    for _ in range(num_bootstraps):

        idx = np.random.choice(len(all_labels_across_folds), len(all_labels_across_folds), replace=True)

        labels = np.array([all_labels_across_folds[i] for i in idx])
        preds = np.array([all_preds_across_folds[i] for i in idx])

        bootstraps.append((labels, preds))
    return bootstraps

def get_95_ci(all_scores_across_folds):

    summary = {}
    for key in all_scores_across_folds[0].keys():

        if any([score[key] is None for score in
                all_scores_across_folds]):

            summary[key] = {
                'mean': None,
                'lower': None,
                'upper': None,
                'formatted': 'N/A'
            }
            continue

        mean = np.mean([score[key] for score in all_scores_across_folds])

        lower = np.percentile([score[key] for score in all_scores_across_folds], 2.5)

        upper = np.percentile([score[key] for score in all_scores_across_folds], 97.5)

        summary[key] = {
            'mean': mean,
            'lower': lower,
            'upper': upper,
            'formatted': f'{mean:.3f} ({lower:.3f}-{upper:.3f})'
        }

    return summary

def get_mean_se(all_scores_across_folds):

    summary = {}

    for key in all_scores_across_folds[0].keys():

        mean = np.mean([score[key] for score in all_scores_across_folds if score[key] is not None])

        std = np.std([score[key] for score in all_scores_across_folds if score[key] is not None])

        se = std / np.sqrt(len(all_scores_across_folds))

        summary[key] = {
            'mean': mean,
            'se': se,
            'formatted': f'{mean:.3f} ± {se:.3f}'
        }

    return summary


def print_summary(
        title=None,
        summary=None,
        table=False
):

    if title is not None:
        line = "=" * (len(title) + 45)
        print(f"\n{line}")
        print(title.center(len(line)))
        print(f"{line}")

    if summary is None:
        return

    if table:
        print("+--------------------------+----------------------+")
        print("|        Metric            |       Value          |")
        print("+--------------------------+----------------------+")
        for k, v in summary.items():
            val = v["formatted"] if isinstance(v, dict) and "formatted" in v else v
            print(f"| {k:<24} | {str(val):<20} |")
        print("+--------------------------+----------------------+\n")

    else:
        for k, v in summary.items():
            val = v["formatted"] if isinstance(v, dict) and "formatted" in v else v
            print(f"{k}: {val}")


