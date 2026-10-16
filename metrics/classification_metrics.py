import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, f1_score, balanced_accuracy_score, precision_score, recall_score, cohen_kappa_score, classification_report
from sklearn.preprocessing import label_binarize

sns.set_style("white")
COLORS = [
    "#E64B35",
    "#4DBBD5",
    "#CDDDAA",
    "#F0E443",
    "#A020F0",
    "#FF69B4",
    "#8B4513",
    "#20B2AA",
    "#FFD700"
]

def auc_roc_plot(y_true, probs, num_classes, saveto=None):

    y_true_bin = get_binarize_labels(y_true, num_classes)


    fig, ax = plt.subplots(figsize=(7, 7))


    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)


    auc_scores = roc_auc_score(y_true_bin, probs, average=None, multi_class="ovr")


    for j in range(num_classes):

        fpr, tpr, _ = roc_curve(y_true_bin[:, j], probs[:, j])

        color = COLORS[j % len(COLORS)]

        ax.plot(fpr, tpr, linewidth=2.3,
                color=color,
                label=f"Class {j}  AUC={auc_scores[j]:.3f}")


    ax.plot([0,1], [0,1], linestyle="--", color="gray", linewidth=1)


    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(f"ROC Curve ({num_classes} classes)", fontsize=16)


    ax.legend(frameon=False, fontsize=12, loc="lower right")


    fig.tight_layout()

    if saveto is not None:

        os.makedirs(os.path.dirname(saveto), exist_ok=True)

        fig.savefig(saveto, dpi=300, bbox_inches="tight")

    plt.close(fig)

def precision_recall_plot(y_true, probs, num_classes, saveto=None):

    y_true_bin = get_binarize_labels(y_true, num_classes)


    fig, ax = plt.subplots(figsize=(7, 7))


    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)


    for j in range(num_classes):

        precision, recall, _ = precision_recall_curve(y_true_bin[:, j], probs[:, j])

        auc_pr = auc(recall, precision)

        color = COLORS[j % len(COLORS)]

        ax.plot(recall, precision, linewidth=2.3,
                color=color,
                label=f"Class {j}  AUC={auc_pr:.3f}")


        no_skill = sum(y_true_bin[:, j]) / len(y_true_bin[:, j])

        ax.plot([0, 1], [no_skill, no_skill], linestyle="--",
                color=color, linewidth=1)


    ax.set_xlabel("Recall", fontsize=14)

    ax.set_ylabel("Precision", fontsize=14)

    ax.set_title(f"Precision–Recall Curve ({num_classes} classes)", fontsize=16)


    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.legend(frameon=False, fontsize=12, loc="lower right")


    fig.tight_layout()

    if saveto is not None:

        os.makedirs(os.path.dirname(saveto), exist_ok=True)

        fig.savefig(saveto, dpi=300, bbox_inches="tight")

    plt.close()

def confusion_matrix_plot(y_true, probs, num_classes, threshold=None, saveto=None):

    preds_labels = get_predicted_labels(probs)

    cm = confusion_matrix(y_true, preds_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(7, 7))

    disp.plot(cmap=plt.cm.Blues)

    plt.title(f"Confusion Matrix")


    if saveto is not None:

        os.makedirs(os.path.dirname(saveto), exist_ok=True)

        plt.savefig(saveto, dpi=300, bbox_inches="tight")


    plt.close()

def classification_metrics_report(y_true, probs, num_classes, threshold=None, saveto=None):


    y_pred = get_predicted_labels(probs, threshold)


    scores = {"threshold": threshold}


    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=list(range(num_classes)))


    if num_classes == 2:
        probs = probs[:, 1]

    try:

        macro_ovr_auc = roc_auc_score(y_true, probs, average="macro", multi_class="ovr", labels = list(range(num_classes)))
        macro_ovo_auc = roc_auc_score(y_true, probs, average="macro", multi_class="ovo", labels = list(range(num_classes)))
    except Exception as e:

        print(f"WARNING: Could not compute AUC scores because {e}. Will set to None.")
        macro_ovr_auc = None
        macro_ovo_auc = None


    scores["overall"] = {
        "macro-ovr-auc": macro_ovr_auc,
        "macro-ovo-auc": macro_ovo_auc,
        "macro-precision": report["macro avg"]["precision"],
        "macro-recall": report["macro avg"]["recall"],
        "macro-f1": report["macro avg"]["f1-score"],
        "weighted-precision": report["weighted avg"]["precision"],
        "weighted-recall": report["weighted avg"]["recall"],
        "weighted-f1": report["weighted avg"]["f1-score"],
        "acc": np.mean(np.array(y_true) == np.array(y_pred)),
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "weighted_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "support": report["macro avg"]["support"]
    }


    if saveto is not None:
        os.makedirs(os.path.dirname(saveto), exist_ok=True)
        with open(saveto, "w") as f:
            json.dump(scores, f, indent=4)

    return scores


def get_binarize_labels(y_true, num_classes):
    if num_classes == 2:


        y_true_bin = label_binarize(y_true, classes=[0,1])
        return np.hstack((1 - y_true_bin, y_true_bin))
    else:

        return label_binarize(y_true, classes=range(num_classes))

def get_predicted_labels(probs, threshold=None):

    if threshold is not None:



        return np.array([np.argmax(row) if max(row) > threshold else -1 for row in probs])
    else:

        return np.argmax(probs, axis=1)