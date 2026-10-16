# General Multiple Instance Learning Training Framework with Adversarial Training — V1

English | [简体中文](README.zh-CN.md)

A PyTorch-based workstation for pathology multiple instance learning (MIL). Its distinguishing feature is **support for adversarial training strategies for MIL models**, alongside natural training. V1 integrates feature-space PGD adversarial training into a unified workflow for pre-extracted pathology embeddings.

## Key Features

- **Unified MIL training:** shared model adapters, dataset interfaces, and training/evaluation loops.
- **MIL model integration:** shared adapters and a model registry, demonstrated here with gated ABMIL.
- **Natural and adversarial training:** standard optimization or PGD-based perturbation of patch features, with clean and adversarial evaluation.
- **Task-specific learning:** classification workflows and discrete-time survival modeling components.
- **Memory-conscious loading:** load feature bags per case, optionally subsample instances, and process one case per batch.
- **Evaluation and reporting:** classification metrics, survival concordance, fold summaries, prediction CSVs, and diagnostic plots.
- **Multimodal extensibility:** modality-aligned dataset interfaces for future pathology and omics modeling.

## Framework Overview

![Framework Overview](figs/fig1.png)

The framework connects data splits and feature loading to natural/adversarial training, MIL prediction, and evaluation. It operates on pre-extracted pathology embeddings, with survival modeling components and interfaces for future multimodal research.

## MIL Model Example: ABMIL

This README uses gated ABMIL to illustrate model integration and training. Its adapter is registered in [`utils/load_init_model.py`](utils/load_init_model.py):

| `--model_name` | Implementation | Notes |
| --- | --- | --- |
| `abmil` | ABMIL with gated attention | MLP feature projection, attention pooling, and a linear prediction head. |

ABMIL accepts patch features shaped `[1, N, D]`, applies attention pooling, and produces case-level predictions. For more MIL architectures and reference implementations, see [MIL-Lab](https://github.com/mahmoodlab/MIL-Lab).

## Tasks & Training Modes

| Task | Approach |
| --- | --- |
| Classification | Integer class labels, cross-entropy loss, and binary/multiclass evaluation. Custom labels can represent subtype, mutation status, or other case-level targets. |
| Survival modeling components | Discrete-time `NLLSurvLoss`, event/time attributes, hazard-derived risk scores, and censored concordance index. |

Set the task YAML's `task_type` and the corresponding `--task_type` option consistently. The examples below demonstrate classification with ABMIL.

- **Natural training:** omit `--adversarial_training`. Training uses clean feature bags, followed by clean validation and test evaluation.
- **Adversarial training:** add `--adversarial_training --perturb_config pgd_at --adv_training_types at`. Feature perturbations use random starts, normalized gradient steps, L2 projection, and activation-dependent scaling. Training uses perturbed bags, followed by adversarial validation and both clean/adversarial tests.

For multi-slide cases, `--slide_merge_mode list` averages per-slide logits; `concat` concatenates instances before encoding.

## Environment

Reference Conda environment:

| Component | Version |
| --- | --- |
| Python | `3.10.14` |
| PyTorch | `2.7.1+cu118` (CUDA 11.8) |

The framework runs on **CUDA** by default.

## Quick Start

### 1. Prepare feature bags and task metadata

For upstream pathology tissue segmentation, patching, and feature representation extraction, see [TRIDENT](https://github.com/mahmoodlab/TRIDENT). Prepare patch embeddings externally, then organize the features and metadata according to the HDF5 schema below.

Place one `<slide_id>.h5` file per slide directly in the feature directory, containing:

| HDF5 entry | Content |
| --- | --- |
| `features` | Patch embeddings with shape `[N, D]`; `D` must match `--input_feature_dim`. |
| `coords` | Patch coordinates aligned with the feature rows, typically `[N, 2]`. |
| `coords.attrs["patch_size_level0"]` | Numeric patch size at level 0. |

Use `test_task/t1_vs_t2/` as a classification metadata template and supply HDF5 feature bags with matching slide IDs.

- **Task YAML:** defines `sample_col`, `task_col`, `task_type`, `extra_cols`, `label_dict`, and `metrics`. Encode classification labels as integers `0` through `C-1`, consistent with `label_dict`.
- **Split TSV:** contains the sample column, `slide_id`, target column, extra attributes, and contiguous `fold_0`, `fold_1`, ... columns. Each fold should include `train`, `val`, and `test` subsets. Rows for the same case must have consistent labels and fold assignments.

Dataset transforms can be customized through the Python API.

### 2. Run natural training

Run from the repository root. Examples use Bash line continuation; replace the feature path and match `--input_feature_dim` to your embeddings.

```bash
python main_args.py \
  --mode train --task_type classification --model_name abmil \
  --split test_task/t1_vs_t2/k=all.tsv \
  --task_config test_task/t1_vs_t2/config.yaml \
  --modalities slide --slide_merge_mode list \
  --slide_embeddings_dirs /absolute/path/to/features \
  --input_feature_dim 1024 --embed_dim 512 \
  --batch_size 1 --gradient_accumulation 1 --bag_size None \
  --checkpoint_strategy best-val-loss \
  --results_dir_root ./mil_results/natural
```

This trains ABMIL across all five supplied folds, then reloads saved checkpoints for validation and testing.

### 3. Run adversarial training

```bash
python main_args.py \
  --mode train --task_type classification --model_name abmil \
  --split test_task/t1_vs_t2/k=all.tsv \
  --task_config test_task/t1_vs_t2/config.yaml \
  --modalities slide --slide_merge_mode list \
  --slide_embeddings_dirs /absolute/path/to/features \
  --input_feature_dim 1024 --embed_dim 512 \
  --batch_size 1 --gradient_accumulation 1 --bag_size None \
  --adversarial_training --perturb_config pgd_at --adv_training_types at \
  --adv_eps 1.0 --adv_alpha 0.2 --adv_steps 5 \
  --checkpoint_strategy best-val-loss \
  --results_dir_root ./mil_results/adversarial
```

For evaluation, reuse the matching command with `--mode test`. Keep the result root, model, dimensions, and task configuration consistent. Checkpoints load from `<results_dir_root>/<model_name>/checkpoints/fold_<i>/`, selecting the highest epoch when several exist.

### 4. Optional: generate new fold assignments

To generate new stratified case-level folds before training:

```bash
python get_split_args.py \
  --split test_task/t1_vs_t2/k=all.tsv \
  --task_config test_task/t1_vs_t2/config.yaml \
  --num_folds 5 --cv_type kfold_test --val_frac 0.2 --seed 1
```

The helper writes `k=all_new.tsv`, `folds_split.png`, and `labels_split.png` beside the input. Pass the new TSV to `--split`. The `kfold_test` strategy rotates the test fold and holds out validation cases from the remaining cases.

### Key options

| Option | Default / behavior |
| --- | --- |
| `--device`, `--seed` | `cuda`, `1`. |
| `--num_epochs` | `300` epochs per fold. |
| `--base_learning_rate`, `--weight_decay` | Both `1e-5`. |
| `--optimizer_type`, `--scheduler_type` | `AdamW` and `cosine`; use `--scheduler_type None` for a constant learning rate. |
| `--bag_size` | `None` uses all instances; a positive integer subsamples each bag. Sampling also applies during validation and testing. |
| `--batch_size`, `--gradient_accumulation` | Both default to `1`; the examples process one case per batch. |
| `--balanced` | Applies per-fold class weights to cross-entropy loss. |
| `--checkpoint_strategy` | `best-val-loss`, `every-K`, or `last-K`, e.g. `every-5` or `last-1`. Use a positive integer for `K`. |
| `--early_stopping` | Optional; uses validation loss with patience 5 after at least 10 epochs, within the `best-val-loss` path. |
| `--results_dir_root` | `./mil_results`; files are stored beneath `<model_name>`. Choose a separate root for each experiment. |
| `--view_progress` | `bar` or `verbose`. |

## Evaluation & Outputs

Classification reports include accuracy (`acc`), balanced accuracy (`bacc`), macro OvR/OvO ROC AUC, macro/weighted precision, recall and F1, quadratic weighted kappa, and support. Plots show ROC, precision-recall with PR area, and confusion matrices. Survival reporting computes censored concordance (`cindex`).

Multi-fold metrics are summarized as mean and standard error.

Outputs are written under `<results_dir_root>/<model_name>/`:

| Path | Contents |
| --- | --- |
| `experiment.txt`, `model.txt`, `split.csv` | Selected settings, model structure/parameter counts, and the split used. |
| `progress.txt` | Appended fold/epoch/phase progress records. |
| `checkpoints/fold_<i>/epoch_<e>.pt` | Model state dictionaries. |
| `loss_metrics/fold_<i>/` | Loss records in `loss_record.json` and plots in `loss_curve.png`. |
| `<setting>_test_probs/fold_<i>.csv` | Case IDs, labels, predicted classes and class probabilities; survival exports contain event, time, and risk. |
| `<setting>_<val-or-test>_metrics/fold_<i>/` | `metrics.json` and classification plots: `roc_curves.png`, `pr_curves.png`, `confusion_matrices.png`. |
| `<setting>_<val-or-test>_metrics_summary.json` | Aggregate fold statistics. |
| `pgd_at_e<eps>_a<alpha>_s<steps>_config.txt` | Attack configuration when adversarial mode is enabled. |

`<setting>` is `clean` or `adv_e<eps>_a<alpha>_s<steps>`.

## Project Structure

```text
.
├── main_args.py                 # Training and checkpoint evaluation CLI
├── get_split_args.py            # Fold generation and split visualizations
├── datasets/                    # Splits, HDF5 slide bags, labels, omics, transforms
├── models/
│   ├── slide_only_models/       # MIL implementations, base interface, attention layers
│   └── attack_models/           # Feature-space PGD implementation
├── losses/                      # Discrete-time survival NLL and Cox utilities
├── metrics/                     # Classification, survival, statistics, plots
├── utils/
│   ├── core/                   # Natural/adversarial wrappers and shared loops
│   ├── extensions/             # Serialization and representation helpers
│   ├── prepare_utils.py        # Dataset, loss, model, optimizer preparation
│   ├── load_init_model.py      # MIL model registry and adapters
│   ├── load_attack_model.py    # Attack registry and adapters
│   ├── get_split_task_utils.py # Task configuration and split loading
│   └── checkpoint_utils.py     # Checkpoint I/O and model inspection
├── test_task/                   # Classification and OS survival metadata examples
├── figs/fig1.png                # Framework overview
└── LICENSE.txt                 # MIT license
```

## Extensibility

- **MIL models:** implement the MIL interface in `models/slide_only_models/`, add a `BaseSlideEncoder` adapter, and register it in `utils/load_init_model.py`. Preserve the logits/log dictionary contract.
- **Training strategies:** extend the attack registry in `utils/load_attack_model.py`, the adversarial wrapper, and the shared training loop to introduce new perturbations and objectives.
- **Dataset interfaces:** customize feature transforms and compose case-aligned slide/omics datasets through the dataset API.
- **Future directions:** broader task coverage, including continuous-target modeling, and multimodal pathology/omics fusion.

## Version

**V1**
