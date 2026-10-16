import os
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from IPython.display import display
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from utils.extensions.JSONSaver import JSONsaver
from pprint import pformat

class DataSplit:
    def __init__(self,
                 path: str,
                 id_col: str,
                 attr_cols: list,
                 label_cols: list,
                 skip_labels: dict = None,
                 ignore_ids: list = None,
                 verbose: bool = True):

        self.path = path
        self.id_col = id_col
        self.attr_cols = attr_cols
        self.label_cols = label_cols
        self.skip_labels = skip_labels
        self.ignore_ids = ignore_ids

        assert os.path.exists(self.path), f'{self.path} does not exist.'
        self.data = pd.read_csv(self.path, sep='\t' if self.path.endswith('.tsv') else ',', dtype={'case_id': str, 'slide_id': str, 'id': str})

        self.data = self.convert_to_json(self.data)
        self.num_folds = len(self.data[0]['folds']) if 'folds' in self.data[0] else 0

        if verbose:
            print(f'Loaded split from {self.path} with {len(self.data)} samples and {self.num_folds} folds assigned.')

    def convert_to_json(self, df):

        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

        df = df.dropna(subset=self.label_cols)
        if self.skip_labels:
            for label_name, skip_vals in self.skip_labels.items():
                df = df[~df[label_name].isin(skip_vals)]

        samples_list = []
        for sample_id in df[self.id_col].unique():

            sample_df = df[df[self.id_col] == sample_id]
            sample_dict = {'id': str(sample_id)}

            if self.ignore_ids and str(sample_id) in [str(id) for id in self.ignore_ids]:
                print(f'Ignoring sample with ID {sample_id}.')
                continue

            if self.label_cols:

                labels = {col: sample_df[col].tolist() for col in self.label_cols}

                unique_label_sets = sample_df.drop_duplicates(subset=self.label_cols, keep='first')
                assert len(unique_label_sets) == 1, f'Inconsistent labels found for sample(s) with ID {sample_id}.\n{unique_label_sets}'

                sample_dict['labels'] = {k:v[0] for k, v in labels.items()}

            folds = []
            fold_columns = [col for col in sample_df.columns if 'fold_' in col]
            if len(fold_columns) > 0:
                fold_columns.sort(key=lambda x: int(x.split('_')[-1]))
                for col in fold_columns:
                    folds_for_sample = set(sample_df[col].values)
                    assert len(folds_for_sample) == 1, f'Inconsistent {col} found for sample ID {sample_id}.'
                    folds.append(folds_for_sample.pop())
                sample_dict['folds'] = folds

            if self.attr_cols:
                for attr_col in self.attr_cols:
                    sample_dict[attr_col] = sample_df[attr_col].tolist()

            samples_list.append(sample_dict)

        return samples_list

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        from pprint import pformat
        msg = (
            f"<SplitObject samples={len(self.data)}, folds={self.num_folds}>\n"
            f"First samples:\n{pformat(self.data[:5], indent=4)}"
        )
        return msg

    def __add__(self, other):

        if not isinstance(other, DataSplit):
            raise TypeError(f"Cannot add Split with type {type(other)}")

        if self.label_cols != other.label_cols:
            raise ValueError("Cannot add Splits with different label columns.")
        if self.num_folds != other.num_folds:
            raise ValueError("Cannot add Splits with different numbers of folds.")

        self_ids = set([sample['id'] for sample in self.data])
        other_ids = set([sample['id'] for sample in other.data])
        if len(self_ids.intersection(other_ids)) > 0:
            raise ValueError(f"Cannot add Splits with overlapping IDs.")

        combined_data = self.data + other.data

        new_split = DataSplit(path=self.path, id_col=self.id_col, attr_cols=self.attr_cols, label_cols=self.label_cols, skip_labels=self.skip_labels, ignore_ids=self.ignore_ids)
        new_split.data = combined_data
        new_split.num_folds = self.num_folds

        return new_split

    def convert_labels_to_integers(self, label_attr, int_to_label = None):

        if int_to_label is None:
            unique_labels = set([sample['labels'][label_attr] for sample in self.data])

            label_to_int = {label: i for i, label in enumerate(sorted(unique_labels))}
            print('Generated int-to-label mapping:')
            display({str(v): k for k, v in label_to_int.items()})
        else:
            print('Using provided int-to-label mapping:')
            display(int_to_label)
            label_to_int = {v: int(k) for k, v in int_to_label.items()}

        for sample in self.data:
            assert sample['labels'][label_attr] in label_to_int, f'Label {sample["labels"][label_attr]} (dtype {type(sample["labels"][label_attr])}) not found in label-to-int mapping.'
            sample['labels'][label_attr] = label_to_int[sample['labels'][label_attr]]

    @staticmethod
    def is_integer(value):
        try:
            int_value = int(value)
            if int_value == float(value):
                return True
        except (ValueError, TypeError):
            return False
        return False

    def save(self, export_to, row_divisor = None):

        os.makedirs(os.path.dirname(export_to), exist_ok=True)
        if export_to.endswith('.json'):
            with open(export_to, 'w') as f:
                json.dump(self.data, f, indent=4, cls = JSONsaver)
            return

        rows = []
        for sample in self.data:
            row = {k: v for k, v in sample.items() if k not in ['labels', 'folds']}

            for label_col, label in sample['labels'].items():
                row[label_col] = label

            for fold, fold_assignment in enumerate(sample['folds']):
                row[f'fold_{fold}'] = fold_assignment

            for k, v in row.items():
                if isinstance(v, list) and len(set(v)) == 1:
                    row[k] = v[0]

            if row_divisor is not None:
                for unique_val in sample[row_divisor]:
                    new_row = row.copy()
                    new_row[row_divisor] = unique_val
                    rows.append(new_row)
            else:
                rows.append(row)

        df = pd.DataFrame(rows)
        if 'id' in df.columns:
            df = df.rename(columns={'id': 'case_id'})
        if 'slide_id' in df.columns:
            slide_col = df.pop('slide_id')
            case_idx = df.columns.get_loc('case_id')
            df.insert(case_idx + 1, 'slide_id', slide_col)
        if export_to.endswith('.csv'):
            df.to_csv(export_to, index=False)
        elif export_to.endswith('.tsv'):
            df.to_csv(export_to, sep='\t', index=False)
        else:
            raise ValueError(f'Export path must end in .json, .csv, or .tsv. Received {export_to}.')

    def assign_folds(self, cv_type, num_folds, test_frac=0.1, val_frac=None, stratify=True, predefined_test_ids=None, method='kfold', aggregate_by=None, seed=None):

        if stratify:
            print(f'Assigning {num_folds} folds to split, stratified by {list(self.data[0]["labels"].keys())}.')
        else:
            print(f'Assigning {num_folds} folds to split, not stratified.')

        if aggregate_by is not None:
            print(f'Aggregating samples by {aggregate_by} before splitting.')

        random.seed(seed)
        random.shuffle(self.data)

        if aggregate_by:
            agg_id_to_sample_ids = {}
            for sample in self.data:
                agg_ids = sample[aggregate_by]
                if isinstance(agg_ids, str):
                    agg_ids = [agg_ids]
                assert len(set(agg_ids)) == 1, f'Aggregate attribute {aggregate_by} must have a single unique value per sample. Received {agg_ids}.'
                agg_id = agg_ids[0]
                if agg_id not in agg_id_to_sample_ids:
                    agg_id_to_sample_ids[agg_id] = []
                agg_id_to_sample_ids[agg_id].append(sample['id'])
        else:
            agg_id_to_sample_ids = {sample['id']: [sample['id']] for sample in self.data}

        if stratify:
            id_to_labels = {sample['id']: ''.join(str(label) for label in sample['labels'].values()) for sample in self.data}
            agg_id_to_labels = {}
            for agg_id, sample_ids in agg_id_to_sample_ids.items():
                assert len(set(id_to_labels[sample_id] for sample_id in sample_ids)) == 1, f'Trying to aggregate by {aggregate_by}, but found inconsistent labels for samples with aggregator attribute {agg_id}.'
                agg_id_to_labels[agg_id] = id_to_labels[sample_ids[0]]
        else:
            agg_id_to_labels = {agg_id: 0 for agg_id in agg_id_to_sample_ids.keys()}

        agg_ids = list(agg_id_to_labels.keys())
        agg_labels = list(agg_id_to_labels.values())

        if cv_type == 'fixed_test':
            print("\n=== CV MODE: fixed_test (original behavior) ===")

            if num_folds == 1 or method == 'monte-carlo':
                assert val_frac is not None, 'val_frac must be specified if num_folds == 1 or method == "monte-carlo".'
            else:
                assert val_frac is None, 'val_frac must be None if num_folds > 1 and method == "kfold".'

            if predefined_test_ids is not None:
                test_ids = predefined_test_ids
                train_val_agg_ids = []
                for agg_id, sample_ids in agg_id_to_sample_ids.items():
                    if all(sample_id not in test_ids for sample_id in sample_ids):
                        train_val_agg_ids.append(agg_id)
                    elif all(sample_id in test_ids for sample_id in sample_ids):
                        pass
                    else:
                        raise ValueError(f'Predefined test IDs are incompatible with aggregator attribute {agg_id}.')
                print(f'    Manually assigned {len(test_ids)} test samples out of {len(self.data)}.')
            elif test_frac == 1:
                test_ids = list(set([sample['id'] for sample in self.data]))
                train_val_agg_ids = []
                print(f'    Assigned all {len(test_ids)} samples to test.')
            elif test_frac == 0:
                train_val_agg_ids = list(agg_id_to_labels.keys())
                test_ids = []
                print(f'    Assigned zero samples to test.')
            else:
                train_val_agg_ids, test_agg_ids = train_test_split(list(agg_id_to_labels.keys()), test_size=int(test_frac * len(agg_id_to_labels)), random_state=seed, stratify=list(agg_id_to_labels.values()) if stratify else None)
                test_ids = [id for agg_id in test_agg_ids for id in agg_id_to_sample_ids[agg_id]]
                print(f'    Assigned {len(test_ids)} test samples out of {len(self.data)}.')

            if val_frac == 0:
                val_ids = {f'fold_{fold}': [] for fold in range(num_folds)}
                print(f'    Assigned zero samples to val out of remaining {len(self.data) - len(test_ids)} samples.')
            elif num_folds == 1:
                train_agg_ids, val_agg_ids = train_test_split(train_val_agg_ids, test_size=int(val_frac * len(agg_id_to_labels)), random_state=seed, stratify=[agg_id_to_labels[agg_id] for agg_id in train_val_agg_ids] if stratify else None)
                val_ids = {'fold_0': [id for agg_id in val_agg_ids for id in agg_id_to_sample_ids[agg_id]]}
                print(f'    Assigned {len(val_ids["fold_0"])} val samples out of remaining {len(self.data) - len(test_ids)} samples.')
            else:
                val_ids = {}
                if method == 'monte-carlo':
                    for fold in range(num_folds):
                        new_seed = seed + fold
                        train_agg_ids, val_agg_ids = train_test_split(train_val_agg_ids, test_size=int(val_frac * len(agg_id_to_labels)), random_state=new_seed, stratify=[agg_id_to_labels[agg_id] for agg_id in train_val_agg_ids] if stratify else None)
                        val_ids[f'fold_{fold}'] = [id for agg_id in val_agg_ids for id in agg_id_to_sample_ids[agg_id]]
                    print(f'    Train: {len(self.data) - len(test_ids) - len(val_ids["fold_0"])} samples | Val: {len(val_ids["fold_0"])} samples | Test: {len(test_ids)} samples | Sampling: Monte-Carlo')
                elif method == 'kfold':
                    for fold, (train_index, val_index) in enumerate(StratifiedKFold(num_folds).split(train_val_agg_ids, [agg_id_to_labels[agg_id] for agg_id in train_val_agg_ids])):
                        val_ids[f'fold_{fold}'] = [id for i in val_index for id in agg_id_to_sample_ids[train_val_agg_ids[i]]]
                    print(f'    Train: {len(self.data) - len(test_ids) - len(val_ids["fold_0"])} samples | Val: {len(val_ids["fold_0"])} samples | Test: {len(test_ids)} samples | Sampling: StratifiedKFold')
                else:
                    raise ValueError(f'Sampling method must be "kfold" or "monte-carlo". Received {method}.')

            for sample in self.data:
                sample['folds'] = []
                for fold_idx in range(num_folds):
                    if sample['id'] in test_ids:
                        sample['folds'].append('test')
                    elif sample['id'] in val_ids[f'fold_{fold_idx}']:
                        sample['folds'].append('val')
                    else:
                        sample['folds'].append('train')

            self.data.sort(key=lambda x: x['folds'] + [x['id']])
            self.num_folds = num_folds

        elif cv_type == 'kfold_test':

            print("\n=== CV MODE: kfold_test (Standard K-fold + train-split-val) ===")
            assert val_frac is not None, "val_frac must be provided under kfold_test mode."

            skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

            fold_test_ids = {}
            fold_val_ids = {}

            for fold_idx, (train_index, test_index) in enumerate(skf.split(agg_ids, agg_labels)):

                test_aggs = [agg_ids[i] for i in test_index]
                fold_test_ids[fold_idx] = [
                    id for agg in test_aggs for id in agg_id_to_sample_ids[agg]
                ]

                train_aggs = [agg_ids[i] for i in train_index]

                t_aggs, v_aggs = train_test_split(
                    train_aggs,
                    test_size=val_frac,
                    random_state=seed,
                    stratify=[agg_id_to_labels[a] for a in train_aggs] if stratify else None
                )

                fold_val_ids[fold_idx] = [
                    id for agg in v_aggs for id in agg_id_to_sample_ids[agg]
                ]
            for sample in self.data:
                sid = sample['id']
                sample['folds'] = []

                for fold_idx in range(num_folds):
                    if sid in fold_test_ids[fold_idx]:
                        sample['folds'].append('test')
                    elif sid in fold_val_ids[fold_idx]:
                        sample['folds'].append('val')
                    else:
                        sample['folds'].append('train')

            self.data.sort(key=lambda x: x['folds'] + [x['id']])
            self.num_folds = num_folds

        else:
            raise ValueError(f"Unknown cv_type {cv_type}. Use 'fixed_test' or 'kfold_test'.")

    def remove_all_folds(self):

        print('Removing all fold assignments.')
        for sample in self.data:
            sample['folds'] = []
        self.num_folds = 0

    def replace_folds(self, replace_from, replace_to, selected_ids = None, selected_folds = None):

        reassigned_samples_per_fold = {fold_idx: 0 for fold_idx in range(self.num_folds)}
        for sample in self.data:
            if selected_ids is not None and sample['id'] not in selected_ids:
                continue
            for fold_idx, current_assignment in enumerate(sample['folds']):
                if selected_folds is not None and fold_idx not in selected_folds:
                    continue
                if current_assignment == replace_from:
                    sample['folds'][fold_idx] = replace_to
                    reassigned_samples_per_fold[fold_idx] += 1

        for fold_idx, num_reassigned in reassigned_samples_per_fold.items():
            if num_reassigned > 0:
                print(f'Reassigned {num_reassigned} samples in fold {fold_idx} from {replace_from} to {replace_to}.')

    def visualize_folds(self, saveto=None):

        assert self.num_folds > 0

        def normalize(f):
            if f is None:
                return "train"
            return (
                f.strip().lower()
                .replace("\u200b", "")
                .replace("\ufeff", "")
                .replace("\t", "")
                .replace("\r", "")
                .replace("\n", "")
            )

        for sample in self.data:
            sample["folds"] = [normalize(f) for f in sample["folds"]]

        valid_labels = {"train", "val", "test"}
        for sample in self.data:
            sample["folds"] = [f if f in valid_labels else "train" for f in sample["folds"]]

        data_to_plot = sorted(self.data, key=lambda x: x["folds"] + [x["id"]])
        state_map = {"train": 0, "val": 1, "test": 2}

        matrix = np.array([
            [state_map[sample["folds"][f]] for sample in data_to_plot]
            for f in range(self.num_folds)
        ])

        num_samples = matrix.shape[1]
        fig_width = max(8, num_samples * 0.12)
        fig_height = max(2.5, self.num_folds * 0.45)

        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111)

        cmap = ListedColormap([
            "#4DBBD5",
            "#CDDDAA",
            "#E64B35",
        ])

        ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")

        for x in range(num_samples):
            ax.axvline(x - 0.5, color="black", linewidth=0.3)
        for y in range(self.num_folds):
            ax.axhline(y - 0.5, color="black", linewidth=0.6)

        ax.set_yticks(range(self.num_folds))
        ax.set_yticklabels([f"fold_{i}" for i in range(self.num_folds)], fontsize=12)
        ax.set_xticks([])

        ax.set_xlabel("Sample Index", fontsize=13)
        ax.set_ylabel("Fold", fontsize=13)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("black")

        legend_patches = [
            mpatches.Patch(color="#4DBBD5", label="Train"),
            mpatches.Patch(color="#CDDDAA", label="Validation"),
            mpatches.Patch(color="#E64B35", label="Test"),
        ]

        ax.legend(
            handles=legend_patches,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=12,
            frameon=False
        )

        fig.tight_layout(pad=0.2)

        if saveto:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            fig.savefig(saveto, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def visualize_labels(self, fold=0, saveto=None):


        data_to_plot = sorted(self.data, key=lambda x: x["folds"] + [x["id"]])
        label_attrs = list(data_to_plot[0]["labels"].keys())

        fig_width = max(6 * len(label_attrs), 6)
        fig_height = 5
        fig, axs = plt.subplots(1, len(label_attrs), figsize=(fig_width, fig_height))
        if len(label_attrs) == 1:
            axs = [axs]

        mode_colors = {
            "train": "#4DBBD5",
            "val": "#CDDDAA",
            "test": "#E64B35",
        }

        mode_order = ["train", "val", "test"]

        for idx, label_attr in enumerate(label_attrs):

            ax = axs[idx]
            modes_present = [m for m in mode_order if any(s["folds"][fold] == m for s in data_to_plot)]

            label_counts = {}
            all_labels = set()

            for mode in modes_present:
                labels = [
                    sample["labels"][label_attr]
                    for sample in data_to_plot
                    if sample["folds"][fold] == mode
                ]
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_counts[mode] = dict(zip(unique_labels, counts))
                all_labels.update(unique_labels)

            all_labels = sorted(all_labels)
            x = np.arange(len(all_labels))
            width = 0.22

            for i, mode in enumerate(modes_present):
                counts = np.array([label_counts[mode].get(lbl, 0) for lbl in all_labels])
                total = counts.sum()
                frac = counts / total if total > 0 else np.zeros_like(counts)

                ax.bar(
                    x + i * width,
                    frac,
                    width,
                    label=mode,
                    color=mode_colors[mode],
                    edgecolor="black",
                    linewidth=0.8
                )
                for j, (f, c) in enumerate(zip(frac, counts)):
                    if c > 0:
                        ax.text(
                            x[j] + i * width,
                            f + 0.01,
                            str(c),
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            rotation=90
                        )

            ax.set_xticks(x + width * len(modes_present) / 3)
            ax.set_xticklabels(all_labels, fontsize=11)

            ax.set_xlabel(label_attr, fontsize=13)
            ax.set_ylabel("Fraction of samples", fontsize=13)
            ax.set_title(f"Label Distribution (Fold {fold})", fontsize=14)

            for spine in ax.spines.values():
                spine.set_linewidth(2)

            ax.grid(False)

            legend_patches = [Patch(color=mode_colors[m], label=m.capitalize()) for m in modes_present]
            ax.legend(
                handles=legend_patches,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=11
            )

        fig.tight_layout(pad=0.3)

        if saveto:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            fig.savefig(saveto, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def get_label_distribution(self, label_attr):

        unique_classes = self.unique_classes(label_attr)
        label_distribution = {label: 0 for label in unique_classes}

        for sample in self.data:
            label_distribution[sample['labels'][label_attr]] += 1

        return label_distribution

    def get_sizes(self, fold):
        sizes = {}
        for mode in ['train', 'val', 'test']:
            sizes[mode] = len(self.get_ids(fold, mode))
        return sizes

    def get_ids(self, fold, mode = None):

        if mode is None:
            return [sample['id'] for sample in self.data]
        else:
            return [sample['id'] for sample in self.data if sample['folds'][fold] == mode]

    def get_ratios(self, fold):
        sizes = self.get_sizes(fold)
        total = sum(sizes.values())
        ratios = {mode: f'{(size / total):.2f}' for mode, size in sizes.items()}
        return ratios

    def unique_classes(self, label_attr):
        return set(self.y(label_attr))

    def y(self, label_attr, fold = None, mode = None):
        assert (fold is not None and mode is not None) or (fold is None and mode is None), 'Either both or none of fold and mode must be provided.'

        if fold is not None and mode is not None:
            return np.array([sample['labels'][label_attr] for sample in self.data if sample['folds'][fold] == mode])
        else:
            return np.array([sample['labels'][label_attr] for sample in self.data])
