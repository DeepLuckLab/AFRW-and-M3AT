import os
import yaml
from datasets.data_split import DataSplit
import pandas as pd
COMBINE_TRAIN_VAL = False

def get_split_and_task_info(path_to_split, path_to_config):


    assert os.path.exists(path_to_split), f"Path to split {path_to_split} does not exist locally."

    assert os.path.exists(path_to_config), f"Path to split config {path_to_config} does not exist locally."


    task_info = get_task_info(path_to_config=path_to_config)

    split = DataSplit(path = path_to_split,
                    id_col = task_info['sample_col'],
                    attr_cols = task_info['extra_cols'] + ['slide_id'],
                    label_cols = [task_info['task_col']])

    if COMBINE_TRAIN_VAL:
        split.replace_folds('val', 'train')

    df = pd.read_csv(path_to_split, sep='\t')
    fold_columns_exist = any(col.startswith('fold_') for col in df.columns)

    if not fold_columns_exist:
        raise RuntimeError(
            "\n\n============================== ERROR ==============================\n"
            "当前 split 文件中未检测到 fold_0, fold_1 ... 等折叠划分字段。\n"
            "请先运行 get_split_sh.py 对数据集进行划分，再重新运行训练。\n"
            "===============================#####=================================\n"
        )
    else:
        print("Fold columns already exist, using existing folds.")
    return split, task_info

def get_task_info(path_to_config = None):

    if path_to_config is None:
        raise ValueError("path_to_config is required")

    with open(path_to_config, 'r') as task_info:
        task_info = yaml.safe_load(task_info)

    assert 'sample_col' in task_info and isinstance(task_info['sample_col'], str), f"sample_col (str) not found in task config at {path_to_config}."
    assert 'extra_cols' in task_info and isinstance(task_info['extra_cols'], list), f"extra_cols (list[str]) not found in task config at {path_to_config}."
    assert 'label_dict' in task_info and isinstance(task_info['label_dict'], dict), f"label_dict (dict) not found in task config at {path_to_config}."
    assert 'metrics' in task_info and isinstance(task_info['metrics'], list), f"metrics (list[str]) not found in task config at {path_to_config}."

    return task_info