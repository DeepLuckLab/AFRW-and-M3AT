from datasets.data_split import DataSplit
from utils.get_split_task_utils import get_task_info
import os


def main(args):

    path_to_split = args.split
    path_to_config = args.task_config


    assert os.path.exists(
        path_to_split
    ), f"Path to split {path_to_split} does not exist locally."

    assert os.path.exists(
        path_to_config
    ), f"Path to split config {path_to_config} does not exist locally."


    task_info = get_task_info(path_to_config=path_to_config)

    split = DataSplit(
        path=path_to_split,
        id_col=task_info["sample_col"],
        attr_cols=task_info["extra_cols"] + ["slide_id"],
        label_cols=[task_info["task_col"]],
    )


    print("\n=== Creating new folds... ===")


    split.remove_all_folds()

    split.assign_folds(
        cv_type=args.cv_type,
        num_folds=args.num_folds,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        stratify=True,
        seed=args.seed,
    )


    if args.combine_train_val:
        split.replace_folds("val", "train")

    dir_path = os.path.dirname(path_to_split)
    new_k_path = os.path.join(dir_path, "k=all_new.tsv")
    split.save(new_k_path, row_divisor="slide_id")


    split.visualize_folds(os.path.join(dir_path, "folds_split.png"))

    split.visualize_labels(fold=0, saveto=os.path.join(dir_path, "labels_split.png"))

    print(f"Fold assignment completed. New split saved to:\n{new_k_path}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        default="test_task/t1_vs_t2/k=all.tsv",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="test_task/t1_vs_t2/config.yaml",
    )
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument(
        "--cv_type", type=str, default="kfold_test", help="fixed_test or kfold_test"
    )
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--combine_train_val", action="store_true")
    parser.add_argument("--seed", type=int, default=112)

    args = parser.parse_args()
    main(args)
