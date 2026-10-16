import argparse
from utils.prepare_utils import prepare_function, set_seed
from utils.core.core_utilis import train_val_main, evaluation
from metrics.statistical_metrics import print_summary
import os

def main(args):

    set_seed(args.seed)
    model_kwargs, dataset = prepare_function(args)

    if args.mode == "train":
        if args.adversarial_training:
            train_val_main(args, model_kwargs, dataset)
            val_summary = evaluation(args, model_kwargs, dataset, mode="val", clean_data=False)
            test_summary_adv = evaluation(args, model_kwargs, dataset, mode="test", clean_data=False)

            test_summary_clean = evaluation(args, model_kwargs, dataset, mode="test", clean_data=True)
            return val_summary, test_summary_clean, test_summary_adv
        else:
            train_val_main(args, model_kwargs, dataset)
            val_summary = evaluation(args, model_kwargs, dataset, mode="val", clean_data=True)
            test_summary_clean = evaluation(args, model_kwargs, dataset, mode="test", clean_data=True)
            return val_summary, test_summary_clean, None

    elif args.mode == "test":
        if args.adversarial_training:
            test_summary_clean = evaluation(args, model_kwargs, dataset, mode="test", clean_data=True)
            test_summary_adv = evaluation(args, model_kwargs, dataset, mode="test", clean_data=False)
            return test_summary_clean, test_summary_adv
        else:
            test_summary_clean = evaluation(args, model_kwargs, dataset, mode="test", clean_data=True)
        return test_summary_clean, None

    else:
        raise "mode must be test or train"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MedOmniCore")

    parser.add_argument(
        "--task_type", type=str, default="classification",
        help="classification or survival"
    )
    parser.add_argument("--model_name", type=str, default="abmil")
    parser.add_argument("--train_source", type=str, default="test_task")

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--adversarial_training", action="store_true")
    parser.add_argument("--perturb_config", type=str, default='pgd_at')
    parser.add_argument("--adv_training_types", type=str, default='at', help='at')
    parser.add_argument("--adv_eps", type=float, default=1)
    parser.add_argument("--adv_alpha", type=float, default=0.2)
    parser.add_argument("--adv_steps", type=int, default=5)

    parser.add_argument("--split", type=str, default="test_task/t1_vs_t2/k=all.tsv")
    parser.add_argument("--task_config", type=str, default="test_task/t1_vs_t2/config.yaml")
    parser.add_argument("--slide_merge_mode", type=str, default='list')
    parser.add_argument("--slide_embeddings_dirs", type=str, default=None)
    parser.add_argument("--omics_embeddings_dirs", type=str, default=None)
    parser.add_argument("--results_dir_root", type=str, default="./mil_results")

    parser.add_argument(
        "--modalities", type=str, default="slide",
        choices=["slide", "omics", "all"]
    )

    parser.add_argument("--input_feature_dim", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--attn_mask", type=int, default=None)

    parser.add_argument("--return_raw_attention", action="store_true")
    parser.add_argument("--return_intermediate_feats", action="store_true")

    parser.add_argument(
        "--bag_size",
        type=lambda x: int(x) if x.isdigit() else None,
        default=None,
        help="Number of instances per bag; use None for all instances",
    )
    parser.add_argument(
        "--balanced", action="store_true",
        help="Enable balanced sampling"
    )

    parser.add_argument("--base_learning_rate", type=float, default=0.00001)
    parser.add_argument("--layer_decay", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.00001)

    parser.add_argument("--optimizer_type", type=str, default="AdamW")
    parser.add_argument(
        "--scheduler_type", type=str, default="cosine",
        help="cosine or None"
    )

    parser.add_argument("--gradient_accumulation", type=int, default=1, help="1")
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1, help="1")

    parser.add_argument(
        "--checkpoint_strategy", type=str,
        default="best-val-loss",
        help="best-val-loss / last / none"
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=False,
        help="Enable early stopping"
    )

    parser.add_argument("--view_progress", type=str, default="bar")

    args = parser.parse_args()

    settings = {
        "model_name": args.model_name,
        "train_source": args.train_source,
        "task_type": args.task_type,
        "bag_size": args.bag_size,
        "base_learning_rate": args.base_learning_rate,
        "layer_decay": args.layer_decay,
        "gradient_accumulation": args.gradient_accumulation,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,

        "scheduler_type": args.scheduler_type,
        "optimizer_type": args.optimizer_type,
        "balanced": args.balanced,
        "checkpoint_strategy": args.checkpoint_strategy,
        "early_stopping": args.early_stopping,
        "input_feature_dim": args.input_feature_dim,
        "embed_dim": args.embed_dim,
        "modalities": args.modalities,
        "training": "Adversarial" if args.adversarial_training else "Standard",
        "perturb_config": args.perturb_config if args.adversarial_training else None,
        "slide_embeddings_dirs": args.slide_embeddings_dirs,
        "omics_embeddings_dirs": args.omics_embeddings_dirs,
        "results_dir_root": args.results_dir_root,
    }

    args.results_dir = os.path.join(args.results_dir_root, args.model_name)
    os.makedirs(args.results_dir, exist_ok=True)

    with open(f"{args.results_dir}/experiment.txt", "w") as f:
        f.write(str(settings) + "\n")

    print_summary(title="Experimental Setup")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))

    print_summary(title="Experiment Begins")

    if args.mode == "train":

        val_summary, test_summary_clean, test_summary_adv = main(args)

        if val_summary is not None:
            print_summary(
                title="Validation set results",
                summary=val_summary,
                table=True
            )

        if test_summary_clean is not None:
            print_summary(
                title="Clean Test set results",
                summary=test_summary_clean,
                table=True
            )

        if test_summary_adv is not None:
            print_summary(
                title="Adv Test set results",
                summary=test_summary_adv,
                table=True
            )



    else:
        test_summary_clean, test_summary_adv = main(args)
        if test_summary_clean is not None:
            print_summary(
                title="Clean Test set results",
                summary=test_summary_clean,
                table=True
            )

        if test_summary_adv is not None:
            print_summary(
                title="Adv Test set results",
                summary=test_summary_adv,
                table=True
            )

    print_summary(title="End of experiment")